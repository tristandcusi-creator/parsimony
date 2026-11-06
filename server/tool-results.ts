// Short-lived overflow for agents tool results: authorize the live turn first; ownership is both bot and conversation.
import { createHash, randomUUID } from "node:crypto";
import { closeSync, fstatSync, mkdirSync, openSync, readSync, readdirSync, realpathSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { basename, join, resolve, sep } from "node:path";
import { redactSecretsInText } from "../shared/redact.ts";

export const TOOL_RESULT_PREVIEW_CHARS = 16_000;
export const TOOL_RESULT_MAX_CHARS = 128 * 1024;
export const TOOL_RESULT_TTL_MS = 60 * 60_000;
const MAX_RESULTS = 128;
const MAX_RESULTS_PER_OWNER = 16;
const MAX_BYTES = 16 * 1024 * 1024;
const MAX_OWNER_BYTES = 2 * 1024 * 1024;

export function toolResultPrefix(text: string, chars: number): string {
  const prefix = text.slice(0, chars);
  const last = prefix.charCodeAt(prefix.length - 1);
  return last >= 0xd800 && last <= 0xdbff ? prefix.slice(0, -1) : prefix;
}

// ponytail: one shared byte-boundary truncator. Header 156-164 and turn-context 117-143 both route through here. ceil: revisit if char-based tool Result previews diverge.
const isContinuation = (b: number): boolean => (b & 0xc0) === 0x80;
const codePointLen = (first: number): number => (first < 0x80 ? 1 : first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4);
function byteBoundaryEnd(buf: Buffer, max: number): number {
  const limit = Math.min(max, buf.length);
  if (limit <= 0) return 0;
  let lead = limit - 1;
  while (lead > 0 && isContinuation(buf[lead]!)) lead -= 1;
  if (isContinuation(buf[lead]!)) return 0;
  const needed = codePointLen(buf[lead]!);
  return lead + needed <= limit ? limit : lead;
}
export function truncateBytes(text: string, maxBytes: number): string {
  const max = Number.isFinite(maxBytes) && maxBytes >= 0 ? Math.floor(maxBytes) : 0;
  if (max === 0) return "";
  const buf = Buffer.from(text, "utf8");
  if (buf.length <= max) return text;
  const end = byteBoundaryEnd(buf, max);
  return end <= 0 ? "" : buf.subarray(0, end).toString("utf8");
}

const MAX_SPILL_BYTES = 1_500_000;
const TOOL_RESULT_MAX_BYTES = MAX_SPILL_BYTES;
const sha256Hex = (text: string): string => createHash("sha256").update(text).digest("hex");

export interface SpilledToolOutput {
  preview: string;
  path: string;
  hash: string;
  size: number;
  originalLength: number;
  originalBytes: number;
  truncated: boolean;
  ok: boolean;
  error: string | null;
  exitCode: number | null;
}
export interface SpillPage {
  text: string;
  offset: number;
  nextOffset: number;
  size: number;
  eof: boolean;
}
const spillHashCache = new Map<string, { value: string; at: number }>();
export function clearSpillHashCache(): void { spillHashCache.clear(); }
export function spillHashCacheSize(): number { return spillHashCache.size; }
function setHashCache(path: string, value: string): void {
  spillHashCache.delete(path);
  spillHashCache.set(path, { value, at: Date.now() });
  while (spillHashCache.size > 128) {
    const oldest = spillHashCache.keys().next().value;
    if (oldest === undefined) break;
    spillHashCache.delete(oldest);
  }
}

let lastPruneAt = 0;
const PRUNE_THROTTLE_MS = 60_000;
export function spillPathRegex(): RegExp {
  return /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.txt$/;
}
export function pruneSpills(dir: string): void {
  const now = Date.now();
  let entries: string[];
  try { entries = readdirSync(dir); } catch { return; }
  const withDates: Array<{ name: string; mtimeMs: number; size: number }> = [];
  for (const name of entries) {
    if (!spillPathRegex().test(name)) continue;
    try {
      const st = statSync(join(dir, name));
      if (!st.isFile()) continue;
      if (now - st.mtimeMs > TOOL_RESULT_TTL_MS) {
        unlinkSync(join(dir, name));
        spillHashCache.delete(join(dir, name));
        continue;
      }
      withDates.push({ name, mtimeMs: st.mtimeMs, size: st.size });
    } catch { continue; }
  }
  withDates.sort((a, b) => a.mtimeMs - b.mtimeMs);
  let bytes = 0;
  for (const entry of withDates) bytes += entry.size;
  for (const entry of withDates) {
    if (bytes <= TOOL_RESULT_MAX_BYTES) break;
    try {
      unlinkSync(join(dir, entry.name));
      spillHashCache.delete(join(dir, entry.name));
      bytes -= entry.size;
    } catch { continue; }
  }
}
function maybePrune(dir: string): void {
  const now = Date.now();
  if (now - lastPruneAt < PRUNE_THROTTLE_MS) return;
  lastPruneAt = now;
  pruneSpills(dir);
}
export function spillToolOutput(dir: string, output: string, ok: boolean, error: string | null, previewBytes: number, exitCode?: number): SpilledToolOutput {
  const finite = Number.isFinite(previewBytes) && previewBytes >= 0 ? Math.floor(previewBytes) : 0;
  const redacted = redactSecretsInText(output);
  const originalBytes = Buffer.byteLength(redacted, "utf8");
  const originalLength = redacted.length;
  // ponytail: unbounded write capped to 1.5MB via byte-boundary truncator so write <= read.
  const storedText = truncateBytes(redacted, MAX_SPILL_BYTES);
  const stored = Buffer.byteLength(storedText, "utf8");
  const hash = sha256Hex(storedText);
  const rel = `${hash}.txt`;
  const path = join(dir, rel);
  const size = originalBytes;
  const truncated = stored !== size;
  if (dir) {
    try {
      mkdirSync(dir, { recursive: true, mode: 0o700 });
    } catch (e) {
      throw new Error(`spill dir not writable: ${String(e)}`);
    }
    try {
      writeFileSync(path, storedText, { flag: "w", mode: 0o600 });
    } catch (e) {
      throw new Error(`spill write failed: ${String(e)}`);
    }
    // ponytail: hash is verified on first read; cache alone is not trusted after tamper (re-hash if mtime/size changed). ceil: O(1) mtime check.
    setHashCache(path, hash);
    maybePrune(dir);
  }
  const marker = `\n[full output ${originalLength} chars (${originalBytes} bytes); at ${path}; hash ${hash}; stored ${stored} chars (${stored} bytes${truncated ? "; truncated to 1.5MB cap" : ""}); ok=${ok}; error=${JSON.stringify(error)}; exitCode=${exitCode ?? "null"}; read with byte offsets 0..${stored}]`;
  const markerBytes = Buffer.byteLength(marker, "utf8");
  const headBudget = Math.max(0, finite - markerBytes);
  const preview = markerBytes > finite ? marker : truncateBytes(redacted, headBudget) + marker;
  return { preview, path, hash, size: stored, originalLength, originalBytes, truncated, ok, error, exitCode: exitCode ?? null };
}

function assertInside(dir: string, rel: string): string {
  if (!rel) throw new Error("invalid spill reference");
  const resolved = resolve(dir, rel);
  const root = resolve(dir);
  if (resolved === root) throw new Error("invalid spill reference");
  if (!resolved.startsWith(`${root}${sep}`)) throw new Error("spill reference escapes spill dir");
  if (!spillPathRegex().test(basename(resolved))) throw new Error("invalid spill filename");
  let realRoot: string;
  let realPath: string;
  try {
    realRoot = realpathSync(dir);
    realPath = realpathSync(resolved);
  } catch {
    return resolved;
  }
  if (realPath !== realRoot && !realPath.startsWith(`${realRoot}${sep}`)) throw new Error("spill reference escapes spill dir");
  return resolved;
}

export function readSpilledToolOutput(dir: string, rel: string, offset: number, limit: number, hash?: string, maxBytes = 750_000): SpillPage {
  const path = assertInside(dir, rel);
  const off = Number.isFinite(offset) && offset >= 0 ? Math.floor(offset) : 0;
  if (!Number.isFinite(limit) || limit < 0 || limit > 8_000) throw new Error("limit out of range");
  if (!Number.isFinite(maxBytes) || maxBytes <= 0 || maxBytes > 1_500_000) throw new Error("maxBytes out of range");
  let fd: number;
  try {
    fd = openSync(path, "r");
  } catch {
    throw new Error("spill file not found");
  }
  try {
    const size = fstatSync(fd).size;
    if (size > 1_500_000) throw new Error("spill file too large");
    const expected = typeof hash === "string" && hash ? hash : basename(path, ".txt");
    // ponytail: never trust cache alone; re-hash always (ceil: mtime check). cache only avoids repeat in same process.
    const hasher = createHash("sha256");
    const chunk = Buffer.alloc(64 * 1024);
    for (let pos = 0; pos < size;) {
      const n = readSync(fd, chunk, 0, Math.min(chunk.length, size - pos), pos);
      if (n <= 0) break;
      hasher.update(chunk.subarray(0, n));
      pos += n;
    }
    const computed = hasher.digest("hex");
    if (computed !== expected) throw new Error(`ref "${rel}" does not match stored hash; evidence may be tampered`);
    setHashCache(path, computed);
    const clamped = Math.min(off, size);
    if (clamped >= size) return { text: "", offset: clamped, nextOffset: size, size, eof: true };
    if (limit === 0) return { text: "", offset: clamped, nextOffset: clamped, size, eof: false };
    const sliceLen = Math.min(limit, size - clamped, maxBytes);
    const buf = Buffer.alloc(sliceLen);
    let filled = 0;
    while (filled < sliceLen) {
      const n = readSync(fd, buf, filled, sliceLen - filled, clamped + filled);
      if (n <= 0) break;
      filled += n;
    }
    const slice = buf.subarray(0, filled);
    if (filled > 0 && isContinuation(slice[0]!)) throw new Error("offset lands mid-codepoint");
    const end = byteBoundaryEnd(slice, slice.length);
    if (end <= 0) {
      if (filled <= 0) throw new Error("spill file changed during read");
      const needed = codePointLen(slice[0]!);
      if (needed > maxBytes) throw new Error("maxBytes too small for one codepoint");
      if (clamped + needed > size) throw new Error("spill file ends mid-codepoint");
      const extra = Buffer.alloc(needed);
      let got = 0;
      while (got < extra.length) {
        const n = readSync(fd, extra, got, extra.length - got, clamped + got);
        if (n <= 0) break;
        got += n;
      }
      if (got !== needed) throw new Error("spill file ends mid-codepoint");
      const nextOffset = clamped + got;
      return { text: extra.toString("utf8"), offset: clamped, nextOffset, size, eof: nextOffset >= size };
    }
    const nextOffset = Math.min(size, clamped + end);
    return { text: slice.subarray(0, end).toString("utf8"), offset: clamped, nextOffset, size, eof: nextOffset >= size };
  } finally {
    try { closeSync(fd); } catch { /* already closed */ }
  }
}

type Owner = { botId: string; threadId: string };
type SavedResult = Owner & { text: string; bytes: number; expiresAt: number; truncated: boolean };

export class ToolResults {
  private readonly results = new Map<string, SavedResult>();
  private readonly now: () => number;

  constructor(now: () => number = Date.now) { this.now = now; }

  private expire(): void {
    const now = this.now();
    for (const [id, result] of this.results) {
      if (result.expiresAt <= now) this.results.delete(id);
    }
  }

  save(owner: Owner, text: string, truncated = false) {
    this.expire();
    // Redact before taking the prefix, including a secret crossing its edge.
    const redacted = redactSecretsInText(text);
    const bounded = toolResultPrefix(redacted, TOOL_RESULT_MAX_CHARS);
    const result = { botId: owner.botId, threadId: owner.threadId, text: bounded, bytes: Buffer.byteLength(bounded),
      expiresAt: this.now() + TOOL_RESULT_TTL_MS, truncated: truncated || bounded.length < redacted.length };
    const id = `r-${randomUUID()}`;
    this.results.set(id, result);
    for (const ownOnly of [true, false]) {
      const entries = [...this.results].filter(([, entry]) => !ownOnly ||
        (entry.botId === owner.botId && entry.threadId === owner.threadId));
      let bytes = entries.reduce((sum, [, entry]) => sum + entry.bytes, 0);
      let count = entries.length;
      for (const [oldId, entry] of entries) {
        if (bytes <= (ownOnly ? MAX_OWNER_BYTES : MAX_BYTES) && count <= (ownOnly ? MAX_RESULTS_PER_OWNER : MAX_RESULTS)) break;
        this.results.delete(oldId);
        bytes -= entry.bytes;
        count--;
      }
    }
    return { id, length: bounded.length, truncated: result.truncated, expiresAt: result.expiresAt };
  }

  read(owner: Owner, id: string, offset: number) {
    this.expire();
    const result = this.results.get(id);
    if (!result || result.botId !== owner.botId || result.threadId !== owner.threadId) return null;
    if (!Number.isSafeInteger(offset) || offset < 0 || offset > result.text.length) return null;
    // A supplied offset may point to the low half of a surrogate pair.
    const char = result.text.charCodeAt(offset);
    const start = char >= 0xdc00 && char <= 0xdfff ? Math.max(0, offset - 1) : offset;
    const text = toolResultPrefix(result.text.slice(start), TOOL_RESULT_PREVIEW_CHARS);
    return { id, text, offset: start, nextOffset: start + text.length, length: result.text.length,
      truncated: result.truncated, expiresAt: result.expiresAt };
  }
}
