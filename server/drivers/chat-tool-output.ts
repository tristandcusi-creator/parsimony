// Content-addressed spill store for tool outputs too large for the model context.
// The full received text is written once under dir/<sha256(text)>.txt and addressed
// by a 64-hex ref; a preview with a reader reference is returned to the model instead.
import { createHash } from "node:crypto";
import { lstatSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { writeFileAtomic } from "../atomic.ts";
import { DEFAULT_RESULT_BUDGET } from "../mcp-trim.ts";
import { pruneSpills, TOOL_RESULT_TTL_MS } from "../tool-results.ts";

const MAX_DIRECT = DEFAULT_RESULT_BUDGET;
const HEAD_OK = 6000;
const TAIL_OK = 1500;
const HEAD_COMPACT = 400;
const TAIL_COMPACT = 400;
const HEAD_ERR = 1000;
const TAIL_ERR = 1000;
const READER_LIMIT_MAX = DEFAULT_RESULT_BUDGET;
const READER_LIMIT_DEFAULT = Math.floor(DEFAULT_RESULT_BUDGET / 2);
const REF_RE = /^[0-9a-f]{64}$/;

export interface ChatToolDefinition {
  type: "function";
  function: { name: string; description: string; parameters: Record<string, unknown> };
}

export const READ_TOOL_OUTPUT: ChatToolDefinition = {
  type: "function",
  function: {
    name: "read_tool_output",
    description: `Read a page of a previously spilled tool output by its content-addressed ref. Results expire after ${Math.ceil(TOOL_RESULT_TTL_MS / 60_000)} minutes or under cache pressure.`,
    parameters: {
      type: "object",
      properties: {
        ref: { type: "string", description: "The 64-hex ref returned with a truncated tool result." },
        offset: { type: "integer", minimum: 0, description: "Character offset to start reading from (default 0)." },
        limit: { type: "integer", minimum: 1, maximum: READER_LIMIT_MAX, description: `Max characters to return (default ${READER_LIMIT_DEFAULT}).` },
      },
      required: ["ref"],
      additionalProperties: false,
    },
  },
};

export interface PreparedToolOutput {
  text: string;
  compactText?: string;
  ref?: string;
}

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function refOf(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

function label(ref: string, total: number): string {
  return `\n\n[tool output stored as ref "${ref}" (${total} chars total). Call read_tool_output({ ref: "${ref}", offset, limit }) to read it in pages.]`;
}

function codePointLength(text: string): number {
  let count = 0;
  for (let index = 0; index < text.length; count++) index += text.codePointAt(index)! > 0xffff ? 2 : 1;
  return count;
}

function codePointOffset(text: string, count: number, start = 0): number {
  let index = start;
  for (let remaining = count; remaining > 0 && index < text.length; remaining--) {
    index += text.codePointAt(index)! > 0xffff ? 2 : 1;
  }
  return index;
}

function codePointStart(text: string, count: number): number {
  let index = text.length;
  for (let remaining = count; remaining > 0 && index > 0; remaining--) {
    const last = text.charCodeAt(index - 1);
    const prior = index > 1 ? text.charCodeAt(index - 2) : 0;
    index -= prior >= 0xd800 && prior <= 0xdbff && last >= 0xdc00 && last <= 0xdfff ? 2 : 1;
  }
  return index;
}

function codePointRangeLength(text: string, start: number, end: number): number {
  let count = 0;
  for (let index = start; index < end; count++) index += text.codePointAt(index)! > 0xffff ? 2 : 1;
  return count;
}

/** Head+tail at Unicode code-point boundaries, returning the whole text when it fits. */
function headTail(text: string, head: number, tail: number): string {
  if (text.length <= head + tail) return text;
  const end = codePointOffset(text, head);
  const start = codePointStart(text, tail);
  if (start <= end) return text;
  const omitted = codePointRangeLength(text, end, start);
  return `${text.slice(0, end)}\n... [${omitted} characters omitted] ...\n${text.slice(start)}`;
}

/** Persist the complete received text once under dir/<sha256(text)>.txt (0700 dir, 0600 file),
 * never rewriting an existing file. When the target already exists it must be a normal file
 * (not a symlink) with byte-identical content to be reused; a symlink or mismatched artifact
 * is refused and never followed or overwritten. Returns the 64-hex ref. Persists even small text. */
export function persistToolOutput(dir: string, text: string): string {
  const ref = refOf(text);
  const path = join(dir, `${ref}.txt`);
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  pruneSpills(dir);
  let existing: string | undefined;
  try {
    const stat = lstatSync(path);
    if (stat.isSymbolicLink()) throw new Error(`ref "${ref}" is a symlink`);
    if (!stat.isFile()) throw new Error(`ref "${ref}" is not a file`);
    existing = readFileSync(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  if (existing !== undefined) {
    if (refOf(existing) !== ref) throw new Error(`ref "${ref}" exists with mismatched content`);
    return ref;
  }
  writeFileAtomic(path, text, { mode: 0o600 });
  pruneSpills(dir, [`${ref}.txt`]);
  return ref;
}

/** Prepare a tool result for the model context. Text at or under 8000 chars passes through.
 * Otherwise the full text is spilled once and a head/tail preview plus reader label is returned.
 * For !ok (diagnostics) keep head/tail 1000 and still include the ref so the full stored text
 * is retrievable. A large spill with no output directory is an explicit error; callers use
 * legacy boundedText when the reader is off. */
export function prepareToolOutput(dir: string | undefined, text: string, ok: boolean): PreparedToolOutput {
  if (text.length <= MAX_DIRECT) return { text };
  const total = codePointLength(text);
  if (total <= MAX_DIRECT) return { text };
  if (!dir) throw new Error("no output directory; cannot spill large tool output");
  const ref = persistToolOutput(dir, text);
  const head = ok ? HEAD_OK : HEAD_ERR;
  const tail = ok ? TAIL_OK : TAIL_ERR;
  return {
    text: headTail(text, head, tail) + label(ref, total),
    compactText: headTail(text, HEAD_COMPACT, TAIL_COMPACT) + label(ref, total),
    ref,
  };
}

export interface ReaderPage {
  range: { offset: number; limit: number };
  total: number;
  eof: boolean;
  text: string;
}
export function validateToolOutputArguments(value: unknown): { ref: string; offset: number; limit: number } {
  if (!object(value)) throw new Error("read_tool_output arguments must be a JSON object");
  for (const key of Object.keys(value)) {
    if (key !== "ref" && key !== "offset" && key !== "limit") {
      throw new Error(`unknown argument: ${key}`);
    }
  }
  if (typeof value.ref !== "string" || !REF_RE.test(value.ref)) {
    throw new Error("ref must be a 64-hex string");
  }
  let offset = 0;
  if (value.offset !== undefined) {
    if (typeof value.offset !== "number" || !Number.isInteger(value.offset) || value.offset < 0) {
      throw new Error("offset must be an integer >= 0");
    }
    offset = value.offset;
  }
  let limit = READER_LIMIT_DEFAULT;
  if (value.limit !== undefined) {
    if (typeof value.limit !== "number" || !Number.isInteger(value.limit) || value.limit < 1 || value.limit > READER_LIMIT_MAX) {
      throw new Error(`limit must be an integer from 1 to ${READER_LIMIT_MAX}`);
    }
    limit = value.limit;
  }
  return { ref: value.ref, offset, limit };
}

export function readToolOutput(dir: string, args: unknown): { ok: boolean; text: string } {
  const { ref, offset, limit } = validateToolOutputArguments(args);
  const path = join(dir, `${ref}.txt`);
  let stat: ReturnType<typeof lstatSync>;
  try {
    stat = lstatSync(path);
  } catch {
    throw new Error(`no stored output for ref "${ref}"`);
  }
  if (stat.isSymbolicLink()) throw new Error(`ref "${ref}" is a symlink`);
  if (!stat.isFile()) throw new Error(`ref "${ref}" is not a file`);
  if (Date.now() - stat.mtimeMs > TOOL_RESULT_TTL_MS) throw new Error(`no stored output for ref "${ref}"`);
  const text = readFileSync(path, "utf8");
  if (refOf(text) !== ref) throw new Error(`ref "${ref}" does not match stored content; evidence may be tampered`);
  const total = codePointLength(text);
  const start = Math.min(offset, total);
  const end = Math.min(start + limit, total);
  const startIndex = codePointOffset(text, start);
  const endIndex = codePointOffset(text, end - start, startIndex);
  const page: ReaderPage = {
    range: { offset: start, limit: end - start },
    total,
    eof: end >= total,
    text: text.slice(startIndex, endIndex),
  };
  return { ok: true, text: JSON.stringify(page) };
}
