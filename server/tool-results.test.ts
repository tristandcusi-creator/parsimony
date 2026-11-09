import { mkdtempSync, rmSync, statSync, writeFileSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { clearSpillHashCache, pruneSpills, readSpilledToolOutput, spillHashCacheSize, spillToolOutput, ToolResults, TOOL_RESULT_MAX_CHARS, TOOL_RESULT_PREVIEW_CHARS, TOOL_RESULT_TTL_MS } from "./tool-results.ts";

const owner = { botId: "a", threadId: "chat" };

describe("temporary agent tool results", () => {
  it("pages losslessly across emoji boundaries and redacts before retaining", () => {
    const results = new ToolResults();
    const secret = `sk-test-${"s".repeat(30)}`;
    const text = `${"a".repeat(15_999)}🌱${"b".repeat(18_000)} ${secret}`;
    const saved = results.save(owner, text);
    let reconstructed = "";
    let offset = 0;
    while (offset < saved.length) {
      const page = results.read(owner, saved.id, offset)!;
      expect(Buffer.from(page.text).toString()).toBe(page.text);
      expect(page.text.length).toBeLessThanOrEqual(TOOL_RESULT_PREVIEW_CHARS);
      reconstructed += page.text;
      expect(page.nextOffset).toBeGreaterThan(offset);
      offset = page.nextOffset;
    }
    expect(reconstructed).toBe(text.replace(secret, `«redacted ${secret.length} chars»`));
    expect(results.read(owner, saved.id, 16_000)?.offset).toBe(15_999);
    expect(results.read(owner, saved.id, saved.length)?.text).toBe("");
  });

  it("scopes reads to both the bot and conversation, including room speakers", () => {
    const results = new ToolResults();
    const saved = results.save(owner, "private result");
    expect(results.read({ ...owner, botId: "peer" }, saved.id, 0)).toBeNull();
    expect(results.read({ ...owner, threadId: "sibling" }, saved.id, 0)).toBeNull();
    expect(results.read(owner, saved.id, 0)?.text).toBe("private result");
    expect(results.read(owner, "../chat", 0)).toBeNull();
    for (const offset of [-1, 0.5, Infinity, NaN, 10_000]) expect(results.read(owner, saved.id, offset)).toBeNull();
  });

  it("expires without extending retention on read, and restart starts empty", () => {
    let now = 100;
    const results = new ToolResults(() => now);
    const saved = results.save(owner, "temporary");
    now += TOOL_RESULT_TTL_MS - 1;
    expect(results.read(owner, saved.id, 0)).not.toBeNull();
    now++;
    expect(results.read(owner, saved.id, 0)).toBeNull();
    expect(new ToolResults().read(owner, saved.id, 0)).toBeNull();
  });

  it("marks storage truncation rather than claiming the whole result survives", () => {
    const results = new ToolResults();
    const saved = results.save(owner, `${"x".repeat(TOOL_RESULT_MAX_CHARS - 1)}🌱tail`);
    expect(saved).toMatchObject({ truncated: true, length: TOOL_RESULT_MAX_CHARS - 1 });
    expect(results.save(owner, "already cut upstream", true).truncated).toBe(true);
    expect(results.read(owner, saved.id, 0)?.truncated).toBe(true);
  });

  it("evicts the owner's oldest entries before neighbours and bounds total count", () => {
    const results = new ToolResults();
    const other = { ...owner, botId: "other" };
    const neighbour = results.save(other, "neighbour");
    const first = results.save(owner, "first");
    for (let i = 0; i < 16; i++) results.save(owner, String(i));
    expect(results.read(owner, first.id, 0)).toBeNull();
    expect(results.read(other, neighbour.id, 0)).not.toBeNull();
    for (let i = 0; i < 128; i++) results.save({ botId: String(i), threadId: "new" }, "small");
    expect(results.read(other, neighbour.id, 0)).toBeNull();
  });

  it("bounds owner and global UTF-8 bytes, not just the number of entries", () => {
    const results = new ToolResults();
    const text = "界".repeat(TOOL_RESULT_MAX_CHARS);
    const first = results.save(owner, text);
    for (let i = 0; i < 5; i++) results.save(owner, text);
    expect(results.read(owner, first.id, 0)).toBeNull();
    const oldest = results.save({ botId: "global", threadId: "old" }, text);
    for (let i = 0; i < 43; i++) results.save({ botId: "global", threadId: String(i) }, text);
    expect(results.read({ botId: "global", threadId: "old" }, oldest.id, 0)).toBeNull();
  });
});

describe("spilled tool output", () => {
  const tempDir = () => mkdtempSync(join(tmpdir(), "vox-spill-"));
  const roundTrip = (dir: string, text: string, limit: number) => {
    const spill = spillToolOutput(dir, text, true, null, 80);
    let offset = 0; let acc = ""; let fffd = false; let pages = 0;
    for (;;) {
      const page = readSpilledToolOutput(dir, spill.path, offset, limit);
      if (page.text.includes("\uFFFD")) fffd = true;
      acc += page.text; pages += 1;
      if (page.eof) break;
      expect(page.nextOffset).toBeGreaterThan(offset);
      offset = page.nextOffset;
    }
    return { spill, acc, fffd, pages, lastOffset: offset };
  };

  it("round-trips mixed CJK/emoji byte-exact across pages", () => {
    const dir = tempDir();
    try {
      const text = "a界🌱".repeat(4000);
      const { spill, acc, fffd } = roundTrip(dir, text, 8000);
      expect(acc).toBe(text);
      expect(fffd).toBe(false);
      expect(spill.size).toBe(32000);
      const cjk = roundTrip(dir, "界".repeat(5000), 8000);
      expect(cjk.acc).toBe("界".repeat(5000));
      expect(cjk.fffd).toBe(false);
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  it("backs off a partial emoji after a complete 2-byte char instead of the unrelated leader", () => {
    const dir = tempDir();
    try {
      const text = "aé" + "🌱" + "b".repeat(100);
      const spill = spillToolOutput(dir, text, true, null, 10);
      const first = readSpilledToolOutput(dir, spill.path, 0, 4);
      expect(first.text).toBe("aé");
      expect(first.text).not.toContain("\uFFFD");
      expect(first.nextOffset).toBe(3);
      const second = readSpilledToolOutput(dir, spill.path, first.nextOffset, 100);
      expect(second.text.startsWith("🌱")).toBe(true);
      const full = roundTrip(dir, text, 4);
      expect(full.acc).toBe(text);
      expect(full.fffd).toBe(false);
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  it("records original bytes and truncation while capping at 1.5MB", () => {
    const dir = tempDir();
    try {
      const big = "界".repeat(600000);
      const spill = spillToolOutput(dir, big, true, null, 8000);
      expect(spill.originalLength).toBe(600000);
      expect(spill.originalBytes).toBe(1800000);
      expect(spill.truncated).toBe(true);
      expect(spill.size).toBe(1500000);
      expect(spill.size).toBeLessThanOrEqual(1500000);
      const small = spillToolOutput(dir, "hello 界🌱", true, null, 8000);
      expect(small.truncated).toBe(false);
      expect(small.originalBytes).toBe(13);
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  it("bounds the spill marker to the preview budget", () => {
    const dir = tempDir();
    try {
      const text = "界🌱".repeat(2000);
      const zero = spillToolOutput(dir, text, true, null, 0);
      expect(zero.preview).toBe("");
      expect(zero.path).toContain(".txt");
      expect(zero.hash).toHaveLength(64);
      expect(zero.size).toBeGreaterThan(0);
      expect(readSpilledToolOutput(dir, zero.path, 0, 100).text.length).toBeGreaterThan(0);
      for (const finite of [10, 100]) {
        const spill = spillToolOutput(dir, text, true, null, finite);
        expect(Buffer.byteLength(spill.preview, "utf8")).toBeLessThanOrEqual(finite);
        expect(spill.preview.startsWith("\n[full out")).toBe(true);
        expect(readSpilledToolOutput(dir, spill.path, 0, 100).text.length).toBeGreaterThan(0);
      }
      const full = spillToolOutput(dir, text, true, null, 1000);
      expect(Buffer.byteLength(full.preview, "utf8")).toBeLessThanOrEqual(1000);
      expect(full.preview).toContain(`at ${full.path}`);
      expect(full.preview).toContain(`hash ${full.hash};`);
      expect(full.preview).toContain("read with byte offsets");
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  it("bounds the hash cache and evicts on prune", () => {
    const dir = tempDir();
    try {
      clearSpillHashCache();
      for (let i = 0; i < 200; i++) {
        const spill = spillToolOutput(dir, `value-${i}`, true, null, 10);
        readSpilledToolOutput(dir, spill.path, 0, 100);
      }
      expect(spillHashCacheSize()).toBeLessThanOrEqual(128);
      const one = spillToolOutput(dir, "only", true, null, 10);
      readSpilledToolOutput(dir, one.path, 0, 100);
      const before = spillHashCacheSize();
      expect(before).toBeGreaterThan(0);
      const oldDate = new Date(Date.now() - TOOL_RESULT_TTL_MS - 1000);
      utimesSync(one.path, oldDate, oldDate);
      pruneSpills(dir);
      expect(spillHashCacheSize()).toBe(before - 1);
    } finally { clearSpillHashCache(); rmSync(dir, { recursive: true, force: true }); }
  });

  it("rejects a tampered file even with the same size", () => {
    const dir = tempDir();
    try {
      const spill = spillToolOutput(dir, "ABCDEFGH", true, null, 10);
      readSpilledToolOutput(dir, spill.path, 0, 100);
      const stat = statSync(spill.path);
      writeFileSync(spill.path, "XBCDEFGH");
      utimesSync(spill.path, stat.atime, stat.mtime);
      expect(() => readSpilledToolOutput(dir, spill.path, 0, 100)).toThrow(/does not match/);
      expect(() => readSpilledToolOutput(dir, spill.path, 0, 100, spill.hash)).toThrow(/does not match/);
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });
});
