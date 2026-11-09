import type { DriverKind } from "./contracts.ts";
import { replaysTranscriptNatively } from "./runtime/capabilities.ts";
import { budgetedToolOutputBytes, type BudgetedHookPreviewInput } from "./context-budget.ts";
import { truncateBytes, spillToolOutput } from "./tool-results.ts";

export function replaysNatively(kind: DriverKind): boolean {
  return replaysTranscriptNatively(kind);
}

// rewound and fresh coincide today but stay distinct: rewound also invalidates other instances' cursors.
export interface TurnContextInput {
  text: string;
  transcript: Array<{ role: "user" | "assistant"; text: string }>;
  rewound: boolean;
  fresh: boolean;
  externallyUpdated: boolean;
  replaysNatively: boolean;
  budget?: { contextWindowTokens?: number; reservedOutputTokens?: number; outputReserveTokens?: number; usedTokens?: number };
  spillDir?: string;
}

export function engineIsFresh(input: {
  instanceId: string;
  lastInstanceId: string | undefined;
  resumeCursors: Record<string, unknown>;
  transcript: Array<{ role: "user" | "assistant"; text: string }>;
}): boolean {
  const { instanceId, lastInstanceId, resumeCursors, transcript } = input;
  if (!transcript.some((m) => m.role === "user")) return false;
  if (lastInstanceId !== undefined) return lastInstanceId !== instanceId || resumeCursors[instanceId] === undefined;
  const cursorIds = Object.keys(resumeCursors);
  return !(cursorIds.length === 1 && cursorIds[0] === instanceId);
}

const REWOUND_PREAMBLE =
  "[The user rewound this conversation (edited a message or switched to another version). Everything before this point was replaced by the following history:]";
const FRESH_PREAMBLE =
  "[You are joining this conversation mid-thread (the user switched this bot over to you). The conversation so far:]";
const EXTERNAL_UPDATE_PREAMBLE =
  "[This conversation received an update outside your provider session. The complete current history follows so you can use that update in your next response:]";

const RECOVERED_PREAMBLE =
  "[Your previous session for this conversation could not be resumed, so this is a new session. The conversation so far:]";
/** Collapse consecutive duplicate non-empty lines; keep a count so repeat evidence is not lost. Display helper only: the bounding paths below never dedupe before measuring or spilling (M7), so a stored "full output" and a replayed transcript stay verbatim; callers use this on a preview string. */
export function dedupeRepeatedLines(text: string): string {
  if (!text.includes("\n")) return text;
  const lines = text.split("\n");
  const out: string[] = [];
  let prev = "";
  let run = 0;
  const flush = () => {
    // m5: run counts total occurrences, so say total - not "repeated N times".
    if (prev !== "" && run > 0) out.push(`[line above appeared ${run + 1} times total]`);
    run = 0;
  };
  for (const line of lines) {
    if (line !== "" && line === prev) { run += 1; continue; }
    flush();
    out.push(line);
    prev = line;
  }
  flush();
  return out.join("\n");
}
// ponytail: one byte-boundary helper lives in tool-results; this module only schedules it.
const prefixFor = (role: "user" | "assistant"): string => `${role === "user" ? "User" : "Assistant"}: `;

function verbatimTranscript(transcript: Array<{ role: "user" | "assistant"; text: string }>): string[] {
  return transcript.map((m) => prefixFor(m.role) + m.text);
}

/** m9: single reserved-bytes helper - actual join overhead for the assembled text. */
function reservedBytes(preamble: string, footer: string, text: string, messageCount: number): number {
  // Elements: preamble, "", N messages, "", footer, "", text = N+6 elements,
  // N+5 "\n" separators, plus one extra separator if an omitted-marker line
  // (counted as a message) is prepended.
  return Buffer.byteLength(preamble, "utf8") + Buffer.byteLength(footer, "utf8") + Buffer.byteLength(text, "utf8") + (messageCount + 6);
}

function boundTranscript(transcript: Array<{ role: "user" | "assistant"; text: string }>, baseLimit: number, reserved: number, spillDir?: string): string[] {
  // M9: reserve the omitted-marker line up front so the final text respects
  // baseLimit; worst-case digits come from the full transcript length.
  const markerReserve = Buffer.byteLength(`[(${Math.max(transcript.length, 1)} earlier message(s) omitted: over budget)]`, "utf8") + 1;
  let spent = reserved + markerReserve;
  const rev: string[] = [];
  let omitted = 0;
  // No MAX_REPLAY slice: the budget is the only drop rule, and every message
  // skipped for budget is counted in the omitted marker (counted consistently).
  for (let i = transcript.length - 1; i >= 0; i -= 1) {
    const m = transcript[i]!;
    const prefix = prefixFor(m.role);
    const prefixBytes = Buffer.byteLength(prefix, "utf8");
    if (m.text === "") {
      // Empty messages must not push the total over budget either.
      if (spent + prefixBytes > baseLimit) { omitted += 1; continue; }
      spent += prefixBytes;
      rev.push(prefix);
      continue;
    }
    const avail = baseLimit - spent - prefixBytes;
    if (avail <= 0) { omitted += 1; continue; }
    const bounded = boundToolMessageContent(m.text, avail, spillDir);
    if (bounded === "") { omitted += 1; continue; }
    spent += prefixBytes + Buffer.byteLength(bounded, "utf8");
    rev.push(prefix + bounded);
  }
  rev.reverse();
  if (omitted > 0) rev.unshift(`[(${omitted} earlier message(s) omitted: over budget)]`);
  return rev;
}

/**
 * B1: no budget supplied means unbounded - the pre-budget verbatim replay.
 * Production callers in index.ts pass neither budget nor spillDir, and must
 * not silently lose history. Bounding (with spill/truncation) only happens
 * when a caller explicitly supplies a budget. M9: the latest `text` itself is
 * explicitly exempt from the replay cap - it is the turn being answered.
 */
function replayLines(
  transcript: Array<{ role: "user" | "assistant"; text: string }>,
  budget: TurnContextInput["budget"],
  spillDir: string | undefined,
  preamble: string,
  footer: string,
  text: string,
): string[] {
  if (!budget) return verbatimTranscript(transcript);
  const baseLimit = budgetedToolOutputBytes(budget, budget.usedTokens);
  const reserved = reservedBytes(preamble, footer, text, transcript.length);
  return boundTranscript(transcript, baseLimit, reserved, spillDir);
}
export function buildRecoveryText(input: {
  text: string;
  transcript: Array<{ role: "user" | "assistant"; text: string }>;
  budget?: { contextWindowTokens?: number; reservedOutputTokens?: number; outputReserveTokens?: number; usedTokens?: number };
  spillDir?: string;
}): string | undefined {
  if (input.transcript.length === 0) return undefined;
  const footer = "[Now reply to the user's latest message:]";
  const lines = replayLines(input.transcript, input.budget, input.spillDir, RECOVERED_PREAMBLE, footer, input.text);
  return [RECOVERED_PREAMBLE, "", ...lines, "", footer, "", input.text].join("\n");
}

export function buildTurnContext(input: TurnContextInput): {
  turnText: string;
  resume: boolean;
} {
  const { text, transcript, rewound, fresh, externallyUpdated, replaysNatively } = input;
  const resume = !rewound && !fresh && !externallyUpdated;
  const replay = !resume && !replaysNatively && transcript.length > 0;
  if (!replay) return { turnText: text, resume };
  const preamble = rewound ? REWOUND_PREAMBLE : externallyUpdated ? EXTERNAL_UPDATE_PREAMBLE : FRESH_PREAMBLE;
  const footer = "[Now reply to the user's latest message:]";
  const lines = replayLines(transcript, input.budget, input.spillDir, preamble, footer, text);
  return { turnText: [preamble, "", ...lines, "", footer, "", text].join("\n"), resume };
}

export function budgetedTurnToolLimit(input: BudgetedHookPreviewInput | undefined, usedTokens?: number): number {
  return budgetedToolOutputBytes(input, usedTokens ?? input?.usedTokens);
}
export const budgetedTurnToolLimitBytes = budgetedTurnToolLimit;

/** One shared UTF-8 byte-boundary helper for all spill/truncation paths (tool-results.ts). */
export const truncateUtf8Bytes = truncateBytes;


/** M2: real tool outcome, threaded through to the spill marker. Absent = unknown. */
export type ToolOutcomeMeta = { ok?: boolean | null; error?: string | null; exitCode?: number | null };

export function boundToolMessageContent(content: string, budgetBytes: number, spillDir?: string, meta?: ToolOutcomeMeta): string {
  const finite = Number.isFinite(budgetBytes) && budgetBytes >= 0 ? Math.floor(budgetBytes) : 0;
  // M4: exhausted budget returns "" - no marker, and no spill file write.
  if (finite <= 0) return "";
  // M7: measure and spill the verbatim content; dedupe is never applied here.
  const contentBytes = Buffer.byteLength(content, "utf8");
  if (contentBytes <= finite) return content;
  if (!spillDir) {
    const marker = `\n[truncated ${contentBytes} bytes; budget ${finite} bytes; no spill dir]`;
    const markerBytes = Buffer.byteLength(marker);
    // Never exceed the budget: a marker larger than the budget is truncated to it.
    if (finite < markerBytes) return truncateBytes(marker, finite);
    return truncateBytes(content, finite - markerBytes) + marker;
  }
  // M2: unknown outcome is null (rendered "unknown"), never fabricated true.
  return spillToolOutput(spillDir, content, meta?.ok ?? null, meta?.error ?? null, finite, meta?.exitCode ?? null).preview;
}

function safeJson(value: unknown): string | undefined {
  try { return JSON.stringify(value); } catch { return undefined; }
}
function outcomeMeta(obj: Record<string, unknown>): ToolOutcomeMeta {
  return {
    ok: typeof obj.ok === "boolean" ? obj.ok : null,
    error: typeof obj.error === "string" ? obj.error : null,
    exitCode: typeof obj.exitCode === "number" ? obj.exitCode : null,
  };
}
function minimalFor(content: unknown): unknown {
  if (Array.isArray(content)) return [];
  if (content !== null && typeof content === "object") return {};
  if (typeof content === "string") return "";
  if (typeof content === "number") return 0;
  if (typeof content === "boolean") return false;
  return null;
}

/**
 * M3: bound structured content in place, preserving its top-level type.
 * String fields are bounded as strings; other fields are kept verbatim when
 * their JSON fits, otherwise replaced by an explicitly bounded JSON string
 * (never String(value), which turned objects into "[object Object]").
 * Unserializable (circular) fields become null instead of crashing.
 */
function boundFieldValue(value: unknown, left: number, spillDir: string | undefined, meta: ToolOutcomeMeta | undefined): { value: unknown; used: number } {
  if (typeof value === "string") {
    if (left <= 0) return { value: "", used: 0 };
    const bounded = boundToolMessageContent(value, left, spillDir, meta);
    return { value: bounded, used: Buffer.byteLength(bounded, "utf8") };
  }
  const json = safeJson(value);
  if (json === undefined) return { value: null, used: 4 };
  const jsonBytes = Buffer.byteLength(json, "utf8");
  if (jsonBytes <= left) return { value, used: jsonBytes };
  const bounded = boundToolMessageContent(json, Math.max(0, left), spillDir, meta);
  return { value: bounded, used: Buffer.byteLength(bounded, "utf8") };
}
function boundObjectFields(obj: Record<string, unknown>, remaining: number, spillDir: string | undefined, meta: ToolOutcomeMeta | undefined): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  let left = remaining;
  for (const [key, value] of Object.entries(obj)) {
    const bounded = boundFieldValue(value, left, spillDir, meta);
    out[key] = bounded.value;
    left = Math.max(0, left - bounded.used);
  }
  return out;
}
function boundArrayItems(arr: unknown[], remaining: number, spillDir: string | undefined, meta: ToolOutcomeMeta | undefined): unknown[] {
  const out: unknown[] = [];
  let left = remaining;
  for (const value of arr) {
    const bounded = boundFieldValue(value, left, spillDir, meta);
    out.push(bounded.value);
    left = Math.max(0, left - bounded.used);
  }
  return out;
}

export function applyBudgetedToolOutputLimit<T extends { role: string; content: unknown }>(messages: T[], budgetBytes: number, spillDir?: string): T[] {
  const finite = Number.isFinite(budgetBytes) && budgetBytes >= 0 ? Math.floor(budgetBytes) : 0;
  // m7: the old _caps parameter was accepted and ignored; it is removed.
  let spent = 0;
  return messages.map((msg) => {
    const overhead = Buffer.byteLength(`\n\n${msg.role}:\n`, "utf8");
    const remaining = Math.max(0, finite - spent - overhead);
    // M4: exhausted budget - clear content in place, no marker and no spill.
    if (remaining <= 0 && typeof msg.content === "string") {
      spent += overhead;
      return { ...msg, content: "" };
    }
    if (typeof msg.content === "string") {
      const bounded = boundToolMessageContent(msg.content, remaining, spillDir);
      spent += overhead + Buffer.byteLength(bounded, "utf8");
      return { ...msg, content: bounded };
    }
    if (msg.content && typeof msg.content === "object") {
      const meta = Array.isArray(msg.content) ? undefined : outcomeMeta(msg.content as Record<string, unknown>);
      if (remaining <= 0) {
        spent += overhead;
        return { ...msg, content: minimalFor(msg.content) };
      }
      if (!Array.isArray(msg.content)) {
        const obj = msg.content as Record<string, unknown>;
        // M3: only a string result is bounded as a string; a non-string
        // result is never String()-coerced - it goes through per-field
        // bounding, which keeps it verbatim or replaces it explicitly.
        if ("result" in obj && typeof obj.result === "string") {
          const wrapper = safeJson({ ...obj, result: "" });
          if (wrapper !== undefined) {
            const wrapperBytes = Buffer.byteLength(wrapper, "utf8");
            const resultBudget = Math.max(0, remaining - wrapperBytes);
            const boundedResult = boundToolMessageContent(obj.result, resultBudget, spillDir, meta);
            const boundedObj = { ...obj, result: boundedResult };
            const boundedJson = safeJson(boundedObj);
            spent += overhead + (boundedJson !== undefined ? Buffer.byteLength(boundedJson, "utf8") : wrapperBytes + Buffer.byteLength(boundedResult, "utf8"));
            return { ...msg, content: boundedObj };
          }
          // Circular wrapper: fall through to per-field bounding below.
        }
        const json = safeJson(obj);
        if (json !== undefined && Buffer.byteLength(json, "utf8") <= remaining) {
          spent += overhead + Buffer.byteLength(json, "utf8");
          return msg;
        }
        const boundedObj = boundObjectFields(obj, remaining, spillDir, meta);
        const boundedJson = safeJson(boundedObj);
        spent += overhead + (boundedJson !== undefined ? Buffer.byteLength(boundedJson, "utf8") : 0);
        return { ...msg, content: boundedObj };
      }
      const arr = msg.content as unknown[];
      const json = safeJson(arr);
      if (json !== undefined && Buffer.byteLength(json, "utf8") <= remaining) {
        spent += overhead + Buffer.byteLength(json, "utf8");
        return msg;
      }
      const boundedArr = boundArrayItems(arr, remaining, spillDir, meta);
      const boundedJson = safeJson(boundedArr);
      spent += overhead + (boundedJson !== undefined ? Buffer.byteLength(boundedJson, "utf8") : 0);
      return { ...msg, content: boundedArr };
    }
    // Primitives / null: keep the original when it fits; otherwise the
    // minimal value of the same type - never a string in its place.
    const json = safeJson(msg.content);
    if (json === undefined) {
      spent += overhead;
      return { ...msg, content: null };
    }
    if (remaining > 0 && Buffer.byteLength(json, "utf8") <= remaining) {
      spent += overhead + Buffer.byteLength(json, "utf8");
      return msg;
    }
    spent += overhead + 1;
    return { ...msg, content: minimalFor(msg.content) };
  });
}
