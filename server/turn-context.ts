import type { DriverKind } from "./contracts.ts";
import { replaysTranscriptNatively } from "./runtime/capabilities.ts";
import { budgetedToolOutputChars, type BudgetedHookPreviewInput } from "./context-budget.ts";
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
/** Collapse consecutive duplicate non-empty lines; keep a count so repeat evidence is not lost. */
export function dedupeRepeatedLines(text: string): string {
  if (!text.includes("\n")) return text;
  const lines = text.split("\n");
  const out: string[] = [];
  let prev = "";
  let run = 0;
  const flush = () => {
    if (prev !== "" && run > 0) out.push(`[previous line repeated ${run + 1} times]`);
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
const MAX_REPLAY = 40;
function boundTranscript(transcript: Array<{ role: "user" | "assistant"; text: string }>, baseLimit: number, reservedBytes: number, spillDir?: string): string[] {
  const window = transcript.slice(-MAX_REPLAY);
  const omittedWindow = transcript.length - window.length;
  const deduped = window.map((m) => ({ role: m.role, text: dedupeRepeatedLines(m.text) }));
  let spent = reservedBytes;
  const rev: string[] = [];
  let omitted = omittedWindow;
  for (let i = deduped.length - 1; i >= 0; i -= 1) {
    const m = deduped[i]!;
    const prefix = `${m.role === "user" ? "User" : "Assistant"}: `;
    const prefixBytes = Buffer.byteLength(prefix, "utf8");
    const avail = Math.max(0, baseLimit - spent - prefixBytes);
    if (m.text !== "" && avail <= 0) { omitted += 1; continue; }
    const bounded = boundToolMessageContent(m.text, avail, spillDir);
    spent += prefixBytes + Buffer.byteLength(bounded, "utf8");
    rev.push(prefix + bounded);
  }
  rev.reverse();
  if (omitted > 0) rev.unshift(`[(${omitted} earlier message(s) omitted: over budget)]`);
  return rev;
}
export function buildRecoveryText(input: {
  text: string;
  transcript: Array<{ role: "user" | "assistant"; text: string }>;
  budget?: { contextWindowTokens?: number; reservedOutputTokens?: number; outputReserveTokens?: number; usedTokens?: number };
  spillDir?: string;
}): string | undefined {
  if (input.transcript.length === 0) return undefined;
  const baseLimit = budgetedToolOutputChars(input.budget, input.budget?.usedTokens || undefined);
  const footer = "[Now reply to the user's latest message:]";
  const reserved = Buffer.byteLength(RECOVERED_PREAMBLE, "utf8") + Buffer.byteLength(footer, "utf8") + Buffer.byteLength(input.text, "utf8") + (5 + input.transcript.length);
  return [RECOVERED_PREAMBLE, "", ...boundTranscript(input.transcript, baseLimit, reserved, input.spillDir), "", footer, "", input.text].join("\n");
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
  const baseLimit = budgetedToolOutputChars(input.budget, input.budget?.usedTokens || undefined);
  const footer = "[Now reply to the user's latest message:]";
  const reserved = Buffer.byteLength(preamble, "utf8") + Buffer.byteLength(footer, "utf8") + Buffer.byteLength(text, "utf8") + (5 + transcript.length);
  return { turnText: [preamble, "", ...boundTranscript(transcript, baseLimit, reserved, input.spillDir), "", footer, "", text].join("\n"), resume };
}

export function budgetedTurnToolLimit(input: BudgetedHookPreviewInput | undefined, usedTokens?: number): number {
  return budgetedToolOutputChars(input, usedTokens ?? input?.usedTokens);
}

/** One shared UTF-8 byte-boundary helper for all spill/truncation paths (tool-results.ts). */
export const truncateUtf8Bytes = truncateBytes;


export function boundToolMessageContent(content: string, budgetBytes: number, spillDir?: string): string {
  const finite = Number.isFinite(budgetBytes) && budgetBytes >= 0 ? Math.floor(budgetBytes) : 0;
  content = dedupeRepeatedLines(content);
  const contentBytes = Buffer.byteLength(content, "utf8");
  if (contentBytes <= finite) return content;
  if (!spillDir) {
    const marker = `\n[truncated ${contentBytes} bytes; budget ${finite} bytes; no spill dir]`;
    const markerBytes = Buffer.byteLength(marker);
    // ponytail: marker is never truncated; tiny budgets return it whole even over budget. Shrink marker only with schema change.
    if (finite < markerBytes) return marker;
    return truncateBytes(content, finite - markerBytes) + marker;
  }
  return spillToolOutput(spillDir, content, true, null, finite).preview;
}

export function applyBudgetedToolOutputLimit<T extends { role: string; content: unknown }>(messages: T[], budgetBytes: number, spillDir?: string, _caps?: { max?: number; min?: number; minBytes?: number }): T[] {
  const finite = Number.isFinite(budgetBytes) && budgetBytes >= 0 ? Math.floor(budgetBytes) : 0;
  // ponytail: unused scale/min are theatre for MAX — removed to keep 8000/16000 honest; MAX stays fixed.
  let spent = 0;
  return messages.map((msg) => {
    const overhead = Buffer.byteLength(`\n\n${msg.role}:\n`, "utf8");
    if (typeof msg.content === "string") {
      const remaining = Math.max(0, finite - spent - overhead);
      const bounded = boundToolMessageContent(msg.content, remaining, spillDir);
      spent += overhead + Buffer.byteLength(bounded, "utf8");
      return { ...msg, content: bounded };
    }
    if (msg.content && typeof msg.content === "object" && !Array.isArray(msg.content)) {
      const obj = msg.content as Record<string, unknown>;
      if ("result" in obj) {
        const wrapper = JSON.stringify({ ...obj, result: "" });
        const wrapperBytes = Buffer.byteLength(wrapper, "utf8");
        const remaining = Math.max(0, finite - spent - overhead - wrapperBytes);
        const boundedResult = boundToolMessageContent(String(obj.result ?? ""), remaining, spillDir);
        spent += overhead + wrapperBytes + Buffer.byteLength(boundedResult, "utf8");
        return { ...msg, content: { ...obj, result: boundedResult } };
      }
      const remaining = Math.max(0, finite - spent - overhead);
      const json = JSON.stringify(obj);
      if (Buffer.byteLength(json, "utf8") <= remaining) {
        spent += overhead + Buffer.byteLength(json, "utf8");
        return msg;
      }
      const bounded = boundToolMessageContent(json, remaining, spillDir);
      spent += overhead + Buffer.byteLength(bounded, "utf8");
      return { ...msg, content: bounded };
    }
    const remaining = Math.max(0, finite - spent - overhead);
    let json: string;
    try { json = JSON.stringify(msg.content) ?? String(msg.content); } catch { json = String(msg.content); }
    if (Buffer.byteLength(json, "utf8") <= remaining) {
      spent += overhead + Buffer.byteLength(json, "utf8");
      return msg;
    }
    const bounded = boundToolMessageContent(json, remaining, spillDir);
    spent += overhead + Buffer.byteLength(bounded, "utf8");
    return { ...msg, content: bounded };
  });
}
