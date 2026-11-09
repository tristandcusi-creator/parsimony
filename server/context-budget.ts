import type { ModelCatalog } from "./contracts.ts";
import { modelContextWindow } from "./model-context-window.ts";

export const DEFAULT_CONTEXT_WINDOW = 128_000;
const positive = (n: number | null | undefined): n is number => typeof n === "number" && Number.isFinite(n) && n > 0;

export function contextWindowFor(model: string | undefined, catalog?: ModelCatalog, reported?: number): number {
  if (positive(reported)) return reported;
  const declared = catalog?.options.find((option) => option.id === model)?.contextWindow;
  if (positive(declared)) return declared;
  const inferred = model ? modelContextWindow(model) : undefined;
  return positive(inferred) ? inferred : DEFAULT_CONTEXT_WINDOW;
}

/** A "meter" reading (an ACP agent's own usage_update) is shown only; it never times OpenMausBot's compaction. */
export function compactionReading<T>(capabilities: { contextReading?: "meter" } | undefined, reading: T | undefined): T | undefined {
  return capabilities?.contextReading === "meter" ? undefined : reading;
}

/** Leave headroom even for a small local model: an 8k floor would make compaction unreachable on an 8k window. */
export function compactBudget(window: number, compactAt?: number, nativeCompactAt?: number): number {
  if (!positive(window)) throw new Error("invalid model context window");
  const configured = positive(compactAt) ? (compactAt < 1 ? window * compactAt : compactAt) : window * 0.8;
  return Math.max(1, Math.floor(Math.min(configured, window * 0.9, positive(nativeCompactAt) ? nativeCompactAt * 0.9 : Infinity)));
}

export function shouldCompact(input: { contextTokens?: number; estimatedBytes: number; budget: number; floor?: number; window: number; nativeCompactAt?: number }): boolean {
  // Summed input across tool rounds is NOT a context-window measurement.
  const size = positive(input.contextTokens) ? input.contextTokens : Math.ceil(input.estimatedBytes / 4);
  // Regrowth avoids refolding an irreducible prompt but never passes a known provider-native compaction boundary.
  const ceiling = compactBudget(input.window, input.window, input.nativeCompactAt);
  const threshold = Math.min(ceiling, positive(input.floor) ? Math.max(input.budget, input.floor * 1.25) : input.budget);
  return size >= threshold;
}
export const TOOL_OUTPUT_DEFAULT_CHARS = 8_000;
export const TOOL_OUTPUT_MIN_CHARS = 1_000;
export const TOOL_OUTPUT_MAX_CHARS = 8_000;
export const TOOL_OUTPUT_HARD_MAX_BYTES = TOOL_OUTPUT_MAX_CHARS;
export const HOOK_PREVIEW_HARD_MAX_BYTES = 16_000;
const TOOL_OUTPUT_BUDGET_SHARE = 0.05;

export type BudgetedHookPreviewInput = {
  contextWindowTokens?: number | null;
  reservedOutputTokens?: number | null;
  outputReserveTokens?: number | null;
  usedTokens?: number;
};
export type ToolOutputBudget = BudgetedHookPreviewInput;

/**
 * Bytes end-to-end: 4 bytes/token consistently; return is bytes (== chars for ASCII).
 * Rename of budgetedToolOutputChars: the return has always been bytes, the old
 * name is kept as an alias so existing imports keep working.
 * B2: TOOL_OUTPUT_MAX_CHARS is a hard ceiling - `min(MAX, share)` - it is never
 * raised with the window. M12: no floor when headroom is exhausted (return 0).
 * m1: non-positive windows use the default, consistently with 0. m2: usedTokens
 * carried on the budget object is honoured when no explicit usedTokens is given.
 */
export function budgetedToolOutputBytes(
  budget: ToolOutputBudget | null | undefined,
  usedTokens?: number,
): number {
  const window = budget?.contextWindowTokens;
  if (!positive(window)) return TOOL_OUTPUT_DEFAULT_CHARS;
  const rawReserve = budget?.reservedOutputTokens ?? budget?.outputReserveTokens ?? 0;
  const reserve = Number.isFinite(rawReserve) && rawReserve > 0 ? rawReserve : 0;
  const rawUsed = usedTokens ?? budget?.usedTokens ?? 0;
  const used = typeof rawUsed === "number" && Number.isFinite(rawUsed) && rawUsed > 0 ? rawUsed : 0;
  const windowBytes = window * 4;
  const headroomBytes = Math.max(0, windowBytes - reserve * 4 - used * 4);
  if (headroomBytes <= 0) return 0;
  const bytes = Math.floor(headroomBytes * TOOL_OUTPUT_BUDGET_SHARE);
  if (!Number.isFinite(bytes)) return TOOL_OUTPUT_DEFAULT_CHARS;
  return Math.min(TOOL_OUTPUT_MAX_CHARS, Math.max(TOOL_OUTPUT_MIN_CHARS, bytes));
}
export const budgetedToolOutputChars = budgetedToolOutputBytes;

export function budgetedHookPreviewBytes(
  budget: ToolOutputBudget | null | undefined,
  usedTokens?: number,
): number {
  const base = budgetedToolOutputBytes(budget, usedTokens);
  if (base <= 0) return 0;
  return Math.min(HOOK_PREVIEW_HARD_MAX_BYTES, Math.max(2_000, base * 2));
}
export const budgetedHookPreviewChars = budgetedHookPreviewBytes;
// m3: TOOL_OUTPUT_DEFAULT_CHARS (8k) is what is returned when no positive window
// is supplied; TOOL_OUTPUT_MAX_CHARS (8k) is the hard ceiling for a budgeted
// tool-output preview regardless of window size. There is no 128k-char max here
// (128k is the default context window in tokens, a different quantity).
