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
const TOOL_OUTPUT_BUDGET_SHARE = 0.05;

export type BudgetedHookPreviewInput = {
  contextWindowTokens?: number | null;
  reservedOutputTokens?: number | null;
  outputReserveTokens?: number | null;
  usedTokens?: number;
};

export function budgetedToolOutputChars(
  budget: { contextWindowTokens?: number | null; reservedOutputTokens?: number | null; outputReserveTokens?: number | null } | null | undefined,
  usedTokens?: number,
): number {
  const window = budget?.contextWindowTokens;
  if (!window || !Number.isFinite(window)) return TOOL_OUTPUT_DEFAULT_CHARS;
  const rawReserve = budget?.reservedOutputTokens ?? budget?.outputReserveTokens ?? 0;
  const reserve = Number.isFinite(rawReserve) && rawReserve > 0 ? rawReserve : 0;
  const used = typeof usedTokens === "number" && Number.isFinite(usedTokens) && usedTokens > 0 ? usedTokens : 0;
  // Bytes end-to-end: 4 bytes/token consistently; return is bytes (== chars for ASCII).
  const windowBytes = window * 4;
  const reserveBytes = reserve * 4;
  const usedBytes = used * 4;
  const headroomBytes = Math.max(0, windowBytes - reserveBytes - usedBytes);
  const bytes = Math.floor(headroomBytes * TOOL_OUTPUT_BUDGET_SHARE);
  if (!Number.isFinite(bytes)) return TOOL_OUTPUT_DEFAULT_CHARS;
  // ponytail: cap scales with the window; fixed 8k clipped large windows. Raise cap when share changes.
  const maxForWindow = Math.max(TOOL_OUTPUT_MAX_CHARS, Math.floor(windowBytes * TOOL_OUTPUT_BUDGET_SHARE));
  return Math.min(maxForWindow, Math.max(TOOL_OUTPUT_MIN_CHARS, bytes));
}

export function budgetedHookPreviewChars(
  budget: { contextWindowTokens?: number | null; reservedOutputTokens?: number | null; outputReserveTokens?: number | null } | null | undefined,
  usedTokens?: number,
): number {
  const window = budget?.contextWindowTokens;
  const windowBytes = typeof window === "number" && Number.isFinite(window) ? window * 4 : 0;
  const maxForWindow = Math.max(16_000, Math.floor(windowBytes * TOOL_OUTPUT_BUDGET_SHARE) * 2);
  return Math.min(maxForWindow, Math.max(2_000, budgetedToolOutputChars(budget, usedTokens) * 2));
}
// ponytail: MAX is fixed at 128_000 chars (~512kB budget). If smaller windows matter, scale MAX from window; keep literals for now. 8000/16000 previews unchanged.
