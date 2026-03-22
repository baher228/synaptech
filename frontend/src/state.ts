import type { AppState } from './types'

type Listener = () => void

const listeners = new Map<string, Set<Listener>>()

export const state: AppState = {
  timescale: 1,
  selectedNeuron: null,
  hoveredNeuron: null,
  activeFaultyNeuron: null,
  activeReplacementNeuron: null,
  replacementStatus: null,
  sweepStatus: 'idle',
  sweepProgress: 0,
  sweepError: null,
}

export function on(event: string, fn: Listener): () => void {
  if (!listeners.has(event)) listeners.set(event, new Set())
  listeners.get(event)!.add(fn)
  return () => listeners.get(event)?.delete(fn)
}

export function emit(event: string): void {
  listeners.get(event)?.forEach((fn) => fn())
}
