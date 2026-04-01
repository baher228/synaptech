import type Graph from 'graphology'
import type { RendererContext } from '../graph/renderer'
import { fetchLiveSimulation } from '../api'
import { state } from '../state'
import type { LiveSimulationResponse, SpikeData } from '../types'

const BASE_INTERVAL_MS = 50
const LIVE_REFRESH_MS = 2_000
const BASE_FIRE_PROBABILITY = 0.04
const MAX_STEP_FIRE_PROBABILITY = 0.95

export interface FiringEngine {
  start(): void
  stop(): void
}

export interface FiringEngineOptions {
  /** When set with spike trains, replay dominates over live polling. */
  spikeData?: SpikeData
  onLiveUpdate?: (payload: LiveSimulationResponse) => void
  onLiveError?: (message: string) => void
}

interface SpikeEvent {
  time: number
  neuron: string
}

/**
 * Prefer replaying spike-trains from the backend; otherwise poll /api/simulation/live
 * for firing rates and drive the visualization (with cosmetic fallback on error).
 */
export function createFiringEngine(
  ctx: RendererContext,
  options: FiringEngineOptions = {},
): FiringEngine {
  if (options.spikeData && Object.keys(options.spikeData.spike_trains).length > 0) {
    // Use replay for graph visualization, but also poll live data
    // so the metrics panel Live tab receives firing rate updates.
    const replay = createReplayEngine(ctx, options.spikeData)
    const poller = createLivePollingEngine(ctx, options)
    return {
      start() { replay.start(); poller.start() },
      stop() { replay.stop(); poller.stop() },
    }
  }
  return createLivePollingEngine(ctx, options)
}

function createReplayEngine(ctx: RendererContext, data: SpikeData): FiringEngine {
  const { graph, glowIntensities } = ctx

  const events: SpikeEvent[] = []
  for (const [neuron, times] of Object.entries(data.spike_trains)) {
    if (!graph.hasNode(neuron)) continue
    for (const t of times) {
      events.push({ time: t, neuron })
    }
  }
  events.sort((a, b) => a.time - b.time)

  const duration = data.duration_ms
  let cursor = 0
  let eventIdx = 0
  let lastWall = 0
  let rafId: number | null = null

  function tick(now: number): void {
    const wallDt = lastWall ? now - lastWall : 16.67
    lastWall = now

    const simDt = wallDt * state.timescale
    const prevCursor = cursor
    cursor += simDt

    if (cursor >= duration) {
      cursor %= duration
      eventIdx = 0
    }

    const lo = prevCursor
    const hi = cursor

    if (lo <= hi) {
      while (eventIdx < events.length && events[eventIdx].time < hi) {
        if (events[eventIdx].time >= lo) {
          fire(events[eventIdx].neuron)
        }
        eventIdx++
      }
    } else {
      while (eventIdx < events.length) {
        fire(events[eventIdx].neuron)
        eventIdx++
      }
      eventIdx = 0
      while (eventIdx < events.length && events[eventIdx].time < hi) {
        fire(events[eventIdx].neuron)
        eventIdx++
      }
    }

    rafId = requestAnimationFrame(tick)
  }

  function fire(neuron: string): void {
    if (!graph.hasNode(neuron)) return
    glowIntensities.set(neuron, 1.0)
    graph.forEachOutNeighbor(neuron, (neighbor) => {
      const current = glowIntensities.get(neighbor) || 0
      glowIntensities.set(neighbor, Math.min(1.0, current + 0.25))
    })
  }

  return {
    start() {
      if (rafId !== null) return
      lastWall = 0
      rafId = requestAnimationFrame(tick)
    },
    stop() {
      if (rafId !== null) {
        cancelAnimationFrame(rafId)
        rafId = null
      }
    },
  }
}

function createLivePollingEngine(
  ctx: RendererContext,
  options: FiringEngineOptions,
): FiringEngine {
  const { graph, glowIntensities } = ctx
  let timer: ReturnType<typeof setInterval> | null = null
  let refreshTimer: ReturnType<typeof setInterval> | null = null
  let timescaleWatcher: ReturnType<typeof setInterval> | null = null
  let isFetchingLive = false
  let hasLiveData = false

  const fireProbabilities = new Map<string, number>()

  function setFallbackProbabilities(): void {
    for (const nodeId of graph.nodes()) {
      const centrality = (graph.getNodeAttribute(nodeId, 'baseSize') - 3) / 18
      fireProbabilities.set(nodeId, BASE_FIRE_PROBABILITY * (0.2 + 0.8 * centrality))
    }
  }

  function setProbabilitiesFromLiveRates(ratesHzByNode: Record<string, number>): boolean {
    let hasActive = false
    const nextProbabilities = new Map<string, number>()
    for (const nodeId of graph.nodes()) {
      const hz = ratesHzByNode[nodeId] ?? 0
      if (hz > 0) hasActive = true
      const probability = Math.min(
        MAX_STEP_FIRE_PROBABILITY,
        Math.max(0, (hz * BASE_INTERVAL_MS) / 1000),
      )
      nextProbabilities.set(nodeId, probability)
    }
    if (!hasActive) return false
    nextProbabilities.forEach((value, nodeId) => {
      fireProbabilities.set(nodeId, value)
    })
    return hasActive
  }

  async function refreshLiveModel(): Promise<void> {
    if (isFetchingLive) return
    // Don't compete with a running sweep for the shared engine
    if (state.sweepStatus === 'connecting' || state.sweepStatus === 'receiving') return
    isFetchingLive = true

    try {
      const payload = await fetchLiveSimulation()
      const hasActivity = setProbabilitiesFromLiveRates(payload.firing_rates_hz_by_node)
      if (!hasActivity) {
        // Guard against stale live sessions briefly returning all-zero rates:
        // keep prior live probabilities (or fallback if we have none yet).
        if (!hasLiveData) {
          setFallbackProbabilities()
        }
      } else {
        hasLiveData = true
      }
      options.onLiveUpdate?.(payload)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown live-data error'
      options.onLiveError?.(message)
      if (!hasLiveData) {
        setFallbackProbabilities()
      }
    } finally {
      isFetchingLive = false
    }
  }

  function tick(): void {
    for (const nodeId of graph.nodes()) {
      const baseProb = fireProbabilities.get(nodeId) ?? 0
      const prob = Math.min(MAX_STEP_FIRE_PROBABILITY, baseProb * state.timescale)
      if (Math.random() < prob) {
        glowIntensities.set(nodeId, 1.0)
        cascadeToNeighbors(graph, glowIntensities, nodeId)
      }
    }
  }

  function getInterval(): number {
    return Math.max(5, BASE_INTERVAL_MS / state.timescale)
  }

  function start(): void {
    if (timer !== null) return
    setFallbackProbabilities()
    void refreshLiveModel()
    refreshTimer = setInterval(() => {
      void refreshLiveModel()
    }, LIVE_REFRESH_MS)

    timer = setInterval(tick, getInterval())

    timescaleWatcher = setInterval(() => {
      if (timer === null) {
        if (timescaleWatcher !== null) {
          clearInterval(timescaleWatcher)
          timescaleWatcher = null
        }
        return
      }
      clearInterval(timer)
      timer = setInterval(tick, getInterval())
    }, 200)
  }

  function stop(): void {
    if (timer !== null) {
      clearInterval(timer)
      timer = null
    }
    if (refreshTimer !== null) {
      clearInterval(refreshTimer)
      refreshTimer = null
    }
    if (timescaleWatcher !== null) {
      clearInterval(timescaleWatcher)
      timescaleWatcher = null
    }
  }

  return { start, stop }
}

function cascadeToNeighbors(
  graph: Graph,
  glowIntensities: Map<string, number>,
  sourceId: string,
): void {
  if (!graph.hasNode(sourceId)) return
  graph.forEachOutNeighbor(sourceId, (neighbor) => {
    const current = glowIntensities.get(neighbor) || 0
    glowIntensities.set(neighbor, Math.min(1.0, current + 0.25))
  })
}
