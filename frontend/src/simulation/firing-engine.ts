import type Graph from 'graphology'
import type { RendererContext } from '../graph/renderer'
import { fetchLiveSimulation } from '../api'
import { state } from '../state'
import type { LiveSimulationResponse } from '../types'

const BASE_INTERVAL_MS = 50
const LIVE_REFRESH_MS = 12_000
const BASE_FIRE_PROBABILITY = 0.04
const MAX_STEP_FIRE_PROBABILITY = 0.95

export interface FiringEngine {
  start(): void
  stop(): void
}

export interface FiringEngineOptions {
  onLiveUpdate?: (payload: LiveSimulationResponse) => void
  onLiveError?: (message: string) => void
}

export function createFiringEngine(
  ctx: RendererContext,
  options: FiringEngineOptions = {},
): FiringEngine {
  const { graph, glowIntensities } = ctx
  let timer: ReturnType<typeof setInterval> | null = null
  let refreshTimer: ReturnType<typeof setInterval> | null = null
  let timescaleWatcher: ReturnType<typeof setInterval> | null = null
  const nodeIds = graph.nodes()
  let isFetchingLive = false
  let hasLiveData = false

  const fireProbabilities = new Map<string, number>()

  function setFallbackProbabilities(): void {
    for (const nodeId of nodeIds) {
      const centrality = (graph.getNodeAttribute(nodeId, 'baseSize') - 3) / 18
      fireProbabilities.set(nodeId, BASE_FIRE_PROBABILITY * (0.2 + 0.8 * centrality))
    }
  }

  function setProbabilitiesFromLiveRates(ratesHzByNode: Record<string, number>): void {
    for (const nodeId of nodeIds) {
      const hz = ratesHzByNode[nodeId] ?? 0
      const probability = Math.min(
        MAX_STEP_FIRE_PROBABILITY,
        Math.max(0, (hz * BASE_INTERVAL_MS) / 1000),
      )
      fireProbabilities.set(nodeId, probability)
    }
  }

  async function refreshLiveModel(): Promise<void> {
    if (isFetchingLive) return
    isFetchingLive = true

    try {
      const payload = await fetchLiveSimulation({
        durationMs: 2000,
        burnInMs: 500,
        seed: Math.floor(Date.now() % 1_000_000),
      })
      setProbabilitiesFromLiveRates(payload.firing_rates_hz_by_node)
      hasLiveData = true
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
    for (const nodeId of nodeIds) {
      const baseProb = fireProbabilities.get(nodeId) ?? 0
      const prob = Math.min(MAX_STEP_FIRE_PROBABILITY, baseProb * state.timescale)
      if (Math.random() < prob) {
        // Fire this neuron
        glowIntensities.set(nodeId, 1.0)

        // Cascade: boost direct neighbors
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

    // Re-create interval when timescale changes so tick rate adapts
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
  graph.forEachOutNeighbor(sourceId, (neighbor) => {
    const current = glowIntensities.get(neighbor) || 0
    glowIntensities.set(neighbor, Math.min(1.0, current + 0.25))
  })
}
