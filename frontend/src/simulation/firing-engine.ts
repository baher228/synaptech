import type Graph from 'graphology'
import type { RendererContext } from '../graph/renderer'
import { state } from '../state'

const BASE_INTERVAL_MS = 50
const BASE_FIRE_PROBABILITY = 0.04

export interface FiringEngine {
  start(): void
  stop(): void
}

export function createFiringEngine(ctx: RendererContext): FiringEngine {
  const { graph, glowIntensities } = ctx
  let timer: ReturnType<typeof setInterval> | null = null
  const nodeIds = graph.nodes()

  // Pre-compute firing probabilities based on degree centrality
  const fireProbabilities = new Map<string, number>()
  for (const nodeId of nodeIds) {
    const centrality = (graph.getNodeAttribute(nodeId, 'baseSize') - 3) / 18
    fireProbabilities.set(nodeId, BASE_FIRE_PROBABILITY * (0.2 + 0.8 * centrality))
  }

  function tick(): void {
    for (const nodeId of nodeIds) {
      const prob = fireProbabilities.get(nodeId)! * state.timescale
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
    timer = setInterval(tick, getInterval())

    // Re-create interval when timescale changes so tick rate adapts
    const checkTimescale = setInterval(() => {
      if (timer === null) {
        clearInterval(checkTimescale)
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
