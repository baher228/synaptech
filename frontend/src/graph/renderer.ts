import Graph from 'graphology'
import Sigma from 'sigma'
import { EdgeLineProgram } from 'sigma/rendering'

import type { GraphData, GraphNode } from '../types'
import { state } from '../state'
import { normalizePositions } from './layout'
import { NEURON_COLORS, NEURON_GLOW_COLORS, EDGE_COLOR, lerpColor } from './colors'

export interface RendererContext {
  graph: Graph
  sigma: Sigma
  glowIntensities: Map<string, number>
  nodeIndex: Map<string, GraphNode>
}

export function createRenderer(
  container: HTMLElement,
  data: GraphData,
): RendererContext {
  const graph = new Graph({ multi: false, type: 'directed', allowSelfLoops: true })
  const positions = normalizePositions(data.nodes)
  const glowIntensities = new Map<string, number>()
  const nodeIndex = new Map<string, GraphNode>()

  for (const node of data.nodes) {
    const pos = positions[node.id]
    const baseSize = 3 + node.degree_centrality * 18
    graph.addNode(node.id, {
      x: pos.x,
      y: pos.y,
      size: baseSize,
      color: NEURON_COLORS[node.type] || '#888',
      label: node.id,
      type: 'circle',
      // store base values for the reducer
      baseSize,
      baseColor: NEURON_COLORS[node.type] || '#888',
      neuronType: node.type,
    })
    glowIntensities.set(node.id, 0)
    nodeIndex.set(node.id, node)
  }

  for (const edge of data.edges) {
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) continue
    const key = `${edge.source}->${edge.target}`
    if (graph.hasEdge(key)) continue
    graph.addEdgeWithKey(key, edge.source, edge.target, {
      size: Math.max(0.3, Math.log(edge.weight + 1) * 0.4),
      color: EDGE_COLOR,
      weight: edge.weight,
    })
  }

  const sigma = new Sigma(graph, container, {
    defaultEdgeType: 'line',
    edgeProgramClasses: {
      line: EdgeLineProgram,
    },
    renderLabels: true,
    labelFont: 'DM Mono, Consolas, monospace',
    labelSize: 11,
    labelWeight: '500',
    labelColor: { color: '#1A1A2E' },
    labelRenderedSizeThreshold: 8,
    labelDensity: 0.5,
    labelGridCellSize: 100,
    stagePadding: 40,
    minEdgeThickness: 0.3,
    zoomingRatio: 1.3,
    minCameraRatio: 0.08,
    maxCameraRatio: 3,
    autoRescale: true,
    autoCenter: true,
    hideEdgesOnMove: false,
    antiAliasingFeather: 0.6,
    defaultDrawNodeHover: () => {},

    nodeReducer: (nodeId, attrs) => {
      const glow = glowIntensities.get(nodeId) || 0
      const baseSize = (attrs as any).baseSize || attrs.size
      const baseColor = (attrs as any).baseColor || attrs.color
      const neuronType = (attrs as any).neuronType || 'I'
      const glowColor = NEURON_GLOW_COLORS[neuronType] || '#FFF'

      const isSelected = state.selectedNeuron === nodeId
      const isHovered = state.hoveredNeuron === nodeId
      const hasSelection = state.selectedNeuron !== null
      const isNeighborOfSelected =
        hasSelection &&
        state.selectedNeuron !== nodeId &&
        (graph.hasEdge(state.selectedNeuron!, nodeId) || graph.hasEdge(nodeId, state.selectedNeuron!))

      // Glow effect: boost size and shift color toward glow color
      let size = baseSize + glow * baseSize * 1.2
      let color = glow > 0.01 ? lerpColor(baseColor, glowColor, glow * 0.7) : baseColor

      // Selection/hover dimming
      let alpha = 1
      if (hasSelection && !isSelected && !isNeighborOfSelected) {
        alpha = 0.15
      }
      if (isSelected || isHovered) {
        size *= 1.15
      }

      return {
        ...attrs,
        size,
        color: alpha < 1 ? applyAlpha(color, alpha) : color,
        zIndex: isSelected ? 2 : glow > 0.01 ? 1 : 0,
      }
    },

    edgeReducer: (edgeId, attrs) => {
      const hasSelection = state.selectedNeuron !== null
      if (!hasSelection) {
        return { ...attrs, color: EDGE_COLOR, hidden: false }
      }

      const [source, target] = graph.extremities(edgeId)
      const isConnected = source === state.selectedNeuron || target === state.selectedNeuron

      return {
        ...attrs,
        color: isConnected ? '#9CA3AF' : EDGE_COLOR,
        hidden: !isConnected,
      }
    },
  })

  return { graph, sigma, glowIntensities, nodeIndex }
}

function applyAlpha(color: string, alpha: number): string {
  // For rgb() format
  if (color.startsWith('rgb(')) {
    return color.replace('rgb(', 'rgba(').replace(')', `,${alpha})`)
  }
  // For hex format
  if (color.startsWith('#')) {
    const a = Math.round(alpha * 255)
      .toString(16)
      .padStart(2, '0')
    return color.length === 7 ? color + a : color.slice(0, 7) + a
  }
  return color
}
