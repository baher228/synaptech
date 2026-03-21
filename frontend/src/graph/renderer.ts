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
  const glowIntensities = new Map<string, number>()
  const nodeIndex = new Map<string, GraphNode>()

  applyGraphData(graph, glowIntensities, nodeIndex, data)

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
      const isReplacement = Boolean((attrs as any).isReplacement)
      const isGhosted = Boolean((attrs as any).isGhosted)
      const glowColor = isReplacement
        ? '#6EE7B7'
        : NEURON_GLOW_COLORS[neuronType] || '#FFF'

      const isSelected = state.selectedNeuron === nodeId
      const isHovered = state.hoveredNeuron === nodeId
      const isActiveFaulty =
        state.replacementStatus === 'in_progress' && state.activeFaultyNeuron === nodeId
      const isActiveReplacement =
        state.replacementStatus === 'in_progress' &&
        state.activeReplacementNeuron === nodeId
      const hasSelection = state.selectedNeuron !== null
      const isNeighborOfSelected =
        hasSelection &&
        state.selectedNeuron !== nodeId &&
        (graph.hasEdge(state.selectedNeuron!, nodeId) || graph.hasEdge(nodeId, state.selectedNeuron!))

      // Glow effect: boost size and shift color toward glow color
      let size = baseSize + glow * baseSize * 1.2
      let color = glow > 0.01 ? lerpColor(baseColor, glowColor, glow * 0.7) : baseColor

      if (isGhosted) {
        color = applyAlpha(color, 0.45)
      }

      // Selection/hover dimming
      let alpha = 1
      if (hasSelection && !isSelected && !isNeighborOfSelected) {
        alpha = 0.15
      }
      if (isSelected || isHovered) {
        size *= 1.15
      }
      if (isActiveFaulty) {
        color = lerpColor(color, '#F97316', 0.55)
        size *= 1.12
      }
      if (isActiveReplacement) {
        color = lerpColor(color, '#10B981', 0.55)
        size *= 1.12
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
      const faulty = state.activeFaultyNeuron
      const replacement = state.activeReplacementNeuron
      const handoffActive =
        state.replacementStatus === 'in_progress' &&
        Boolean(faulty) &&
        Boolean(replacement)

      const [source, target] = graph.extremities(edgeId)
      const touchesReplacement =
        handoffActive && (source === replacement || target === replacement)
      const touchesFaulty = handoffActive && (source === faulty || target === faulty)

      if (!hasSelection) {
        if (touchesReplacement) {
          return { ...attrs, color: '#10B981', size: attrs.size * 1.35, hidden: false }
        }
        if (touchesFaulty) {
          return { ...attrs, color: '#F97316', size: attrs.size * 1.2, hidden: false }
        }
        return { ...attrs, color: EDGE_COLOR, hidden: false }
      }

      const isConnected = source === state.selectedNeuron || target === state.selectedNeuron

      if (touchesReplacement) {
        return {
          ...attrs,
          color: '#10B981',
          size: attrs.size * 1.35,
          hidden: !isConnected,
        }
      }
      if (touchesFaulty) {
        return {
          ...attrs,
          color: '#F97316',
          size: attrs.size * 1.2,
          hidden: !isConnected,
        }
      }

      return {
        ...attrs,
        color: isConnected ? '#9CA3AF' : EDGE_COLOR,
        hidden: !isConnected,
      }
    },
  })

  return { graph, sigma, glowIntensities, nodeIndex }
}

export function replaceRendererData(
  ctx: RendererContext,
  data: GraphData,
): void {
  applyGraphData(ctx.graph, ctx.glowIntensities, ctx.nodeIndex, data)
  ctx.sigma.refresh()
}

function applyGraphData(
  graph: Graph,
  glowIntensities: Map<string, number>,
  nodeIndex: Map<string, GraphNode>,
  data: GraphData,
): void {
  const previousGlow = new Map(glowIntensities)
  graph.clear()
  glowIntensities.clear()
  nodeIndex.clear()

  const positions = normalizePositions(data.nodes)
  const nodesById = new Map<string, GraphNode>(data.nodes.map((n) => [n.id, n]))
  for (const node of data.nodes) {
    const pos = resolveDisplayPosition(node, positions, nodesById)
    const baseColor = resolveNodeColor(node)
    const sizeMultiplier = node.is_ghosted ? 0.65 : node.is_replacement ? 1.05 : 1
    const baseSize = (3 + node.degree_centrality * 18) * sizeMultiplier
    graph.addNode(node.id, {
      x: pos.x,
      y: pos.y,
      size: baseSize,
      color: baseColor,
      label: node.id,
      type: 'circle',
      baseSize,
      baseColor,
      neuronType: node.type,
      isReplacement: Boolean(node.is_replacement),
      isGhosted: Boolean(node.is_ghosted),
    })
    glowIntensities.set(node.id, previousGlow.get(node.id) ?? 0)
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
}

function resolveNodeColor(node: GraphNode): string {
  if (node.is_ghosted) return '#9CA3AF'
  if (node.is_replacement) return '#10B981'
  return NEURON_COLORS[node.type] || '#888'
}

function resolveDisplayPosition(
  node: GraphNode,
  positions: Record<string, { x: number; y: number }>,
  nodesById: Map<string, GraphNode>,
): { x: number; y: number } {
  const direct = positions[node.id]
  if (!direct) return { x: 0, y: 0 }

  if (!node.is_replacement || !node.replacement_for) {
    return direct
  }

  const parent = positions[node.replacement_for]
  if (!parent) {
    return direct
  }
  const parentNode = nodesById.get(node.replacement_for)
  const mergedIntoParentPosition =
    parentNode?.is_ghosted === true && parentNode.replaced_by === node.id
  if (mergedIntoParentPosition) {
    return parent
  }

  // Keep replacement visibly next to the faulty neuron while staying nearby.
  const minRadius = 7
  const maxRadius = 11
  const dx = direct.x - parent.x
  const dy = direct.y - parent.y
  const dist = Math.hypot(dx, dy)

  // If backend-provided offset is already good (visible + nearby), keep it.
  if (dist >= minRadius && dist <= maxRadius) {
    return direct
  }

  const repIndex = extractReplacementIndex(node.id)
  const fallbackAngle = repIndex * 2.399963229728653
  const angle = dist > 1e-6 ? Math.atan2(dy, dx) : fallbackAngle
  const radius = minRadius + (repIndex % 3) * 2
  return {
    x: parent.x + Math.cos(angle) * radius,
    y: parent.y + Math.sin(angle) * radius,
  }
}

function extractReplacementIndex(nodeId: string): number {
  const match = /__rep_(\d+)$/.exec(nodeId)
  if (!match) return 0
  const n = parseInt(match[1], 10)
  return Number.isFinite(n) ? n : 0
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
