import type { GraphNode } from '../types'

export interface NormalizedPositions {
  [nodeId: string]: { x: number; y: number }
}

export function normalizePositions(nodes: GraphNode[]): NormalizedPositions {
  if (nodes.length === 0) return {}

  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity

  for (const node of nodes) {
    if (node.pos_x < minX) minX = node.pos_x
    if (node.pos_x > maxX) maxX = node.pos_x
    if (node.pos_y < minY) minY = node.pos_y
    if (node.pos_y > maxY) maxY = node.pos_y
  }

  const rangeX = maxX - minX || 1
  const rangeY = maxY - minY || 1
  const scale = Math.max(rangeX, rangeY)
  const centerX = (minX + maxX) / 2
  const centerY = (minY + maxY) / 2

  const positions: NormalizedPositions = {}
  for (const node of nodes) {
    positions[node.id] = {
      x: ((node.pos_x - centerX) / scale) * 200,
      y: ((node.pos_y - centerY) / scale) * 200,
    }
  }
  return positions
}
