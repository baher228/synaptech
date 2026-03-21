import type { GraphData } from './types'

export async function fetchGraphData(): Promise<GraphData> {
  const response = await fetch('/api/connectome/graph')
  if (!response.ok) {
    throw new Error(`Failed to fetch graph: ${response.status}`)
  }
  return response.json() as Promise<GraphData>
}
