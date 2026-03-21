import type { GraphData, LiveSimulationResponse } from './types'

export async function fetchGraphData(): Promise<GraphData> {
  const response = await fetch('/api/connectome/graph')
  if (!response.ok) {
    throw new Error(`Failed to fetch graph: ${response.status}`)
  }
  return response.json() as Promise<GraphData>
}

export async function fetchLiveSimulation(
  params: { durationMs?: number; burnInMs?: number; seed?: number } = {},
): Promise<LiveSimulationResponse> {
  const search = new URLSearchParams()
  if (params.durationMs !== undefined) {
    search.set('duration_ms', String(params.durationMs))
  }
  if (params.burnInMs !== undefined) {
    search.set('burn_in_ms', String(params.burnInMs))
  }
  if (params.seed !== undefined) {
    search.set('seed', String(params.seed))
  }

  const query = search.toString()
  const response = await fetch(`/api/simulation/live${query ? `?${query}` : ''}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch live simulation: ${response.status}`)
  }
  return response.json() as Promise<LiveSimulationResponse>
}
