import type {
  GraphData,
  LiveSimulationResponse,
  ReplacementSession,
  SpikeData,
  SweepBaselineEvent,
  SweepStepEvent,
  SweepStrategy,
} from './types'

export async function fetchGraphData(): Promise<GraphData> {
  const response = await fetch('/api/connectome/graph')
  if (!response.ok) {
    throw new Error(`Failed to fetch graph: ${response.status}`)
  }
  return response.json() as Promise<GraphData>
}

export async function fetchLiveSimulation(): Promise<LiveSimulationResponse> {
  const response = await fetch('/api/simulation/live')
  if (!response.ok) {
    throw new Error(`Failed to fetch live simulation: ${response.status}`)
  }
  return response.json() as Promise<LiveSimulationResponse>
}

export async function fetchSpikeTrains(): Promise<SpikeData> {
  const response = await fetch('/api/simulation/spikes')
  if (!response.ok) {
    throw new Error(`Failed to fetch spikes: ${response.status}`)
  }
  return response.json() as Promise<SpikeData>
}

export async function fetchReplacementGraph(): Promise<GraphData> {
  const response = await fetch('/api/replacement/graph')
  if (!response.ok) {
    throw new Error(`Failed to fetch replacement graph: ${response.status}`)
  }
  return response.json() as Promise<GraphData>
}

export async function fetchRandomFaultyNeurons(
  count = 1,
  seed?: number,
): Promise<string[]> {
  const search = new URLSearchParams()
  search.set('count', String(count))
  if (seed !== undefined) {
    search.set('seed', String(seed))
  }
  const response = await fetch(`/api/replacement/faulty/random?${search.toString()}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch faulty neurons: ${response.status}`)
  }
  const payload = (await response.json()) as { neurons: string[] }
  return payload.neurons
}

export async function startReplacement(input: {
  faultyNeuron?: string
  edgeOrder?: 'random' | 'deterministic'
  seed?: number
}): Promise<ReplacementSession> {
  const response = await fetch('/api/replacement/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      faulty_neuron: input.faultyNeuron ?? null,
      edge_order: input.edgeOrder ?? 'random',
      seed: input.seed ?? null,
    }),
  })
  if (!response.ok) {
    throw new Error(`Failed to start replacement: ${response.status}`)
  }
  const payload = (await response.json()) as { session: ReplacementSession }
  return payload.session
}

export async function stepReplacement(
  sessionId: string,
  edgesToMigrate = 1,
): Promise<ReplacementSession> {
  const response = await fetch('/api/replacement/step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      edges_to_migrate: edgesToMigrate,
    }),
  })
  if (!response.ok) {
    throw new Error(`Failed to step replacement: ${response.status}`)
  }
  const payload = (await response.json()) as { session: ReplacementSession }
  return payload.session
}

export async function resetReplacement(): Promise<void> {
  const response = await fetch('/api/replacement/reset', { method: 'POST' })
  if (!response.ok) {
    throw new Error(`Failed to reset replacement graph: ${response.status}`)
  }
}

// ─── Replacement sweep SSE stream ───

export interface SweepStreamParams {
  fraction?: number
  strategy?: SweepStrategy
  neuronModel?: string
  stepMs?: number
  edgesPerStep?: number
  seed?: number
}

export interface SweepStreamCallbacks {
  onBaseline: (event: SweepBaselineEvent) => void
  onStep: (event: SweepStepEvent) => void
  onDone: () => void
  onError: (error: string) => void
}

export function connectSweepStream(
  params: SweepStreamParams,
  callbacks: SweepStreamCallbacks,
): { close: () => void } {
  const search = new URLSearchParams()
  if (params.fraction !== undefined) search.set('fraction', String(params.fraction))
  if (params.strategy) search.set('strategy', params.strategy)
  if (params.neuronModel) search.set('neuron_model', params.neuronModel)
  if (params.stepMs !== undefined) search.set('step_ms', String(params.stepMs))
  if (params.edgesPerStep !== undefined) search.set('edges_per_step', String(params.edgesPerStep))
  if (params.seed !== undefined) search.set('seed', String(params.seed))

  const url = `/api/simulation/replacement-sweep/stream?${search.toString()}`
  const source = new EventSource(url)

  source.addEventListener('baseline', (e: MessageEvent) => {
    callbacks.onBaseline(JSON.parse(e.data) as SweepBaselineEvent)
  })
  source.addEventListener('step', (e: MessageEvent) => {
    callbacks.onStep(JSON.parse(e.data) as SweepStepEvent)
  })
  source.addEventListener('done', () => {
    source.close()
    callbacks.onDone()
  })
  source.onerror = () => {
    source.close()
    callbacks.onError('Connection to sweep stream lost')
  }

  return { close: () => source.close() }
}
