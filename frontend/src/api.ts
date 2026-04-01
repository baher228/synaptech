import type {
  BehaviorPerformanceRequest,
  BehaviorPerformanceResponse,
  GraphData,
  LiveSimulationResponse,
  ReplacementMode,
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
  mode?: ReplacementMode
  theta?: number
  sigma?: number
}): Promise<ReplacementSession> {
  const response = await fetch('/api/replacement/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      faulty_neuron: input.faultyNeuron ?? null,
      edge_order: input.edgeOrder ?? 'random',
      seed: input.seed ?? null,
      mode: input.mode ?? 'instant',
      theta: input.theta ?? null,
      sigma: input.sigma ?? null,
    }),
  })
  if (!response.ok) {
    throw new Error(`Failed to start replacement: ${response.status}`)
  }
  const payload = (await response.json()) as { session: ReplacementSession }
  return payload.session
}

export async function tickOU(sessionId: string): Promise<ReplacementSession> {
  const response = await fetch('/api/replacement/tick-ou', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  })
  if (!response.ok) {
    throw new Error(`Failed to tick OU replacement: ${response.status}`)
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

// ─── Behaviour assay ───

export async function fetchBehaviorPerformance(
  req: BehaviorPerformanceRequest = {},
): Promise<BehaviorPerformanceResponse> {
  const response = await fetch('/api/simulation/behavior/performance', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      behavior: req.behavior ?? 'forward_locomotion',
      graph_source: req.graph_source ?? 'canonical',
      engine: req.engine ?? 'brian2',
      neuron_model: req.neuron_model ?? 'lif',
      burn_in_ms: req.burn_in_ms ?? 1500.0,
      duration_ms: req.duration_ms ?? 4000.0,
      integration_step_ms: req.integration_step_ms ?? 10.0,
      include_traces: req.include_traces ?? true,
      protocol: req.protocol ?? {},
    }),
  })
  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText)
    throw new Error(`Behavior assay failed (${response.status}): ${detail}`)
  }
  return response.json() as Promise<BehaviorPerformanceResponse>
}

// ─── Replacement sweep SSE stream ───

export interface SweepStreamParams {
  fraction?: number
  strategy?: SweepStrategy
  neuronModel?: string
  stepMs?: number
  edgesPerStep?: number
  seed?: number
  replacementMode?: ReplacementMode
  ouTheta?: number
  ouSigma?: number
  integration?: string
  batchSize?: number
  settleMs?: number
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
  if (params.replacementMode) search.set('replacement_mode', params.replacementMode)
  if (params.ouTheta !== undefined) search.set('ou_theta', String(params.ouTheta))
  if (params.ouSigma !== undefined) search.set('ou_sigma', String(params.ouSigma))
  if (params.integration) search.set('integration', params.integration)
  if (params.batchSize !== undefined) search.set('batch_size', String(params.batchSize))
  if (params.settleMs !== undefined) search.set('settle_ms', String(params.settleMs))

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
