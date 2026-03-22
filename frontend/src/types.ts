export interface GraphNode {
  id: string
  type: 'S' | 'I' | 'M'
  region: 'head' | 'body' | 'tail'
  degree_centrality: number
  pos_x: number
  pos_y: number
  in_degree: number
  out_degree: number
  is_replacement?: boolean
  is_ghosted?: boolean
  replacement_for?: string | null
  replaced_by?: string | null
}

export interface GraphEdge {
  source: string
  target: string
  chemical_weight: number
  gap_weight: number
  weight: number
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface SpikeData {
  duration_ms: number
  spike_trains: Record<string, number[]>
  neuron_count: number
  active_count: number
}

export interface AppState {
  timescale: number
  selectedNeuron: string | null
  hoveredNeuron: string | null
  activeFaultyNeuron: string | null
  activeReplacementNeuron: string | null
  replacementStatus: 'in_progress' | 'completed' | null
  sweepStatus: 'idle' | 'connecting' | 'receiving' | 'done' | 'error'
  sweepProgress: number
  sweepError: string | null
}

// ─── Replacement sweep metrics ───

export type SweepStrategy = 'random' | 'hub_first' | 'periphery_first'

export interface StepMetrics {
  step_index: number
  neuron_being_replaced: string
  edges_migrated: number
  total_edges: number
  kuramoto_r: number
  pca_deviation: number
  pca_sigma: number
  voltage_entropy: number
  firing_rate_mean: number
  synchrony: number
  pathway_fidelity_val: number
  ou_convergence: number | null
}

export interface SweepBaseline {
  mean_firing_rate: number
  std_firing_rate: number
  synchrony: number
  entropy: number
  pathway_fidelity: number
  kuramoto_r: number
  pca_deviation: number
  pca_sigma: number
  voltage_entropy: number
}

export interface SweepBaselineEvent {
  type: 'baseline'
  data: SweepBaseline
  strategy: string
  neuron_model: string
  replacement_order: string[]
  total_steps: number
}

export interface SweepStepEvent {
  type: 'step'
  data: StepMetrics
}

export interface SweepDoneEvent {
  type: 'done'
}

export interface LiveFiringSummary {
  overall_mean_hz: number
  sensory_mean_hz: number
  interneuron_mean_hz: number
  motor_mean_hz: number
  active_fraction: number
}

export interface TopFiringNeuron {
  name: string
  firing_rate_hz: number
}

export interface LiveSimulationResponse {
  node_count: number
  population_spike_rate_hz: number
  firing_summary_hz: LiveFiringSummary
  firing_rates_hz_by_node: Record<string, number>
  top_firing_neurons: TopFiringNeuron[]
}

export interface ReplacementEdgeMigration {
  migration_id: string
  old_source: string
  old_target: string
  new_source: string
  new_target: string
  chemical_weight: number
  gap_weight: number
}

export type ReplacementMode = 'instant' | 'ou'

export interface ReplacementSession {
  session_id: string
  faulty_neuron: string
  replacement_neuron: string
  status: 'in_progress' | 'completed'
  pending_count: number
  completed_count: number
  next_edge: ReplacementEdgeMigration | null
  completed_edges: ReplacementEdgeMigration[]
  mode: ReplacementMode
  ou_params?: { theta: number; sigma: number }
  ou_convergence?: number
}
