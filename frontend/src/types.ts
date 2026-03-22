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
  behaviourStatus: 'idle' | 'running' | 'done' | 'error'
  behaviourError: string | null
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

// ─── Behaviour assay ───

export type GraphSource = 'canonical' | 'replacement'

export interface BehaviorPerformanceRequest {
  behavior?: 'forward_locomotion'
  graph_source?: GraphSource
  engine?: string
  neuron_model?: string
  burn_in_ms?: number
  duration_ms?: number
  integration_step_ms?: number
  include_traces?: boolean
  protocol?: {
    targets?: string[]
    amplitude_pA?: number
    period_ms?: number
    duty_cycle?: number
    start_ms?: number
    stop_ms?: number | null
  }
}

export interface PopulationRateStats {
  mean_rate_hz: number
  median_rate_hz: number
  active_fraction: number
  per_neuron_rate_hz: Record<string, number>
}

export interface MotorNeuronFiringPatterns {
  b_type: PopulationRateStats
  d_type: PopulationRateStats
  dorsal_ventral_correlation: number
  dorsal_ventral_anti_phase_index: number
  head_to_tail_delay_ms_per_segment: number
  head_to_tail_fit_r2: number
  segments_with_activity: number
}

export interface MuscleCaWaveProxy {
  available: boolean
  is_proxy: boolean
  bin_ms: number
  tau_ms: number
  segment_count: number
  wave_travel_delay_ms_per_segment: number
  wave_travel_fit_r2: number
  adjacent_segment_coherence: number
  dorsal_to_ventral_phase_lag_ms: number
  mean_wave_amplitude: number
  time_ms?: number[]
  dorsal_ca_proxy?: Record<string, number[]>
  ventral_ca_proxy?: Record<string, number[]>
}

export interface BehaviorPerformanceResponse {
  behavior: {
    behavior_id: string
    description: string
    canonical_circuit: {
      command_interneurons: string[]
      b_type_motor_neurons: string[]
      d_type_motor_neurons: string[]
      stimulus_entry_targets: string[]
    }
  }
  input_protocol: Record<string, unknown>
  behavioral_readout: {
    motor_neuron_firing_patterns: MotorNeuronFiringPatterns
    muscle_ca2_wave_proxy: MuscleCaWaveProxy
    body_kinematics: { available: boolean; reason?: string }
  }
  assay_context: {
    engine: string
    neuron_model: string
    burn_in_ms: number
    duration_ms: number
    integration_step_ms: number
    graph_source?: string
  }
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
