export interface GraphNode {
  id: string
  type: 'S' | 'I' | 'M'
  region: 'head' | 'body' | 'tail'
  degree_centrality: number
  pos_x: number
  pos_y: number
  in_degree: number
  out_degree: number
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

export interface AppState {
  timescale: number
  selectedNeuron: string | null
  hoveredNeuron: string | null
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
