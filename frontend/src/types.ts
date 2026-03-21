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
