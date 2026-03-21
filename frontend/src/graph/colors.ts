export const NEURON_COLORS: Record<string, string> = {
  S: '#0EA5E9', // sensory — sky blue
  I: '#8B5CF6', // interneuron — violet
  M: '#F59E0B', // motor — amber
}

export const NEURON_GLOW_COLORS: Record<string, string> = {
  S: '#7DD3FC', // sensory glow — light sky
  I: '#C4B5FD', // interneuron glow — light violet
  M: '#FCD34D', // motor glow — light amber
}

export const EDGE_COLOR = '#D1D5DB'
export const EDGE_HIGHLIGHT_COLOR = '#9CA3AF'

export const TYPE_LABELS: Record<string, string> = {
  S: 'Sensory',
  I: 'Interneuron',
  M: 'Motor',
}

export function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const n = parseInt(hex.slice(1), 16)
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 }
}

export function lerpColor(from: string, to: string, t: number): string {
  const a = hexToRgb(from)
  const b = hexToRgb(to)
  const r = Math.round(a.r + (b.r - a.r) * t)
  const g = Math.round(a.g + (b.g - a.g) * t)
  const bl = Math.round(a.b + (b.b - a.b) * t)
  return `rgb(${r},${g},${bl})`
}
