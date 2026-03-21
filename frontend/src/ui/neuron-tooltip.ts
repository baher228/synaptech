import type Sigma from 'sigma'
import type { RendererContext } from '../graph/renderer'
import { state, on } from '../state'
import { NEURON_COLORS, TYPE_LABELS } from '../graph/colors'

let tooltipEl: HTMLElement | null = null

export function createNeuronTooltip(
  container: HTMLElement,
  ctx: RendererContext,
): void {
  tooltipEl = document.createElement('div')
  tooltipEl.className = 'neuron-tooltip'
  tooltipEl.style.display = 'none'
  container.appendChild(tooltipEl)

  on('selection', () => updateTooltip(ctx))

  // Reposition on camera move
  ctx.sigma.on('afterRender', () => {
    if (state.selectedNeuron && tooltipEl?.style.display !== 'none') {
      positionTooltip(ctx.sigma, state.selectedNeuron)
    }
  })
}

function updateTooltip(ctx: RendererContext): void {
  if (!tooltipEl) return

  if (!state.selectedNeuron) {
    tooltipEl.style.display = 'none'
    return
  }

  const node = ctx.nodeIndex.get(state.selectedNeuron)
  if (!node) {
    tooltipEl.style.display = 'none'
    return
  }

  const color = NEURON_COLORS[node.type] || '#888'
  const typeLabel = TYPE_LABELS[node.type] || node.type

  tooltipEl.innerHTML = `
    <div class="tooltip-header">
      <span class="tooltip-name">${node.id}</span>
      <span class="tooltip-badge" style="--badge-color: ${color}">
        <span class="tooltip-dot" style="background: ${color}"></span>
        ${typeLabel}
      </span>
    </div>
    <div class="tooltip-stats">
      <div class="tooltip-stat">
        <span class="tooltip-stat-label">Region</span>
        <span class="tooltip-stat-value">${capitalize(node.region)}</span>
      </div>
      <div class="tooltip-stat">
        <span class="tooltip-stat-label">Centrality</span>
        <span class="tooltip-stat-value">${node.degree_centrality.toFixed(3)}</span>
      </div>
      <div class="tooltip-stat">
        <span class="tooltip-stat-label">Connections</span>
        <span class="tooltip-stat-value">${node.in_degree} in / ${node.out_degree} out</span>
      </div>
    </div>
  `

  tooltipEl.style.display = 'block'
  positionTooltip(ctx.sigma, state.selectedNeuron)
}

function positionTooltip(sigma: Sigma, nodeId: string): void {
  if (!tooltipEl) return

  const pos = sigma.graphToViewport(
    sigma.getGraph().getNodeAttributes(nodeId) as { x: number; y: number },
  )

  const rect = tooltipEl.getBoundingClientRect()
  const viewW = window.innerWidth
  const viewH = window.innerHeight
  const offset = 16

  let left = pos.x + offset
  let top = pos.y - rect.height / 2

  // Flip if too close to right edge
  if (left + rect.width > viewW - 20) {
    left = pos.x - rect.width - offset
  }
  // Clamp vertically
  if (top < 10) top = 10
  if (top + rect.height > viewH - 10) top = viewH - rect.height - 10

  tooltipEl.style.left = `${left}px`
  tooltipEl.style.top = `${top}px`
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}
