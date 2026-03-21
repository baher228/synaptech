import { NEURON_COLORS, TYPE_LABELS } from '../graph/colors'

export function createLegend(container: HTMLElement): void {
  const el = document.createElement('div')
  el.className = 'legend'
  el.innerHTML = Object.entries(TYPE_LABELS)
    .map(
      ([key, label]) => `
      <div class="legend-item">
        <span class="legend-dot" style="background: ${NEURON_COLORS[key]}"></span>
        <span class="legend-label">${label}</span>
      </div>
    `,
    )
    .join('')
  container.appendChild(el)
}
