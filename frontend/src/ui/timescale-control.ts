import { state, emit } from '../state'

const SPEEDS = [0.1, 0.5, 1, 10] as const

export function createTimescaleControl(container: HTMLElement): void {
  const el = document.createElement('div')
  el.className = 'timescale-control'
  el.innerHTML = `
    <span class="timescale-label">Speed</span>
    <div class="timescale-buttons">
      ${SPEEDS.map(
        (s) =>
          `<button class="speed-btn${s === state.timescale ? ' active' : ''}" data-speed="${s}">${s}x</button>`,
      ).join('')}
    </div>
  `

  el.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('.speed-btn')
    if (!btn) return

    const speed = parseFloat(btn.dataset.speed!)
    state.timescale = speed
    emit('timescale')

    el.querySelectorAll('.speed-btn').forEach((b) => b.classList.remove('active'))
    btn.classList.add('active')
  })

  container.appendChild(el)
}
