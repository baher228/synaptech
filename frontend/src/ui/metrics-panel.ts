import { connectSweepStream } from '../api'
import { state, emit } from '../state'
import { createSweepChart, type SweepChart } from './sweep-chart'
import type {
  LiveSimulationResponse,
  SweepBaselineEvent,
  SweepStepEvent,
  SweepStrategy,
} from '../types'

type PanelMode = 'idle' | 'sweep' | 'done'

export function createMetricsPanel(container: HTMLElement): {
  onLiveUpdate: (payload: LiveSimulationResponse) => void
} {
  const el = document.createElement('div')
  el.className = 'metrics-panel'
  container.appendChild(el)

  // ─── Local state ───
  let mode: PanelMode = 'idle'
  let strategy: SweepStrategy = 'hub_first'
  let fraction = 0.05
  let activeStream: { close: () => void } | null = null
  let totalSteps = 0
  let currentStep = 0
  let currentNeuron = ''
  let statusText = 'Live activity'

  // ─── Persistent DOM structure ───
  const controlsDiv = document.createElement('div')
  controlsDiv.className = 'metrics-controls-area'
  el.appendChild(controlsDiv)

  const chartHost = document.createElement('div')
  chartHost.className = 'metrics-charts'

  // Idle mode: single chart with 3 series
  const idleChartDiv = document.createElement('div')
  idleChartDiv.className = 'metrics-chart-slot'
  chartHost.appendChild(idleChartDiv)

  // Sweep mode: 3 separate charts
  const kuramotoDiv = document.createElement('div')
  kuramotoDiv.className = 'metrics-chart-slot'
  const pcaDiv = document.createElement('div')
  pcaDiv.className = 'metrics-chart-slot'
  const entropyDiv = document.createElement('div')
  entropyDiv.className = 'metrics-chart-slot'

  el.appendChild(chartHost)

  // ─── Chart instances ───
  let idleChart: SweepChart | null = null
  let sweepCharts: {
    kuramoto: SweepChart
    pca: SweepChart
    entropy: SweepChart
  } | null = null
  let liveTickCount = 0

  // ─── Idle chart ───
  function ensureIdleChart(): void {
    if (idleChart) return
    chartHost.innerHTML = ''
    chartHost.appendChild(idleChartDiv)
    idleChartDiv.innerHTML = ''
    const w = Math.max(200, el.clientWidth - 32)
    idleChart = createSweepChart({
      title: 'Firing Rates (Hz)',
      series: [
        { label: 'Sensory', color: '#0EA5E9' },
        { label: 'Inter', color: '#8B5CF6' },
        { label: 'Motor', color: '#F59E0B' },
      ],
      container: idleChartDiv,
      width: w,
      height: 150,
    })
  }

  function destroyIdleChart(): void {
    if (idleChart) {
      idleChart.destroy()
      idleChart = null
    }
    idleChartDiv.innerHTML = ''
  }

  // ─── Sweep charts ───
  function initSweepCharts(baseline: SweepBaselineEvent): void {
    destroySweepCharts()
    destroyIdleChart()
    chartHost.innerHTML = ''
    chartHost.append(kuramotoDiv, pcaDiv, entropyDiv)
    kuramotoDiv.innerHTML = ''
    pcaDiv.innerHTML = ''
    entropyDiv.innerHTML = ''

    const w = Math.max(200, el.clientWidth - 32)
    const h = 120

    sweepCharts = {
      kuramoto: createSweepChart({
        title: 'Kuramoto R',
        series: [{ label: 'R', color: '#0EA5E9' }],
        baselineValue: baseline.data.kuramoto_r || undefined,
        container: kuramotoDiv,
        width: w,
        height: h,
      }),
      pca: createSweepChart({
        title: 'PCA Deviation',
        series: [{ label: 'D', color: '#8B5CF6' }],
        baselineValue: baseline.data.pca_deviation || undefined,
        container: pcaDiv,
        width: w,
        height: h,
      }),
      entropy: createSweepChart({
        title: 'Voltage Entropy',
        series: [{ label: 'bits', color: '#F59E0B' }],
        baselineValue: baseline.data.voltage_entropy || undefined,
        container: entropyDiv,
        width: w,
        height: h,
      }),
    }
  }

  function destroySweepCharts(): void {
    if (sweepCharts) {
      sweepCharts.kuramoto.destroy()
      sweepCharts.pca.destroy()
      sweepCharts.entropy.destroy()
      sweepCharts = null
    }
    kuramotoDiv.innerHTML = ''
    pcaDiv.innerHTML = ''
    entropyDiv.innerHTML = ''
  }

  // ─── Render controls ───
  function renderControls(): void {
    const isRunning =
      state.sweepStatus === 'connecting' || state.sweepStatus === 'receiving'
    const progressPct = Math.round(state.sweepProgress * 100)

    controlsDiv.innerHTML = `
      <div class="metrics-title">Replacement Metrics</div>
      <div class="metrics-row">
        <label class="metrics-label">Strategy
          <select class="metrics-select" data-field="strategy" ${isRunning ? 'disabled' : ''}>
            <option value="random" ${strategy === 'random' ? 'selected' : ''}>Random</option>
            <option value="hub_first" ${strategy === 'hub_first' ? 'selected' : ''}>Hub-first</option>
            <option value="periphery_first" ${strategy === 'periphery_first' ? 'selected' : ''}>Periphery-first</option>
          </select>
        </label>
        <label class="metrics-label">Fraction
          <span class="metrics-fraction-val">${(fraction * 100).toFixed(0)}%</span>
          <input type="range" class="metrics-range" data-field="fraction"
            min="0.01" max="0.30" step="0.01" value="${fraction}"
            ${isRunning ? 'disabled' : ''} />
        </label>
      </div>
      <div class="metrics-row">
        ${
          isRunning
            ? `<button class="metrics-btn metrics-btn-cancel" data-action="cancel">Cancel</button>`
            : mode === 'done'
              ? `<button class="metrics-btn metrics-btn-start" data-action="start">New Sweep</button>
                 <button class="metrics-btn metrics-btn-reset" data-action="reset">Reset</button>`
              : `<button class="metrics-btn metrics-btn-start" data-action="start">Start Sweep</button>`
        }
      </div>
      <div class="metrics-line">${statusText}</div>
      ${
        isRunning || mode === 'done'
          ? `<div class="metrics-progress"><div class="metrics-progress-bar" style="width:${progressPct}%"></div></div>`
          : ''
      }
    `
  }

  // ─── Sweep lifecycle ───
  function startSweep(): void {
    if (activeStream) activeStream.close()
    destroySweepCharts()
    destroyIdleChart()
    mode = 'sweep'

    state.sweepStatus = 'connecting'
    state.sweepProgress = 0
    state.sweepError = null
    statusText = 'Capturing baseline from live simulation...'
    renderControls()

    activeStream = connectSweepStream(
      { fraction, strategy, edgesPerStep: 5 },
      {
        onBaseline(event: SweepBaselineEvent) {
          state.sweepStatus = 'receiving'
          totalSteps = event.total_steps
          currentStep = 0
          statusText = `Replacing ${event.replacement_order.length} neurons (${totalSteps} steps)...`
          initSweepCharts(event)
          renderControls()
        },
        onStep(event: SweepStepEvent) {
          currentStep = event.data.step_index + 1
          currentNeuron = event.data.neuron_being_replaced
          state.sweepProgress = totalSteps > 0 ? currentStep / totalSteps : 0
          statusText = `Step ${currentStep}/${totalSteps} — replacing ${currentNeuron}`

          sweepCharts?.kuramoto.addPoint(event.data.step_index, event.data.kuramoto_r)
          sweepCharts?.pca.addPoint(event.data.step_index, event.data.pca_deviation)
          sweepCharts?.entropy.addPoint(event.data.step_index, event.data.voltage_entropy)

          renderControls()
        },
        onDone() {
          mode = 'done'
          state.sweepStatus = 'done'
          state.sweepProgress = 1
          activeStream = null
          statusText = `Sweep complete — ${currentStep} steps`
          emit('sweep:done')
          renderControls()
        },
        onError(msg: string) {
          mode = 'done'
          state.sweepStatus = 'error'
          state.sweepError = msg
          activeStream = null
          statusText = `Error: ${msg}`
          emit('sweep:error')
          renderControls()
        },
      },
    )
  }

  function cancelSweep(): void {
    if (activeStream) {
      activeStream.close()
      activeStream = null
    }
    resetToIdle()
  }

  function resetToIdle(): void {
    destroySweepCharts()
    mode = 'idle'
    state.sweepStatus = 'idle'
    state.sweepProgress = 0
    statusText = 'Live activity'
    liveTickCount = 0
    ensureIdleChart()
    renderControls()
  }

  // ─── Event delegation ───
  el.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('.metrics-btn')
    if (!btn) return
    if (btn.dataset.action === 'start') startSweep()
    if (btn.dataset.action === 'cancel') cancelSweep()
    if (btn.dataset.action === 'reset') resetToIdle()
  })

  el.addEventListener('change', (e) => {
    const target = e.target as HTMLInputElement | HTMLSelectElement
    if (target.dataset.field === 'strategy') {
      strategy = target.value as SweepStrategy
    }
    if (target.dataset.field === 'fraction') {
      fraction = parseFloat(target.value)
      renderControls()
    }
  })

  el.addEventListener('input', (e) => {
    const target = e.target as HTMLInputElement
    if (target.dataset.field === 'fraction') {
      fraction = parseFloat(target.value)
      const valSpan = controlsDiv.querySelector('.metrics-fraction-val')
      if (valSpan) valSpan.textContent = `${(fraction * 100).toFixed(0)}%`
    }
  })

  // ─── Resize observer ───
  const ro = new ResizeObserver(() => {
    const w = Math.max(200, el.clientWidth - 32)
    idleChart?.resize(w, 150)
    if (sweepCharts) {
      sweepCharts.kuramoto.resize(w, 120)
      sweepCharts.pca.resize(w, 120)
      sweepCharts.entropy.resize(w, 120)
    }
  })
  ro.observe(el)

  // ─── Initialize in idle mode ───
  ensureIdleChart()
  renderControls()

  // ─── Public: live simulation data feed ───
  return {
    onLiveUpdate(payload: LiveSimulationResponse) {
      if (mode !== 'idle' || !idleChart) return
      const s = payload.firing_summary_hz
      idleChart.addPoint(
        liveTickCount++,
        s.sensory_mean_hz,
        s.interneuron_mean_hz,
        s.motor_mean_hz,
      )
    },
  }
}
