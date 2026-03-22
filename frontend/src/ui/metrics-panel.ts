import { connectSweepStream, fetchBehaviorPerformance } from '../api'
import { state, emit } from '../state'
import { createSweepChart, type SweepChart } from './sweep-chart'
import type {
  BehaviorPerformanceResponse,
  GraphSource,
  LiveSimulationResponse,
  SweepBaselineEvent,
  SweepStepEvent,
  SweepStrategy,
} from '../types'

type Tab = 'live' | 'sweep' | 'behaviour'
type SweepMode = 'idle' | 'sweep' | 'done'

export function createMetricsPanel(container: HTMLElement): {
  onLiveUpdate: (payload: LiveSimulationResponse) => void
} {
  const el = document.createElement('div')
  el.className = 'metrics-panel'
  container.appendChild(el)

  // ─── Tab state ───
  let activeTab: Tab = 'live'

  // ─── Sweep local state ───
  let sweepMode: SweepMode = 'idle'
  let strategy: SweepStrategy = 'hub_first'
  let fraction = 0.05
  let activeStream: { close: () => void } | null = null
  let totalSteps = 0
  let currentStep = 0
  let currentNeuron = ''
  let sweepStatusText = 'Live activity'

  // ─── Behaviour local state ───
  let behaviourGraphSource: GraphSource = 'canonical'
  let behaviourResult: BehaviorPerformanceResponse | null = null
  let behaviourStatusText = 'Ready'

  // ─── Persistent DOM structure ───
  const tabBar = document.createElement('div')
  tabBar.className = 'metrics-tab-bar'
  el.appendChild(tabBar)

  const contentArea = document.createElement('div')
  contentArea.className = 'metrics-content'
  el.appendChild(contentArea)

  // ─── Chart host containers (per tab) ───
  const liveChartHost = document.createElement('div')
  liveChartHost.className = 'metrics-charts'
  const liveChartDiv = document.createElement('div')
  liveChartDiv.className = 'metrics-chart-slot'

  const sweepControlsDiv = document.createElement('div')
  sweepControlsDiv.className = 'metrics-controls-area'
  const sweepChartHost = document.createElement('div')
  sweepChartHost.className = 'metrics-charts'
  const kuramotoDiv = document.createElement('div')
  kuramotoDiv.className = 'metrics-chart-slot'
  const pcaDiv = document.createElement('div')
  pcaDiv.className = 'metrics-chart-slot'
  const entropyDiv = document.createElement('div')
  entropyDiv.className = 'metrics-chart-slot'

  const behaviourControlsDiv = document.createElement('div')
  behaviourControlsDiv.className = 'metrics-controls-area'
  const behaviourStatsDiv = document.createElement('div')
  behaviourStatsDiv.className = 'behaviour-stats'
  const behaviourChartHost = document.createElement('div')
  behaviourChartHost.className = 'metrics-charts'
  const firingChartDiv = document.createElement('div')
  firingChartDiv.className = 'metrics-chart-slot'
  const caWaveChartDiv = document.createElement('div')
  caWaveChartDiv.className = 'metrics-chart-slot'

  // ─── Chart instances ───
  let idleChart: SweepChart | null = null
  let liveTickCount = 0
  let sweepCharts: {
    kuramoto: SweepChart
    pca: SweepChart
    entropy: SweepChart
  } | null = null
  let behaviourCharts: {
    firing: SweepChart
    caWave: SweepChart
  } | null = null

  // ─── Tab bar ───
  function renderTabBar(): void {
    tabBar.innerHTML = ''
    const tabs: { id: Tab; label: string }[] = [
      { id: 'live', label: 'Live' },
      { id: 'sweep', label: 'Sweep' },
      { id: 'behaviour', label: 'Behaviour' },
    ]
    for (const tab of tabs) {
      const btn = document.createElement('button')
      btn.className = `metrics-tab${activeTab === tab.id ? ' metrics-tab-active' : ''}`
      btn.textContent = tab.label
      btn.dataset.tab = tab.id
      tabBar.appendChild(btn)
    }
  }

  tabBar.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('.metrics-tab')
    if (!btn?.dataset.tab) return
    const newTab = btn.dataset.tab as Tab
    if (newTab === activeTab) return
    activeTab = newTab
    renderTabBar()
    renderActiveTab()
  })

  // ─── Render active tab content ───
  function renderActiveTab(): void {
    contentArea.innerHTML = ''
    if (activeTab === 'live') renderLiveTab()
    else if (activeTab === 'sweep') renderSweepTab()
    else if (activeTab === 'behaviour') renderBehaviourTab()
  }

  // ─── LIVE TAB ───
  function renderLiveTab(): void {
    liveChartHost.innerHTML = ''
    liveChartHost.appendChild(liveChartDiv)
    contentArea.appendChild(liveChartHost)
    ensureIdleChart()
  }

  function ensureIdleChart(): void {
    if (idleChart) return
    liveChartDiv.innerHTML = ''
    const w = Math.max(200, el.clientWidth - 32)
    idleChart = createSweepChart({
      title: 'Firing Rates (Hz)',
      series: [
        { label: 'Sensory', color: '#0EA5E9' },
        { label: 'Inter', color: '#8B5CF6' },
        { label: 'Motor', color: '#F59E0B' },
      ],
      container: liveChartDiv,
      width: w,
      height: 150,
    })
  }

  // ─── SWEEP TAB ───
  function renderSweepTab(): void {
    contentArea.appendChild(sweepControlsDiv)
    contentArea.appendChild(sweepChartHost)
    renderSweepControls()

    if (sweepCharts) {
      sweepChartHost.innerHTML = ''
      sweepChartHost.append(kuramotoDiv, pcaDiv, entropyDiv)
    } else if (sweepMode === 'idle') {
      sweepChartHost.innerHTML = ''
    }
  }

  function renderSweepControls(): void {
    const isRunning =
      state.sweepStatus === 'connecting' || state.sweepStatus === 'receiving'
    const progressPct = Math.round(state.sweepProgress * 100)

    sweepControlsDiv.innerHTML = `
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
            : sweepMode === 'done'
              ? `<button class="metrics-btn metrics-btn-start" data-action="start">New Sweep</button>
                 <button class="metrics-btn metrics-btn-reset" data-action="reset">Reset</button>`
              : `<button class="metrics-btn metrics-btn-start" data-action="start">Start Sweep</button>`
        }
      </div>
      <div class="metrics-line">${sweepStatusText}</div>
      ${
        isRunning || sweepMode === 'done'
          ? `<div class="metrics-progress"><div class="metrics-progress-bar" style="width:${progressPct}%"></div></div>`
          : ''
      }
    `
  }

  function initSweepCharts(baseline: SweepBaselineEvent): void {
    destroySweepCharts()
    sweepChartHost.innerHTML = ''
    sweepChartHost.append(kuramotoDiv, pcaDiv, entropyDiv)
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

  function startSweep(): void {
    if (activeStream) activeStream.close()
    destroySweepCharts()
    sweepMode = 'sweep'

    state.sweepStatus = 'connecting'
    state.sweepProgress = 0
    state.sweepError = null
    sweepStatusText = 'Capturing baseline from live simulation...'
    renderSweepControls()

    activeStream = connectSweepStream(
      { fraction, strategy, edgesPerStep: 5 },
      {
        onBaseline(event: SweepBaselineEvent) {
          state.sweepStatus = 'receiving'
          totalSteps = event.total_steps
          currentStep = 0
          sweepStatusText = `Replacing ${event.replacement_order.length} neurons (${totalSteps} steps)...`
          initSweepCharts(event)
          renderSweepControls()
        },
        onStep(event: SweepStepEvent) {
          currentStep = event.data.step_index + 1
          currentNeuron = event.data.neuron_being_replaced
          state.sweepProgress = totalSteps > 0 ? currentStep / totalSteps : 0
          sweepStatusText = `Step ${currentStep}/${totalSteps} — replacing ${currentNeuron}`

          sweepCharts?.kuramoto.addPoint(event.data.step_index, event.data.kuramoto_r)
          sweepCharts?.pca.addPoint(event.data.step_index, event.data.pca_deviation)
          sweepCharts?.entropy.addPoint(event.data.step_index, event.data.voltage_entropy)

          renderSweepControls()
        },
        onDone() {
          sweepMode = 'done'
          state.sweepStatus = 'done'
          state.sweepProgress = 1
          activeStream = null
          sweepStatusText = `Sweep complete — ${currentStep} steps`
          emit('sweep:done')
          renderSweepControls()
        },
        onError(msg: string) {
          sweepMode = 'done'
          state.sweepStatus = 'error'
          state.sweepError = msg
          activeStream = null
          sweepStatusText = `Error: ${msg}`
          emit('sweep:error')
          renderSweepControls()
        },
      },
    )
  }

  function cancelSweep(): void {
    if (activeStream) {
      activeStream.close()
      activeStream = null
    }
    destroySweepCharts()
    sweepMode = 'idle'
    state.sweepStatus = 'idle'
    state.sweepProgress = 0
    sweepStatusText = 'Cancelled'
    renderSweepControls()
  }

  function resetSweep(): void {
    destroySweepCharts()
    sweepMode = 'idle'
    state.sweepStatus = 'idle'
    state.sweepProgress = 0
    sweepStatusText = 'Ready'
    sweepChartHost.innerHTML = ''
    renderSweepControls()
  }

  // Sweep event delegation
  sweepControlsDiv.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('.metrics-btn')
    if (!btn) return
    if (btn.dataset.action === 'start') startSweep()
    if (btn.dataset.action === 'cancel') cancelSweep()
    if (btn.dataset.action === 'reset') resetSweep()
  })

  sweepControlsDiv.addEventListener('change', (e) => {
    const target = e.target as HTMLInputElement | HTMLSelectElement
    if (target.dataset.field === 'strategy') {
      strategy = target.value as SweepStrategy
    }
    if (target.dataset.field === 'fraction') {
      fraction = parseFloat(target.value)
      renderSweepControls()
    }
  })

  sweepControlsDiv.addEventListener('input', (e) => {
    const target = e.target as HTMLInputElement
    if (target.dataset.field === 'fraction') {
      fraction = parseFloat(target.value)
      const valSpan = sweepControlsDiv.querySelector('.metrics-fraction-val')
      if (valSpan) valSpan.textContent = `${(fraction * 100).toFixed(0)}%`
    }
  })

  // ─── BEHAVIOUR TAB ───
  function renderBehaviourTab(): void {
    contentArea.appendChild(behaviourControlsDiv)
    contentArea.appendChild(behaviourStatsDiv)
    contentArea.appendChild(behaviourChartHost)
    renderBehaviourControls()
    renderBehaviourStats()

    if (behaviourCharts) {
      behaviourChartHost.innerHTML = ''
      behaviourChartHost.append(firingChartDiv, caWaveChartDiv)
    }
  }

  function renderBehaviourControls(): void {
    const isRunning = state.behaviourStatus === 'running'

    behaviourControlsDiv.innerHTML = `
      <div class="metrics-row">
        <label class="metrics-label">Graph source
          <select class="metrics-select" data-field="graph-source" ${isRunning ? 'disabled' : ''}>
            <option value="canonical" ${behaviourGraphSource === 'canonical' ? 'selected' : ''}>Canonical</option>
            <option value="replacement" ${behaviourGraphSource === 'replacement' ? 'selected' : ''}>Replacement</option>
          </select>
        </label>
      </div>
      <div class="metrics-row">
        ${
          isRunning
            ? `<button class="metrics-btn" disabled>Running assay...</button>`
            : behaviourResult
              ? `<button class="metrics-btn metrics-btn-start" data-action="run-behaviour">Re-run Assay</button>
                 <button class="metrics-btn metrics-btn-reset" data-action="reset-behaviour">Clear</button>`
              : `<button class="metrics-btn metrics-btn-start" data-action="run-behaviour">Run Assay</button>`
        }
      </div>
      <div class="metrics-line">${behaviourStatusText}</div>
      ${isRunning ? '<div class="metrics-progress"><div class="metrics-progress-bar behaviour-progress-indeterminate"></div></div>' : ''}
    `
  }

  function renderBehaviourStats(): void {
    if (!behaviourResult) {
      behaviourStatsDiv.innerHTML = ''
      return
    }

    const motor = behaviourResult.behavioral_readout.motor_neuron_firing_patterns
    const wave = behaviourResult.behavioral_readout.muscle_ca2_wave_proxy

    behaviourStatsDiv.innerHTML = `
      <div class="behaviour-stat-grid">
        <div class="behaviour-stat">
          <span class="behaviour-stat-label">B-type mean rate</span>
          <span class="behaviour-stat-value">${motor.b_type.mean_rate_hz.toFixed(1)} Hz</span>
        </div>
        <div class="behaviour-stat">
          <span class="behaviour-stat-label">D-type mean rate</span>
          <span class="behaviour-stat-value">${motor.d_type.mean_rate_hz.toFixed(1)} Hz</span>
        </div>
        <div class="behaviour-stat">
          <span class="behaviour-stat-label">D/V correlation</span>
          <span class="behaviour-stat-value">${motor.dorsal_ventral_correlation.toFixed(3)}</span>
        </div>
        <div class="behaviour-stat">
          <span class="behaviour-stat-label">Anti-phase index</span>
          <span class="behaviour-stat-value">${motor.dorsal_ventral_anti_phase_index.toFixed(3)}</span>
        </div>
        <div class="behaviour-stat">
          <span class="behaviour-stat-label">Wave delay</span>
          <span class="behaviour-stat-value">${motor.head_to_tail_delay_ms_per_segment.toFixed(1)} ms/seg</span>
        </div>
        <div class="behaviour-stat">
          <span class="behaviour-stat-label">Wave fit R\u00B2</span>
          <span class="behaviour-stat-value">${motor.head_to_tail_fit_r2.toFixed(3)}</span>
        </div>
        <div class="behaviour-stat">
          <span class="behaviour-stat-label">Segment coherence</span>
          <span class="behaviour-stat-value">${wave.adjacent_segment_coherence.toFixed(3)}</span>
        </div>
        <div class="behaviour-stat">
          <span class="behaviour-stat-label">D/V phase lag</span>
          <span class="behaviour-stat-value">${wave.dorsal_to_ventral_phase_lag_ms.toFixed(1)} ms</span>
        </div>
      </div>
    `
  }

  function buildBehaviourCharts(result: BehaviorPerformanceResponse): void {
    destroyBehaviourCharts()
    behaviourChartHost.innerHTML = ''
    behaviourChartHost.append(firingChartDiv, caWaveChartDiv)
    firingChartDiv.innerHTML = ''
    caWaveChartDiv.innerHTML = ''

    const w = Math.max(200, el.clientWidth - 32)
    const motor = result.behavioral_readout.motor_neuron_firing_patterns

    // Per-neuron firing rate chart: B-type and D-type as two series
    const bRates = motor.b_type.per_neuron_rate_hz
    const dRates = motor.d_type.per_neuron_rate_hz
    const allNeurons = [...Object.keys(bRates), ...Object.keys(dRates)].sort()

    const firingChart = createSweepChart({
      title: 'Motor Neuron Firing Rates (Hz)',
      series: [
        { label: 'B-type', color: '#F59E0B' },
        { label: 'D-type', color: '#0EA5E9' },
      ],
      container: firingChartDiv,
      width: w,
      height: 130,
    })

    for (let i = 0; i < allNeurons.length; i++) {
      const name = allNeurons[i]
      firingChart.addPoint(i, bRates[name] ?? 0, dRates[name] ?? 0)
    }

    // Ca²⁺ wave proxy time-series chart
    const wave = result.behavioral_readout.muscle_ca2_wave_proxy
    const caWaveChart = createSweepChart({
      title: 'Ca\u00B2\u207A Wave Proxy (mean amplitude)',
      series: [
        { label: 'Dorsal', color: '#8B5CF6' },
        { label: 'Ventral', color: '#10B981' },
      ],
      container: caWaveChartDiv,
      width: w,
      height: 130,
    })

    if (wave.time_ms && wave.dorsal_ca_proxy && wave.ventral_ca_proxy) {
      const dorsalMean = meanAcrossSegments(wave.dorsal_ca_proxy, wave.time_ms.length)
      const ventralMean = meanAcrossSegments(wave.ventral_ca_proxy, wave.time_ms.length)
      for (let i = 0; i < wave.time_ms.length; i++) {
        caWaveChart.addPoint(wave.time_ms[i], dorsalMean[i], ventralMean[i])
      }
    }

    behaviourCharts = { firing: firingChart, caWave: caWaveChart }
  }

  function destroyBehaviourCharts(): void {
    if (behaviourCharts) {
      behaviourCharts.firing.destroy()
      behaviourCharts.caWave.destroy()
      behaviourCharts = null
    }
    firingChartDiv.innerHTML = ''
    caWaveChartDiv.innerHTML = ''
  }

  async function runBehaviourAssay(): Promise<void> {
    state.behaviourStatus = 'running'
    state.behaviourError = null
    behaviourStatusText = `Running forward locomotion assay (${behaviourGraphSource})...`
    renderBehaviourControls()

    try {
      behaviourResult = await fetchBehaviorPerformance({
        graph_source: behaviourGraphSource,
        include_traces: true,
      })
      state.behaviourStatus = 'done'
      behaviourStatusText = `Assay complete (${behaviourGraphSource})`
      renderBehaviourControls()
      renderBehaviourStats()
      buildBehaviourCharts(behaviourResult)
    } catch (err) {
      state.behaviourStatus = 'error'
      const msg = err instanceof Error ? err.message : 'Unknown error'
      state.behaviourError = msg
      behaviourStatusText = `Error: ${msg}`
      renderBehaviourControls()
    }
  }

  function resetBehaviour(): void {
    destroyBehaviourCharts()
    behaviourResult = null
    state.behaviourStatus = 'idle'
    state.behaviourError = null
    behaviourStatusText = 'Ready'
    behaviourStatsDiv.innerHTML = ''
    behaviourChartHost.innerHTML = ''
    renderBehaviourControls()
  }

  // Behaviour event delegation
  behaviourControlsDiv.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('.metrics-btn')
    if (!btn) return
    if (btn.dataset.action === 'run-behaviour') void runBehaviourAssay()
    if (btn.dataset.action === 'reset-behaviour') resetBehaviour()
  })

  behaviourControlsDiv.addEventListener('change', (e) => {
    const target = e.target as HTMLSelectElement
    if (target.dataset.field === 'graph-source') {
      behaviourGraphSource = target.value as GraphSource
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
    if (behaviourCharts) {
      behaviourCharts.firing.resize(w, 130)
      behaviourCharts.caWave.resize(w, 130)
    }
  })
  ro.observe(el)

  // ─── Initialize ───
  renderTabBar()
  renderActiveTab()

  // ─── Public: live simulation data feed ───
  return {
    onLiveUpdate(payload: LiveSimulationResponse) {
      if (activeTab !== 'live' || !idleChart) return
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

// ─── Helpers ───

function meanAcrossSegments(
  segmentData: Record<string, number[]>,
  length: number,
): number[] {
  const segments = Object.values(segmentData)
  if (segments.length === 0) return new Array(length).fill(0)
  const result = new Array<number>(length).fill(0)
  for (const seg of segments) {
    for (let i = 0; i < length && i < seg.length; i++) {
      result[i] += seg[i]
    }
  }
  const n = segments.length
  for (let i = 0; i < length; i++) {
    result[i] /= n
  }
  return result
}
