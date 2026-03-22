import './style.css'

import { fetchGraphData, fetchReplacementGraph, fetchSpikeTrains } from './api'
import { createRenderer } from './graph/renderer'
import { setupInteractions } from './graph/interactions'
import { createFiringEngine } from './simulation/firing-engine'
import { createAnimationLoop } from './simulation/animation-loop'
import { createTimescaleControl } from './ui/timescale-control'
import { createNeuronTooltip } from './ui/neuron-tooltip'
import { createLegend } from './ui/legend'
import { createReplacementControl } from './ui/replacement-control'
import { createMetricsPanel } from './ui/metrics-panel'
import type { SpikeData } from './types'

async function boot(): Promise<void> {
  const graphContainer = document.getElementById('graph-container')
  const uiOverlay = document.getElementById('ui-overlay')

  if (!graphContainer || !uiOverlay) {
    throw new Error('Missing #graph-container or #ui-overlay')
  }

  // Show loading state
  const loading = document.createElement('div')
  loading.className = 'loading'
  loading.innerHTML = `
    <div class="loading-spinner"></div>
    <span class="loading-text">Loading connectome...</span>
  `
  document.body.appendChild(loading)

  try {
    let data
    try {
      data = await fetchReplacementGraph()
    } catch {
      data = await fetchGraphData()
    }

    // Build graph renderer
    const ctx = createRenderer(graphContainer, data)

    // Wire up interactions (click, hover)
    setupInteractions(ctx)

    // Mount metrics panel first so we can feed it live data
    const metricsPanel = createMetricsPanel(uiOverlay)

    let spikeData: SpikeData | undefined
    try {
      loading.querySelector('.loading-text')!.textContent = 'Running simulation...'
      spikeData = await fetchSpikeTrains()
      console.log(
        `Loaded spike data: ${spikeData.active_count}/${spikeData.neuron_count} active neurons over ${spikeData.duration_ms}ms`,
      )
    } catch {
      console.warn('Simulation endpoint unavailable — using live / fallback firing')
    }

    const firingEngine = createFiringEngine(ctx, {
      spikeData,
      onLiveUpdate: (payload) => metricsPanel.onLiveUpdate(payload),
    })
    const animationLoop = createAnimationLoop(ctx)
    firingEngine.start()
    animationLoop.start()

    // Mount remaining UI overlays
    const brand = document.createElement('div')
    brand.className = 'brand'
    brand.innerHTML = `
      <span class="brand-name">Synaptech</span>
      <span class="brand-sub">C. elegans connectome</span>
    `
    uiOverlay.appendChild(brand)

    createLegend(uiOverlay)
    createTimescaleControl(uiOverlay)
    createReplacementControl(uiOverlay, ctx)
    createNeuronTooltip(uiOverlay, ctx)

    // Remove loading
    loading.remove()
  } catch (err) {
    loading.innerHTML = `
      <span class="loading-text">Failed to load connectome data.</span>
      <span class="loading-text" style="font-size: 0.75rem; opacity: 0.6">${err instanceof Error ? err.message : 'Unknown error'}</span>
    `
  }
}

void boot()
