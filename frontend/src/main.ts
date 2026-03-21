import './style.css'

import { fetchGraphData } from './api'
import { createRenderer } from './graph/renderer'
import { setupInteractions } from './graph/interactions'
import { createFiringEngine } from './simulation/firing-engine'
import { createAnimationLoop } from './simulation/animation-loop'
import { createTimescaleControl } from './ui/timescale-control'
import { createNeuronTooltip } from './ui/neuron-tooltip'
import { createLegend } from './ui/legend'
import type { LiveSimulationResponse } from './types'

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
    const data = await fetchGraphData()

    // Build graph renderer
    const ctx = createRenderer(graphContainer, data)

    // Wire up interactions (click, hover)
    setupInteractions(ctx)

    const liveStatus = document.createElement('div')
    liveStatus.className = 'live-status'
    liveStatus.innerHTML = `
      <div class="live-title">Live Simulation</div>
      <div class="live-line">Status: Connecting...</div>
      <div class="live-line">Awaiting backend stream</div>
    `
    uiOverlay.appendChild(liveStatus)

    // Start backend-driven firing engine + animation loop
    const firingEngine = createFiringEngine(ctx, {
      onLiveUpdate: (payload) => renderLiveStatus(liveStatus, payload),
      onLiveError: (message) => renderLiveError(liveStatus, message),
    })
    const animationLoop = createAnimationLoop(ctx)
    firingEngine.start()
    animationLoop.start()

    // Mount UI overlays
    const brand = document.createElement('div')
    brand.className = 'brand'
    brand.innerHTML = `
      <span class="brand-name">Synaptech</span>
      <span class="brand-sub">C. elegans connectome</span>
    `
    uiOverlay.appendChild(brand)

    createLegend(uiOverlay)
    createTimescaleControl(uiOverlay)
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

function renderLiveStatus(el: HTMLElement, payload: LiveSimulationResponse): void {
  const summary = payload.firing_summary_hz
  const top = payload.top_firing_neurons[0]

  el.classList.remove('live-status-error')
  el.innerHTML = `
    <div class="live-title">Live Simulation</div>
    <div class="live-line">Status: Live backend data</div>
    <div class="live-line">Overall: ${summary.overall_mean_hz.toFixed(2)} Hz</div>
    <div class="live-line">Sensory/Inter/Motor: ${summary.sensory_mean_hz.toFixed(2)} / ${summary.interneuron_mean_hz.toFixed(2)} / ${summary.motor_mean_hz.toFixed(2)} Hz</div>
    <div class="live-line">Active neurons: ${(summary.active_fraction * 100).toFixed(1)}%</div>
    <div class="live-line">Population spike rate: ${payload.population_spike_rate_hz.toFixed(2)} Hz</div>
    <div class="live-line">Top firing neuron: ${top ? `${top.name} (${top.firing_rate_hz.toFixed(2)} Hz)` : 'n/a'}</div>
  `
}

function renderLiveError(el: HTMLElement, message: string): void {
  if (!el.textContent?.includes('Live backend data')) {
    el.classList.add('live-status-error')
    el.innerHTML = `
      <div class="live-title">Live Simulation</div>
      <div class="live-line">Status: Fallback mode</div>
      <div class="live-line">${message}</div>
    `
  }
}

void boot()
