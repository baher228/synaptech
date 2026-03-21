import './style.css'

import { fetchGraphData } from './api'
import { createRenderer } from './graph/renderer'
import { setupInteractions } from './graph/interactions'
import { createFiringEngine } from './simulation/firing-engine'
import { createAnimationLoop } from './simulation/animation-loop'
import { createTimescaleControl } from './ui/timescale-control'
import { createNeuronTooltip } from './ui/neuron-tooltip'
import { createLegend } from './ui/legend'

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

    // Start mock firing engine + animation loop
    const firingEngine = createFiringEngine(ctx)
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

void boot()
