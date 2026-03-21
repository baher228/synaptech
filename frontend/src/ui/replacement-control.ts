import {
  fetchRandomFaultyNeurons,
  fetchReplacementGraph,
  resetReplacement,
  startReplacement,
  stepReplacement,
} from '../api'
import { replaceRendererData, type RendererContext } from '../graph/renderer'
import { emit, state } from '../state'
import type { ReplacementSession } from '../types'

const AUTO_STEP_SIZE = 8
const AUTO_STEP_DELAY_MS = 140

export function createReplacementControl(
  container: HTMLElement,
  ctx: RendererContext,
): void {
  const el = document.createElement('div')
  el.className = 'replacement-control'
  container.appendChild(el)

  let candidateFaulty: string | null = null
  let session: ReplacementSession | null = null
  let busy = false
  let autoRunning = false
  let statusText = 'Ready'

  function syncReplacementFocus(nextSession: ReplacementSession | null): void {
    if (!nextSession) {
      state.activeFaultyNeuron = null
      state.activeReplacementNeuron = null
      state.replacementStatus = null
      ctx.sigma.refresh()
      return
    }
    state.activeFaultyNeuron = nextSession.faulty_neuron
    state.activeReplacementNeuron = nextSession.replacement_neuron
    state.replacementStatus = nextSession.status
    ctx.sigma.refresh()
  }

  function selectedNeuron(): string | null {
    if (!state.selectedNeuron) return null
    return ctx.graph.hasNode(state.selectedNeuron) ? state.selectedNeuron : null
  }

  async function refreshGraph(): Promise<void> {
    const graphData = await fetchReplacementGraph()
    replaceRendererData(ctx, graphData)

    if (state.selectedNeuron && !ctx.graph.hasNode(state.selectedNeuron)) {
      state.selectedNeuron = null
      emit('selection')
    }
    if (state.hoveredNeuron && !ctx.graph.hasNode(state.hoveredNeuron)) {
      state.hoveredNeuron = null
    }
  }

  function render(): void {
    const selected = selectedNeuron()
    const activeFaulty = selected || candidateFaulty || session?.faulty_neuron || 'n/a'
    const activeReplacement = session?.replacement_neuron || 'n/a'
    const canStep = Boolean(session && session.status !== 'completed')
    const nextEdge = session?.next_edge
      ? `${session.next_edge.old_source} -> ${session.next_edge.old_target} => ${session.next_edge.new_source} -> ${session.next_edge.new_target}`
      : 'n/a'

    el.innerHTML = `
      <div class="replacement-title">Replacement Control</div>
      <div class="replacement-line">Status: ${statusText}</div>
      <div class="replacement-line">Selected neuron: ${selected ?? 'none'}</div>
      <div class="replacement-line">Faulty neuron: ${activeFaulty}</div>
      <div class="replacement-line">Replacement neuron: ${activeReplacement}</div>
      <div class="replacement-line">Session: ${session?.session_id ?? 'none'}</div>
      <div class="replacement-line">Progress: ${session ? `${session.completed_count} migrated / ${session.pending_count} pending` : 'not started'}</div>
      <div class="replacement-line replacement-next">Next edge: ${nextEdge}</div>
      <div class="replacement-actions">
        <button class="replacement-btn" data-action="pick" ${busy || autoRunning ? 'disabled' : ''}>Pick Random Faulty</button>
        <button class="replacement-btn" data-action="start" ${busy || autoRunning ? 'disabled' : ''}>Start Replacement</button>
        <button class="replacement-btn" data-action="step" ${busy || autoRunning || !canStep ? 'disabled' : ''}>Migrate One Edge</button>
        <button class="replacement-btn" data-action="auto" ${busy || autoRunning || !canStep ? 'disabled' : ''}>Auto Complete</button>
        <button class="replacement-btn replacement-btn-reset" data-action="reset" ${busy || autoRunning ? 'disabled' : ''}>Reset Graph</button>
      </div>
    `
  }

  async function pickRandomFaulty(): Promise<void> {
    busy = true
    statusText = 'Selecting random faulty neuron...'
    render()
    try {
      const neurons = await fetchRandomFaultyNeurons(1)
      candidateFaulty = neurons[0] ?? null
      statusText = candidateFaulty
        ? `Candidate selected: ${candidateFaulty}`
        : 'No candidate neuron available'
    } catch (error) {
      statusText = `Faulty selection failed: ${formatError(error)}`
    } finally {
      busy = false
      render()
    }
  }

  async function startSession(): Promise<void> {
    busy = true
    statusText = 'Starting replacement session...'
    render()
    try {
      const faulty = selectedNeuron() ?? candidateFaulty ?? undefined
      session = await startReplacement({
        faultyNeuron: faulty,
        edgeOrder: 'random',
      })
      syncReplacementFocus(session)
      candidateFaulty = session.faulty_neuron
      statusText = `Session ${session.session_id} started`
      await refreshGraph()
      emit('selection')
    } catch (error) {
      statusText = `Start failed: ${formatError(error)}`
    } finally {
      busy = false
      render()
    }
  }

  async function migrateStep(edgesToMigrate: number): Promise<void> {
    if (!session) return
    busy = true
    statusText = `Migrating ${edgesToMigrate} edge(s)...`
    render()
    try {
      session = await stepReplacement(session.session_id, edgesToMigrate)
      syncReplacementFocus(session)
      statusText =
        session.status === 'completed'
          ? `Replacement complete for ${session.faulty_neuron}`
          : `Session ${session.session_id}: ${session.pending_count} edges pending`
      await refreshGraph()
    } catch (error) {
      statusText = `Step failed: ${formatError(error)}`
    } finally {
      busy = false
      render()
    }
  }

  async function autoComplete(): Promise<void> {
    if (!session || session.status === 'completed') return
    autoRunning = true
    statusText = `Auto-migrating edges for ${session.faulty_neuron}...`
    render()
    try {
      while (session && session.status !== 'completed') {
        session = await stepReplacement(session.session_id, AUTO_STEP_SIZE)
        syncReplacementFocus(session)
        await refreshGraph()
        render()
        if (session.status !== 'completed') {
          await sleep(AUTO_STEP_DELAY_MS)
        }
      }
      if (session) {
        statusText = `Replacement complete for ${session.faulty_neuron}`
      }
    } catch (error) {
      statusText = `Auto-complete failed: ${formatError(error)}`
    } finally {
      autoRunning = false
      render()
    }
  }

  async function resetGraph(): Promise<void> {
    busy = true
    statusText = 'Resetting replacement graph...'
    render()
    try {
      await resetReplacement()
      await refreshGraph()
      candidateFaulty = null
      session = null
      syncReplacementFocus(null)
      statusText = 'Replacement graph reset'
      emit('selection')
    } catch (error) {
      statusText = `Reset failed: ${formatError(error)}`
    } finally {
      busy = false
      render()
    }
  }

  el.addEventListener('click', (event) => {
    const target = (event.target as HTMLElement).closest<HTMLButtonElement>('.replacement-btn')
    if (!target) return

    const action = target.dataset.action
    if (action === 'pick') {
      void pickRandomFaulty()
      return
    }
    if (action === 'start') {
      void startSession()
      return
    }
    if (action === 'step') {
      void migrateStep(1)
      return
    }
    if (action === 'auto') {
      void autoComplete()
      return
    }
    if (action === 'reset') {
      void resetGraph()
    }
  })

  render()
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : 'Unknown error'
}
