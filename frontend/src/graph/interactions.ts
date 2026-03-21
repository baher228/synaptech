import type { RendererContext } from './renderer'
import { state, emit } from '../state'

export function setupInteractions(ctx: RendererContext): void {
  const { sigma } = ctx

  sigma.on('clickNode', ({ node }) => {
    if (state.selectedNeuron === node) {
      state.selectedNeuron = null
    } else {
      state.selectedNeuron = node
    }
    emit('selection')
    sigma.refresh()
  })

  sigma.on('clickStage', () => {
    if (state.selectedNeuron !== null) {
      state.selectedNeuron = null
      emit('selection')
      sigma.refresh()
    }
  })

  sigma.on('enterNode', ({ node }) => {
    state.hoveredNeuron = node
    document.body.style.cursor = 'pointer'
    sigma.refresh()
  })

  sigma.on('leaveNode', () => {
    state.hoveredNeuron = null
    document.body.style.cursor = 'default'
    sigma.refresh()
  })
}
