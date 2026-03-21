import type { RendererContext } from '../graph/renderer'

const DECAY_RATE = 0.92

export interface AnimationLoop {
  start(): void
  stop(): void
}

export function createAnimationLoop(ctx: RendererContext): AnimationLoop {
  const { sigma, glowIntensities } = ctx
  let rafId: number | null = null
  let lastTime = 0

  function frame(now: number): void {
    const dt = lastTime ? now - lastTime : 16.67
    lastTime = now

    // Exponential decay of glow intensities
    const decayFactor = Math.pow(DECAY_RATE, dt / 16.67)
    let needsRefresh = false

    glowIntensities.forEach((intensity, nodeId) => {
      if (intensity > 0.005) {
        const newIntensity = intensity * decayFactor
        glowIntensities.set(nodeId, newIntensity < 0.005 ? 0 : newIntensity)
        needsRefresh = true
      } else if (intensity !== 0) {
        glowIntensities.set(nodeId, 0)
        needsRefresh = true
      }
    })

    if (needsRefresh) {
      sigma.refresh()
    }

    rafId = requestAnimationFrame(frame)
  }

  function start(): void {
    if (rafId !== null) return
    lastTime = 0
    rafId = requestAnimationFrame(frame)
  }

  function stop(): void {
    if (rafId !== null) {
      cancelAnimationFrame(rafId)
      rafId = null
    }
  }

  return { start, stop }
}
