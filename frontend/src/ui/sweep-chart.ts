import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

export interface SweepChartOptions {
  title: string
  /** One series per entry: { label, color } */
  series: { label: string; color: string }[]
  baselineValue?: number
  container: HTMLElement
  width: number
  height: number
}

export interface SweepChart {
  /** Push one x value and one y per series */
  addPoint(x: number, ...values: number[]): void
  resize(width: number, height: number): void
  reset(): void
  destroy(): void
}

export function createSweepChart(opts: SweepChartOptions): SweepChart {
  const seriesCount = opts.series.length
  // data[0] = xs, data[1..n] = ys per series
  const data: number[][] = Array.from({ length: seriesCount + 1 }, () => [])

  const baselinePlugin: uPlot.Plugin | undefined =
    opts.baselineValue !== undefined
      ? {
          hooks: {
            draw: [
              (u: uPlot) => {
                const ctx = u.ctx
                const yPos = u.valToPos(opts.baselineValue!, 'y', true)
                if (!isFinite(yPos)) return
                ctx.save()
                ctx.strokeStyle = '#9CA3AF'
                ctx.setLineDash([4, 4])
                ctx.lineWidth = 1
                ctx.beginPath()
                ctx.moveTo(u.bbox.left / devicePixelRatio, yPos)
                ctx.lineTo(
                  (u.bbox.left + u.bbox.width) / devicePixelRatio,
                  yPos,
                )
                ctx.stroke()
                ctx.restore()
              },
            ],
          },
        }
      : undefined

  const plugins: uPlot.Plugin[] = []
  if (baselinePlugin) plugins.push(baselinePlugin)

  const uplotSeries: uPlot.Series[] = [
    {}, // x-axis
    ...opts.series.map((s) => ({
      label: s.label,
      stroke: s.color,
      width: 2,
      fill: s.color + '18',
    })),
  ]

  const uplotOpts: uPlot.Options = {
    width: opts.width,
    height: opts.height,
    title: opts.title,
    plugins,
    cursor: { show: true },
    select: { show: false, left: 0, top: 0, width: 0, height: 0 },
    legend: { show: seriesCount > 1 },
    series: uplotSeries,
    axes: [
      {
        stroke: '#9CA3AF',
        grid: { stroke: '#E5E7EB80', width: 1 },
        ticks: { stroke: '#E5E7EB', width: 1 },
        font: '10px DM Mono',
      },
      {
        stroke: '#9CA3AF',
        grid: { stroke: '#E5E7EB80', width: 1 },
        ticks: { stroke: '#E5E7EB', width: 1 },
        font: '10px DM Mono',
        size: 50,
      },
    ],
    scales: {
      x: { time: false },
    },
  }

  const plot = new uPlot(uplotOpts, data as uPlot.AlignedData, opts.container)

  return {
    addPoint(x: number, ...values: number[]) {
      data[0].push(x)
      for (let i = 0; i < seriesCount; i++) {
        data[i + 1].push(values[i] ?? 0)
      }
      plot.setData(data as uPlot.AlignedData)
    },
    resize(width: number, height: number) {
      plot.setSize({ width, height })
    },
    reset() {
      for (const arr of data) arr.length = 0
      plot.setData(data as uPlot.AlignedData)
    },
    destroy() {
      plot.destroy()
    },
  }
}
