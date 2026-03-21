import './style.css'

type ConnectomeSummary = {
  node_count: number
  edge_count: number
  chemical_edge_count: number
  gap_edge_count: number
  type_counts: Record<string, number>
  region_counts: Record<string, number>
  top_degree_centrality: Array<{
    name: string
    degree_centrality: number
  }>
}

const app = document.querySelector<HTMLDivElement>('#app')

if (!app) {
  throw new Error('App container not found')
}

const DEMO_SUMMARY: ConnectomeSummary = {
  node_count: 302,
  edge_count: 2804,
  chemical_edge_count: 2234,
  gap_edge_count: 570,
  type_counts: { S: 83, I: 81, M: 138 },
  region_counts: { head: 206, body: 58, tail: 38 },
  top_degree_centrality: [
    { name: 'AVAL', degree_centrality: 0.241 },
    { name: 'AVAR', degree_centrality: 0.237 },
    { name: 'AVBL', degree_centrality: 0.226 },
    { name: 'AVBR', degree_centrality: 0.222 },
    { name: 'PVCL', degree_centrality: 0.205 },
    { name: 'PVCR', degree_centrality: 0.204 },
    { name: 'RIML', degree_centrality: 0.196 },
    { name: 'RIMR', degree_centrality: 0.194 },
  ],
}

app.innerHTML = `
  <main class="shell">
    <header class="hero card">
      <div>
        <p class="eyebrow">Synaptech Visualizer</p>
        <h1>Connectome Snapshot</h1>
        <p class="subtext">Live metrics for the C. elegans network graph.</p>
      </div>
      <div class="hero-actions">
        <button id="refresh" type="button">Refresh data</button>
        <p id="dataMode" class="pill">Loading...</p>
      </div>
    </header>

    <section class="metric-grid">
      <article class="card metric">
        <p class="metric-label">Nodes</p>
        <p id="nodeCount" class="metric-value">--</p>
      </article>
      <article class="card metric">
        <p class="metric-label">Total Edges</p>
        <p id="edgeCount" class="metric-value">--</p>
      </article>
      <article class="card metric">
        <p class="metric-label">Chemical Edges</p>
        <p id="chemicalCount" class="metric-value">--</p>
      </article>
      <article class="card metric">
        <p class="metric-label">Gap Junction Edges</p>
        <p id="gapCount" class="metric-value">--</p>
      </article>
    </section>

    <section class="viz-grid">
      <article class="card panel">
        <h2>Neuron Types</h2>
        <div id="typeChart" class="bars"></div>
      </article>

      <article class="card panel">
        <h2>Region Distribution</h2>
        <div id="regionChart" class="bars"></div>
      </article>

      <article class="card panel">
        <h2>Synapse Mix</h2>
        <div id="synapseRing" class="ring">
          <span id="synapseLabel">--</span>
        </div>
      </article>

      <article class="card panel">
        <h2>Top Hub Neurons</h2>
        <ol id="hubList" class="hub-list"></ol>
      </article>
    </section>

    <section class="card panel raw-panel">
      <h2>Raw Summary</h2>
      <pre id="rawOutput">Loading...</pre>
    </section>
  </main>
`

const nodeCount = document.querySelector<HTMLElement>('#nodeCount')
const edgeCount = document.querySelector<HTMLElement>('#edgeCount')
const chemicalCount = document.querySelector<HTMLElement>('#chemicalCount')
const gapCount = document.querySelector<HTMLElement>('#gapCount')
const typeChart = document.querySelector<HTMLDivElement>('#typeChart')
const regionChart = document.querySelector<HTMLDivElement>('#regionChart')
const synapseRing = document.querySelector<HTMLDivElement>('#synapseRing')
const synapseLabel = document.querySelector<HTMLSpanElement>('#synapseLabel')
const hubList = document.querySelector<HTMLOListElement>('#hubList')
const rawOutput = document.querySelector<HTMLPreElement>('#rawOutput')
const dataMode = document.querySelector<HTMLParagraphElement>('#dataMode')
const refreshBtn = document.querySelector<HTMLButtonElement>('#refresh')

function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value)
}

function buildBars(
  container: HTMLDivElement | null,
  counts: Record<string, number>,
): void {
  if (!container) return
  container.replaceChildren()

  const total = Object.values(counts).reduce((sum, value) => sum + value, 0)
  const sortedEntries = Object.entries(counts).sort((a, b) => b[1] - a[1])

  for (const [label, value] of sortedEntries) {
    const row = document.createElement('div')
    row.className = 'bar-row'

    const info = document.createElement('div')
    info.className = 'bar-info'

    const name = document.createElement('span')
    name.textContent = label
    const stat = document.createElement('span')
    stat.textContent = `${formatNumber(value)} (${Math.round((value / total) * 100)}%)`
    info.append(name, stat)

    const track = document.createElement('div')
    track.className = 'bar-track'
    const fill = document.createElement('span')
    fill.className = 'bar-fill'
    fill.style.width = `${(value / total) * 100}%`
    track.append(fill)

    row.append(info, track)
    container.append(row)
  }
}

async function fetchConnectomeSummary(): Promise<{
  summary: ConnectomeSummary
  source: 'live' | 'demo'
}> {
  try {
    const response = await fetch('/api/connectome/summary')
    if (!response.ok) {
      throw new Error(`Status ${response.status}`)
    }

    const payload = (await response.json()) as ConnectomeSummary
    return { summary: payload, source: 'live' }
  } catch {
    return { summary: DEMO_SUMMARY, source: 'demo' }
  }
}

function renderSummary(summary: ConnectomeSummary, source: 'live' | 'demo'): void {
  if (nodeCount) nodeCount.textContent = formatNumber(summary.node_count)
  if (edgeCount) edgeCount.textContent = formatNumber(summary.edge_count)
  if (chemicalCount) chemicalCount.textContent = formatNumber(summary.chemical_edge_count)
  if (gapCount) gapCount.textContent = formatNumber(summary.gap_edge_count)

  buildBars(typeChart, summary.type_counts)
  buildBars(regionChart, summary.region_counts)

  const totalSynapseEdges = summary.chemical_edge_count + summary.gap_edge_count
  const chemicalShare = totalSynapseEdges > 0 ? summary.chemical_edge_count / totalSynapseEdges : 0

  if (synapseRing) {
    synapseRing.style.setProperty('--chemical-share', `${chemicalShare * 360}deg`)
  }
  if (synapseLabel) {
    synapseLabel.textContent = `${Math.round(chemicalShare * 100)}% chemical`
  }

  if (hubList) {
    hubList.replaceChildren()
    for (const hub of summary.top_degree_centrality.slice(0, 8)) {
      const item = document.createElement('li')
      const name = document.createElement('strong')
      name.textContent = hub.name
      const value = document.createElement('span')
      value.textContent = hub.degree_centrality.toFixed(3)
      item.append(name, value)
      hubList.append(item)
    }
  }

  if (rawOutput) {
    rawOutput.textContent = JSON.stringify(summary, null, 2)
  }

  if (dataMode) {
    dataMode.className = `pill ${source}`
    dataMode.textContent = source === 'live' ? 'Live API data' : 'Demo fallback data'
  }
}

async function refreshDashboard(): Promise<void> {
  if (dataMode) {
    dataMode.className = 'pill'
    dataMode.textContent = 'Loading...'
  }

  const { summary, source } = await fetchConnectomeSummary()
  renderSummary(summary, source)
}

refreshBtn?.addEventListener('click', () => {
  void refreshDashboard()
})

void refreshDashboard()
