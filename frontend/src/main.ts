import './style.css'

type ApiResponse = {
  message?: string
  status?: string
}

const app = document.querySelector<HTMLDivElement>('#app')

if (!app) {
  throw new Error('App container not found')
}

app.innerHTML = `
  <main class="shell">
    <header class="hero">
      <p class="eyebrow">Synaptech Starter</p>
      <h1>FastAPI + Vite</h1>
      <p class="subtext">A lightweight full-stack setup ready for features.</p>
      <button id="checkApi" type="button">Check backend</button>
    </header>

    <section class="panel">
      <h2>API response</h2>
      <pre id="output">Press "Check backend" to call /api endpoints.</pre>
    </section>
  </main>
`

const output = document.querySelector<HTMLPreElement>('#output')
const checkApiBtn = document.querySelector<HTMLButtonElement>('#checkApi')

async function callApi(): Promise<void> {
  if (!output) return

  output.textContent = 'Loading...'

  try {
    const [healthRes, messageRes] = await Promise.all([
      fetch('/api/health'),
      fetch('/api/message'),
    ])

    if (!healthRes.ok || !messageRes.ok) {
      throw new Error('Backend returned a non-200 response')
    }

    const health = (await healthRes.json()) as ApiResponse
    const message = (await messageRes.json()) as ApiResponse

    output.textContent = JSON.stringify(
      {
        health,
        message,
      },
      null,
      2,
    )
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error'
    output.textContent = `Failed to reach backend: ${message}`
  }
}

checkApiBtn?.addEventListener('click', () => {
  void callApi()
})
