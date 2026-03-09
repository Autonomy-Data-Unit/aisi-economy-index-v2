# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # observe.server
#
# FastAPI server that exposes the `NetObserver` API over HTTP.
# Starts non-blocking alongside a running netrun pipeline.

# %%
#|default_exp utils.observe.server

# %%
#|export
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from netrun.core import Net
from ai_index.utils.observe.core import NetObserver
from ai_index.utils.observe.models import (
    NetStatus, NodeStatus, EpochInfo, EdgeStatus, LogEntry,
    SendControlRequest, InjectDataRequest, ControlResponse,
)

# %%
#|export
class ObserveServer:
    """Non-blocking FastAPI server wrapping a NetObserver.

    Usage (inside an async context, e.g. alongside a running netrun)::

        server = ObserveServer(net)
        await server.start()
        # ... pipeline runs ...
        await server.stop()
    """

    def __init__(self, net: Net, host: str = "127.0.0.1", port: int = 8000, ws_interval: float = 1.0):
        self.observer = NetObserver(net)
        self.host = host
        self.port = port
        self.ws_interval = ws_interval
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None
        self._ws_clients: set[WebSocket] = set()
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="Netrun Observer")
        obs = self.observer

        @app.get("/status", response_model=NetStatus)
        def get_status() -> NetStatus:
            return obs.get_status()

        @app.get("/nodes", response_model=list[NodeStatus])
        def get_nodes() -> list[NodeStatus]:
            return obs.get_nodes()

        @app.get("/nodes/{name}", response_model=NodeStatus)
        def get_node(name: str) -> NodeStatus:
            return obs.get_node(name)

        @app.get("/epochs", response_model=list[EpochInfo])
        def get_epochs() -> list[EpochInfo]:
            return obs.get_epochs()

        @app.get("/edges", response_model=list[EdgeStatus])
        def get_edges() -> list[EdgeStatus]:
            return obs.get_edges()

        @app.get("/logs", response_model=list[LogEntry])
        def get_all_logs() -> list[LogEntry]:
            return obs.get_all_logs()

        @app.get("/logs/{node_name}", response_model=list[LogEntry])
        def get_node_logs(node_name: str) -> list[LogEntry]:
            return obs.get_node_logs(node_name)

        # -- Control endpoints --

        @app.post("/nodes/{name}/enable", response_model=ControlResponse)
        def enable_node(name: str) -> ControlResponse:
            return obs.enable_node(name)

        @app.post("/nodes/{name}/disable", response_model=ControlResponse)
        def disable_node(name: str) -> ControlResponse:
            return obs.disable_node(name)

        @app.post("/control", response_model=ControlResponse)
        def send_control(req: SendControlRequest) -> ControlResponse:
            return obs.send_control(req.node_name, req.control_type, req.value)

        @app.post("/inject", response_model=ControlResponse)
        def inject_data(req: InjectDataRequest) -> ControlResponse:
            return obs.inject_data(req.node_name, req.port_name, req.values)

        # -- WebSocket --

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self._ws_clients.add(ws)
            try:
                while True:
                    await ws.receive_text()  # keep connection alive
            except WebSocketDisconnect:
                pass
            finally:
                self._ws_clients.discard(ws)

        # -- Dashboard --

        @app.get("/dashboard", response_class=HTMLResponse)
        def dashboard() -> str:
            return _DASHBOARD_HTML

        return app

    async def _ws_broadcast_loop(self) -> None:
        """Periodically push full state to all WebSocket clients."""
        while True:
            if self._ws_clients:
                obs = self.observer
                payload = json.dumps({
                    "status": obs.get_status().model_dump(),
                    "nodes": [n.model_dump() for n in obs.get_nodes()],
                    "epochs": [e.model_dump() for e in obs.get_epochs()],
                    "logs": [l.model_dump() for l in obs.get_all_logs()],
                })
                dead: list[WebSocket] = []
                for ws in list(self._ws_clients):
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    self._ws_clients.discard(ws)
            await asyncio.sleep(self.ws_interval)

    async def _run_server(self) -> None:
        """Wrapper that catches SystemExit from uvicorn bind failures."""
        try:
            await self._server.serve()
        except SystemExit as e:
            self._startup_error = RuntimeError(
                f"ObserveServer failed to start on {self.host}:{self.port} "
                f"(port may be in use)"
            )

    async def start(self) -> None:
        """Start the server as a background asyncio task (non-blocking).

        Waits until uvicorn is actually listening before returning.
        Raises RuntimeError if the server fails to bind (e.g. port in use).
        """
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._startup_error: RuntimeError | None = None
        self._task = asyncio.create_task(self._run_server())

        # Wait for uvicorn to confirm it's listening, or fail early
        while not self._server.started:
            if self._startup_error:
                raise self._startup_error
            if self._task.done():
                if self._startup_error:
                    raise self._startup_error
                raise RuntimeError(f"ObserveServer exited before starting on {self.host}:{self.port}")
            await asyncio.sleep(0.05)

        self._ws_task = asyncio.create_task(self._ws_broadcast_loop())

    async def stop(self) -> None:
        """Signal the server to shut down and wait for it."""
        if hasattr(self, '_ws_task') and self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        if self._server:
            self._server.should_exit = True
        if self._task:
            await self._task

# %%
#|export
_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Netrun Observer</title>
<style>
  :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3;
          --muted: #8b949e; --green: #3fb950; --yellow: #d29922; --red: #f85149;
          --blue: #58a6ff; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--text); padding: 1.5rem; }
  h1 { font-size: 1.4rem; margin-bottom: 1rem; }
  h2 { font-size: 1.1rem; margin-bottom: .5rem; color: var(--blue); }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
  .card.full { grid-column: 1 / -1; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; }
  th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: .75rem; font-weight: 600; }
  .badge-Finished { background: #23392e; color: var(--green); }
  .badge-Running  { background: #2d2a1e; color: var(--yellow); }
  .badge-Startable { background: #1c2333; color: var(--blue); }
  .badge-busy { background: #2d2a1e; color: var(--yellow); }
  .badge-idle { background: #23392e; color: var(--green); }
  .badge-disabled { background: #2d1f1f; color: var(--red); }
  .logs-box { max-height: 400px; overflow-y: auto; font-family: 'SF Mono', 'Menlo', monospace;
              font-size: .8rem; line-height: 1.5; white-space: pre-wrap; padding: .5rem;
              background: var(--bg); border-radius: 4px; }
  .log-ts { color: var(--muted); }
  .log-node { color: var(--blue); }
  .log-tabs { display: flex; gap: 0; margin-bottom: .5rem; flex-wrap: wrap; }
  .log-tab { background: transparent; border: 1px solid var(--border); border-bottom: none;
             color: var(--muted); padding: 4px 12px; font-size: .75rem; cursor: pointer;
             border-radius: 6px 6px 0 0; font-family: inherit; }
  .log-tab:hover { color: var(--text); }
  .log-tab.active { background: var(--bg); color: var(--blue); border-color: var(--blue);
                    border-bottom: 1px solid var(--bg); font-weight: 600; }
  .status-bar { display: flex; gap: 1.5rem; margin-bottom: 1rem; color: var(--muted); font-size: .85rem; }
  .status-bar span { color: var(--text); font-weight: 600; }
  .ws-indicator { font-size: .75rem; margin-left: auto; }
  .ws-indicator.connected { color: var(--green); }
  .ws-indicator.disconnected { color: var(--red); }
  .btn { background: var(--card); border: 1px solid var(--border); color: var(--text);
         padding: 2px 8px; border-radius: 4px; font-size: .75rem; cursor: pointer; }
  .btn:hover { border-color: var(--blue); }
  .btn-enable { color: var(--green); }
  .btn-disable { color: var(--red); }
</style>
</head>
<body>
<h1>Netrun Observer</h1>
<div class="status-bar" id="status-bar"><span class="ws-indicator disconnected" id="ws-status">disconnected</span></div>
<div class="grid">
  <div class="card"><h2>Nodes</h2><div id="nodes-table"></div></div>
  <div class="card"><h2>Epochs</h2><div id="epochs-table"></div></div>
  <div class="card full"><h2>Logs</h2><div class="log-tabs" id="log-tabs"></div><div class="logs-box" id="logs-box"></div></div>
</div>
<script>
const BASE = window.location.origin;
const WS_URL = (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/ws';
const FALLBACK_POLL_MS = 3000;

let ws = null;
let fallbackTimer = null;
let logFilter = null;  // null = all, string = node name
let lastData = null;   // cache for re-rendering on filter change

function badge(text, cls) {
  return `<span class="badge badge-${cls || text}">${text}</span>`;
}

function fmtDuration(s) {
  if (s == null) return '-';
  if (s < 1) return (s * 1000).toFixed(0) + 'ms';
  if (s < 60) return s.toFixed(1) + 's';
  return (s / 60).toFixed(1) + 'm';
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

async function postAction(path, body) {
  try {
    const r = await fetch(BASE + path, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await r.json();
    if (!data.ok) console.error('Action failed:', data.message);
  } catch (e) { console.error('Action error:', e); }
}

function setLogFilter(name) {
  logFilter = name;
  if (lastData) renderLogs(lastData.logs, lastData.nodes);
}

function renderLogs(logs, nodes) {
  // Build tab bar
  const nodeNames = nodes.map(n => n.name);
  let tabs = `<button class="log-tab ${logFilter === null ? 'active' : ''}" onclick="setLogFilter(null)">All</button>`;
  for (const name of nodeNames) {
    const count = logs.filter(l => l.node_name === name).length;
    const active = logFilter === name ? ' active' : '';
    tabs += `<button class="log-tab${active}" onclick="setLogFilter('${name}')">${name} (${count})</button>`;
  }
  document.getElementById('log-tabs').innerHTML = tabs;

  // Filter and render log lines
  const filtered = logFilter === null ? logs : logs.filter(l => l.node_name === logFilter);
  const box = document.getElementById('logs-box');
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 20;
  let lh = '';
  for (const l of filtered) {
    const ts = l.timestamp.split(' ').pop()?.split('.')[0] || l.timestamp;
    const node = l.node_name || '';
    lh += `<span class="log-ts">${ts}</span> <span class="log-node">[${node}]</span> ${escapeHtml(l.message)}\\n`;
  }
  box.innerHTML = lh;
  if (atBottom) box.scrollTop = box.scrollHeight;
}

function render(data) {
  lastData = data;
  const {status, nodes, epochs, logs} = data;

  // Status bar
  const parts = Object.entries(status.epochs_by_state).map(
    ([k, v]) => `${k}: <span>${v}</span>`
  ).join(' &middot; ');
  const wsEl = document.getElementById('ws-status');
  const wsHtml = wsEl ? wsEl.outerHTML : '';
  document.getElementById('status-bar').innerHTML =
    `Nodes: <span>${status.node_names.length}</span> &middot; ` +
    `Edges: <span>${status.edge_count}</span> &middot; ` +
    `Epochs: <span>${status.total_epochs}</span> &middot; ${parts}` + wsHtml;

  // Nodes table with control buttons
  let nh = '<table><tr><th>Node</th><th>Status</th><th>Epochs</th><th>Running</th><th>Actions</th></tr>';
  for (const n of nodes) {
    const st = !n.enabled ? badge('disabled') : n.is_busy ? badge('busy') : badge('idle');
    const toggleBtn = n.enabled
      ? `<button class="btn btn-disable" onclick="postAction('/nodes/${n.name}/disable')">disable</button>`
      : `<button class="btn btn-enable" onclick="postAction('/nodes/${n.name}/enable')">enable</button>`;
    nh += `<tr><td><a href="#" onclick="setLogFilter('${n.name}');return false" style="color:var(--text);text-decoration:none">${n.name}</a></td><td>${st}</td><td>${n.epoch_count}</td>` +
          `<td>${n.running_epoch_ids.length}</td><td>${toggleBtn}</td></tr>`;
  }
  nh += '</table>';
  document.getElementById('nodes-table').innerHTML = nh;

  // Epochs table (most recent first)
  const sorted = [...epochs].reverse();
  let eh = '<table><tr><th>Epoch</th><th>Node</th><th>State</th><th>Duration</th></tr>';
  for (const e of sorted.slice(0, 50)) {
    eh += `<tr><td style="font-family:monospace;font-size:.75rem">${e.epoch_id.slice(0,8)}</td>` +
          `<td>${e.node_name}</td><td>${badge(e.state)}</td>` +
          `<td>${fmtDuration(e.duration_seconds)}</td></tr>`;
  }
  eh += '</table>';
  document.getElementById('epochs-table').innerHTML = eh;

  // Logs
  renderLogs(logs, nodes);
}

// -- WebSocket with HTTP polling fallback --

function setWsStatus(connected) {
  const el = document.getElementById('ws-status');
  if (!el) return;
  el.textContent = connected ? 'live' : 'polling';
  el.className = 'ws-indicator ' + (connected ? 'connected' : 'disconnected');
}

function connectWs() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    setWsStatus(true);
    if (fallbackTimer) { clearInterval(fallbackTimer); fallbackTimer = null; }
  };
  ws.onmessage = (e) => {
    try { render(JSON.parse(e.data)); } catch (err) { console.error('WS parse error:', err); }
  };
  ws.onclose = () => {
    setWsStatus(false);
    startFallbackPolling();
    setTimeout(connectWs, 2000);
  };
  ws.onerror = () => { ws.close(); };
}

async function fallbackRefresh() {
  try {
    const [status, nodes, epochs, logs] = await Promise.all([
      fetch(BASE + '/status').then(r => r.json()),
      fetch(BASE + '/nodes').then(r => r.json()),
      fetch(BASE + '/epochs').then(r => r.json()),
      fetch(BASE + '/logs').then(r => r.json()),
    ]);
    render({status, nodes, epochs, logs});
  } catch (e) { console.error('Poll failed:', e); }
}

function startFallbackPolling() {
  if (!fallbackTimer) {
    fallbackTimer = setInterval(fallbackRefresh, FALLBACK_POLL_MS);
  }
}

// Initial load via HTTP, then switch to WebSocket
fallbackRefresh();
connectWs();
</script>
</body>
</html>
"""
