let selectedToolId = null;
let selectedToolPayload = null;

function setStatus(message) {
  document.getElementById('statusText').textContent = message;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

async function loadToolCenter() {
  setStatus('Lade Tools ...');
  const filter = document.getElementById('statusFilter').value;
  const url = filter ? `/api/gui/tools?status=${encodeURIComponent(filter)}` : '/api/gui/tools';
  const [dashboardRes, listRes] = await Promise.all([
    fetch('/api/gui/tools/dashboard'),
    fetch(url),
  ]);
  const dashboard = await dashboardRes.json();
  const list = await listRes.json();
  renderSummary(dashboard);
  renderToolList(list.tools || []);
  setStatus(`${list.count || 0} Tool(s) geladen`);
}

function renderSummary(data) {
  const counts = data.status_counts || {};
  document.getElementById('toolCount').textContent = data.tool_count ?? 0;
  document.getElementById('activeCount').textContent = counts.ACTIVE ?? 0;
  document.getElementById('disabledCount').textContent = counts.DISABLED ?? 0;
  document.getElementById('deprecatedCount').textContent = counts.DEPRECATED ?? 0;
}

function renderToolList(tools) {
  const list = document.getElementById('toolList');
  if (!tools.length) {
    list.innerHTML = '<div class="tool-item"><h3>Keine Tools</h3><p>Für den Filter wurden keine Tools gefunden.</p></div>';
    return;
  }
  list.innerHTML = tools.map(tool => `
    <button class="tool-item ${tool.id === selectedToolId ? 'selected' : ''}" onclick="showTool('${escapeHtml(tool.id)}')">
      <h3>${escapeHtml(tool.name)} <span class="badge">${escapeHtml(tool.status)}</span></h3>
      <p>${escapeHtml(tool.description)}</p>
      <span class="badge">${escapeHtml(tool.security_level)}</span>
      <span class="badge">${tool.stats?.executions ?? 0} Runs</span>
    </button>
  `).join('');
}

async function showTool(toolId) {
  selectedToolId = toolId;
  setStatus(`Lade ${toolId} ...`);
  const res = await fetch(`/api/gui/tools/${encodeURIComponent(toolId)}`);
  if (!res.ok) {
    setStatus('Tool nicht gefunden');
    return;
  }
  selectedToolPayload = await res.json();
  renderToolDetail(selectedToolPayload);
  await loadToolCenter();
}

function renderToolDetail(payload) {
  const tool = payload.tool || {};
  const stats = payload.stats || {};
  const detail = document.getElementById('toolDetail');
  detail.innerHTML = `
    <h2>${escapeHtml(tool.name || tool.id)}</h2>
    <p>${escapeHtml(tool.description || '')}</p>
    <div class="badge-row">
      <span class="badge primary">${escapeHtml(tool.status)}</span>
      <span class="badge">${escapeHtml(tool.security_level)}</span>
      <span class="badge">v${escapeHtml(tool.version)}</span>
    </div>
    <div class="meta-grid">
      <div class="meta-box"><span>ID</span>${escapeHtml(tool.id)}</div>
      <div class="meta-box"><span>Modul</span>${escapeHtml(tool.module)}.${escapeHtml(tool.function)}</div>
      <div class="meta-box"><span>Ausführungen</span>${stats.executions ?? 0}</div>
      <div class="meta-box"><span>Fehler</span>${stats.failures ?? 0}</div>
      <div class="meta-box"><span>Letzte Nutzung</span>${escapeHtml(stats.last_used || 'nie')}</div>
      <div class="meta-box"><span>Letzter Fehler</span>${escapeHtml(stats.last_error || '—')}</div>
    </div>
  `;
  document.getElementById('rawTool').textContent = JSON.stringify(payload, null, 2);
  setStatus(`${tool.id} geladen`);
}

async function toolAction(action) {
  if (!selectedToolId) {
    setStatus('Bitte zuerst ein Tool auswählen.');
    return;
  }
  setStatus(`${action} wird ausgeführt ...`);
  const res = await fetch(`/api/gui/tools/${encodeURIComponent(selectedToolId)}/action`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  });
  const payload = await res.json();
  if (!res.ok || payload.success === false) {
    setStatus(`Aktion fehlgeschlagen: ${payload.detail || payload.error || 'Unbekannter Fehler'}`);
    document.getElementById('rawTool').textContent = JSON.stringify(payload, null, 2);
    return;
  }
  await showTool(selectedToolId);
  setStatus(payload.message || 'Aktion abgeschlossen');
}

window.addEventListener('DOMContentLoaded', loadToolCenter);
