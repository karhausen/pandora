let selectedCapabilityId = null;

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function setStatus(message) {
  document.getElementById('statusText').textContent = message;
}

async function loadDashboard() {
  setStatus('Lade Capability Graph ...');
  const graphRes = await fetch('/api/capabilities/graph');
  const graph = await graphRes.json();
  renderSummary(graph);
  await loadCapabilities();
  setStatus('Capability Graph geladen');
}

function renderSummary(graph) {
  const summary = graph.summary || {};
  const byType = summary.by_type || {};
  document.getElementById('capabilityCount').textContent = byType.capability ?? 0;
  document.getElementById('nodeCount').textContent = summary.node_count ?? (graph.nodes || []).length ?? 0;
  document.getElementById('edgeCount').textContent = summary.edge_count ?? (graph.edges || []).length ?? 0;
  document.getElementById('updatedAt').textContent = graph.updated_at ? new Date(graph.updated_at).toLocaleString() : '–';
}

async function loadCapabilities() {
  const query = document.getElementById('queryInput').value.trim();
  const url = query ? `/api/capabilities?query=${encodeURIComponent(query)}&limit=200` : '/api/capabilities?limit=200';
  const res = await fetch(url);
  const payload = await res.json();
  renderCapabilityList(payload.capabilities || []);
}

function renderCapabilityList(capabilities) {
  const list = document.getElementById('capabilityList');
  if (!capabilities.length) {
    list.innerHTML = '<div class="empty">Keine Capabilities gefunden. Baue den Graph neu auf oder ergänze Knowledge-Tags.</div>';
    return;
  }
  list.innerHTML = capabilities.map(cap => {
    const degree = cap.metadata?.degree ?? 0;
    return `
      <button class="capability-item ${cap.id === selectedCapabilityId ? 'selected' : ''}" onclick="showCapability('${escapeHtml(cap.id)}')">
        <h3>${escapeHtml(cap.label)} <span class="badge">${degree}</span></h3>
        <p>${escapeHtml(cap.source || 'unknown')} · ${escapeHtml(cap.id)}</p>
      </button>
    `;
  }).join('');
}

async function showCapability(capabilityId) {
  selectedCapabilityId = capabilityId;
  setStatus(`Lade ${capabilityId} ...`);
  const res = await fetch(`/api/capabilities/${encodeURIComponent(capabilityId)}`);
  if (!res.ok) {
    setStatus('Capability nicht gefunden');
    return;
  }
  const payload = await res.json();
  renderCapabilityDetail(payload);
  await loadCapabilities();
  setStatus(`${payload.capability?.label || capabilityId} geladen`);
}

function renderCapabilityDetail(payload) {
  const cap = payload.capability || {};
  const related = payload.related || [];
  const groups = groupRelations(related);
  document.getElementById('capabilityDetail').innerHTML = `
    <h2>${escapeHtml(cap.label)}</h2>
    <div class="badge-row">
      <span class="badge primary">Capability</span>
      <span class="badge">${escapeHtml(cap.source)}</span>
      <span class="badge">${payload.related_count ?? 0} Beziehungen</span>
    </div>
    <div class="meta-grid">
      <div class="meta-box"><span>ID</span>${escapeHtml(cap.id)}</div>
      <div class="meta-box"><span>Degree</span>${escapeHtml(cap.metadata?.degree ?? 0)}</div>
      <div class="meta-box"><span>Tools</span>${groups.has_tool?.length ?? 0}</div>
      <div class="meta-box"><span>Skills</span>${groups.has_skill?.length ?? 0}</div>
      <div class="meta-box"><span>Knowledge</span>${groups.has_knowledge?.length ?? 0}</div>
      <div class="meta-box"><span>Gaps</span>${groups.has_gap?.length ?? 0}</div>
    </div>
  `;
  document.getElementById('relationGrid').innerHTML = related.length ? related.map(item => relationCard(item)).join('') : '<div class="empty">Keine Beziehungen vorhanden.</div>';
  document.getElementById('rawCapability').textContent = JSON.stringify(payload, null, 2);
}

function groupRelations(related) {
  return related.reduce((acc, item) => {
    const key = item.relation || 'related';
    acc[key] = acc[key] || [];
    acc[key].push(item);
    return acc;
  }, {});
}

function relationCard(item) {
  const node = item.node || {};
  const meta = node.metadata || {};
  const subtitle = meta.relative_path || meta.description || meta.id || node.id;
  return `
    <article class="relation-box">
      <span>${escapeHtml(item.relation)} · ${escapeHtml(node.type)}</span>
      <h3>${escapeHtml(node.label)}</h3>
      <p>${escapeHtml(subtitle || '')}</p>
      <div class="badge-row">
        <span class="badge">${escapeHtml(node.source)}</span>
        ${meta.cloud_allowed === false ? '<span class="badge danger">local only</span>' : ''}
        ${meta.status ? `<span class="badge">${escapeHtml(meta.status)}</span>` : ''}
      </div>
    </article>
  `;
}

async function rebuildGraph() {
  setStatus('Baue Capability Graph neu auf ...');
  const res = await fetch('/api/capabilities/rebuild', { method: 'POST' });
  const payload = await res.json();
  renderSummary(payload);
  await loadCapabilities();
  setStatus('Capability Graph neu aufgebaut');
}

window.addEventListener('DOMContentLoaded', loadDashboard);
