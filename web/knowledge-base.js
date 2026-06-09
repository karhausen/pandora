let selectedArea = null;
let selectedPath = null;

function setStatus(message) { document.getElementById('statusText').textContent = message; }
function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#039;');
}
function bytes(value) {
  const n = Number(value || 0);
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
function policyBadge(item) {
  if (item.cloud_allowed) return `<span class="badge primary">${escapeHtml(item.policy)}</span>`;
  return `<span class="badge">local only</span>`;
}
function metadataBadges(item) {
  const meta = item.metadata || {};
  const tags = (meta.tags || item.tags || []).slice(0, 4).map(t => `<span class="badge">#${escapeHtml(t)}</span>`).join('');
  const priority = meta.priority || item.priority;
  const title = meta.title || item.title;
  return `${title ? `<span class="badge primary">${escapeHtml(title)}</span>` : ''}${priority ? `<span class="badge">${escapeHtml(priority)}</span>` : ''}${tags}`;
}
function governanceBadge(validation) {
  if (!validation) return '<span class="badge">Governance unbekannt</span>';
  if (validation.ok) return '<span class="badge primary">Governance OK</span>';
  return '<span class="badge danger">Policy Problem</span>';
}

async function ensureStructure() {
  setStatus('Lege Struktur an ...');
  const res = await fetch('/api/gui/knowledge/ensure-structure', { method: 'POST' });
  const data = await res.json();
  document.getElementById('rawKnowledge').textContent = JSON.stringify(data, null, 2);
  await loadKnowledgeBase();
  setStatus('Struktur angelegt/geprüft');
}

async function loadKnowledgeBase() {
  setStatus('Lade Knowledge Base ...');
  const res = await fetch('/api/gui/knowledge/dashboard');
  const data = await res.json();
  document.getElementById('areaCount').textContent = data.area_count ?? 0;
  document.getElementById('fileCount').textContent = data.total_files ?? 0;
  document.getElementById('cloudCount').textContent = (data.areas || []).filter(a => a.cloud_allowed).reduce((s, a) => s + (a.file_count || 0), 0);
  document.getElementById('localOnlyCount').textContent = (data.areas || []).filter(a => !a.cloud_allowed).reduce((s, a) => s + (a.file_count || 0), 0);
  renderAreas(data.areas || []);
  document.getElementById('rawKnowledge').textContent = JSON.stringify(data, null, 2);
  await loadGovernanceStatus();
  setStatus('Bereit');
}

function renderAreas(areas) {
  const el = document.getElementById('areaList');
  el.innerHTML = areas.map(area => `
    <button class="knowledge-item ${area.name === selectedArea ? 'selected' : ''}" onclick="loadArea('${escapeHtml(area.name)}')">
      <h3>${escapeHtml(area.name)}</h3>
      <p>${escapeHtml(area.description)}</p>
      <div class="badge-row">${policyBadge(area)}<span class="badge">${area.file_count} Dateien</span><span class="badge">${bytes(area.total_bytes)}</span></div>
    </button>
  `).join('') || '<div class="knowledge-item">Keine Bereiche gefunden.</div>';
}

async function loadArea(area) {
  selectedArea = area;
  selectedPath = null;
  setStatus(`Lade Bereich ${area} ...`);
  const res = await fetch(`/api/gui/knowledge/areas/${encodeURIComponent(area)}`);
  const data = await res.json();
  renderFiles(data.files || []);
  document.getElementById('knowledgeDetail').innerHTML = `<h2>${escapeHtml(area)}</h2><p>${escapeHtml(data.description || '')}</p><div class="badge-row">${policyBadge(data)}<span class="badge">${data.count || 0} Datei(en)</span></div>`;
  document.getElementById('rawKnowledge').textContent = JSON.stringify(data, null, 2);
  await loadKnowledgeBase();
  setStatus(`${data.count || 0} Datei(en) geladen`);
}

function renderFiles(files) {
  const el = document.getElementById('fileList');
  if (!files.length) {
    el.innerHTML = '<div class="knowledge-item"><h3>Keine Dateien</h3><p>Lege Markdown-, Text- oder JSON-Dateien in diesem Bereich ab.</p></div>';
    return;
  }
  el.innerHTML = files.map(file => `
    <button class="knowledge-item ${file.relative_path === selectedPath ? 'selected' : ''}" onclick="showFile('${escapeHtml(file.relative_path)}')">
      <h3>${escapeHtml(file.name)} <span class="badge">${escapeHtml(file.type)}</span></h3>
      <p>${escapeHtml(file.relative_path)}</p>
      <div class="badge-row">${metadataBadges(file)}<span class="badge">${bytes(file.size_bytes)}</span></div>
    </button>
  `).join('');
}

async function showFile(path) {
  if (!selectedArea) return;
  selectedPath = path;
  setStatus(`Lade ${path} ...`);
  const res = await fetch(`/api/gui/knowledge/areas/${encodeURIComponent(selectedArea)}/files/${encodeURIComponent(path)}`);
  const payload = await res.json();
  if (!res.ok) { setStatus('Datei nicht gefunden'); return; }
  document.getElementById('knowledgeDetail').innerHTML = `
    <h2>${escapeHtml(payload.relative_path)}</h2>
    <div class="badge-row">
      <span class="badge primary">${escapeHtml(payload.type)}</span>
      <span class="badge">${bytes(payload.size_bytes)}</span>
      ${policyBadge(payload)}
      <span class="badge">read-only</span>
      ${metadataBadges(payload)}
    </div>
    <h3>Metadaten</h3>
    <pre>${escapeHtml(JSON.stringify(payload.metadata || {}, null, 2))}</pre>
    <h3>Vorschau</h3>
    <pre>${escapeHtml(payload.preview || JSON.stringify(payload.content || {}, null, 2))}</pre>
  `;
  document.getElementById('rawKnowledge').textContent = JSON.stringify(payload, null, 2);
  setStatus('Datei geladen');
}

async function searchKnowledge(cloudOnly) {
  const query = document.getElementById('searchInput').value.trim();
  if (!query) { setStatus('Bitte Suchbegriff eingeben.'); return; }
  setStatus(cloudOnly ? 'Suche cloud-sicher ...' : 'Suche lokal ...');
  const res = await fetch(`/api/gui/knowledge/search?query=${encodeURIComponent(query)}&limit=50&cloud_context=${cloudOnly}`);
  const data = await res.json();
  const el = document.getElementById('searchResults');
  if (!data.results?.length) {
    el.innerHTML = '<div class="knowledge-item">Keine Treffer.</div>';
  } else {
    el.innerHTML = data.results.map(item => `
      <button class="knowledge-item" onclick="selectedArea='${escapeHtml(item.area)}'; showFile('${escapeHtml(item.relative_path)}')">
        <h3>${escapeHtml(item.area)} / ${escapeHtml(item.name)} <span class="badge">${escapeHtml(item.type)}</span></h3>
        <div class="badge-row">${policyBadge(item)}${metadataBadges(item)}<span class="badge">Score ${escapeHtml(item.score || 0)}</span></div>
        <p>${escapeHtml(item.relative_path)}</p>
        <p>${escapeHtml(item.snippet || '')}</p>
      </button>
    `).join('');
  }
  setStatus(`${data.count || 0} Treffer`);
}

async function previewInjection() {
  const query = document.getElementById('searchInput').value.trim();
  if (!query) { setStatus('Bitte Suchbegriff eingeben.'); return; }
  setStatus('Erzeuge Context-Injection Preview ...');
  const res = await fetch(`/api/gui/knowledge/context-injection-preview?query=${encodeURIComponent(query)}&limit=5`);
  const data = await res.json();
  const el = document.getElementById('searchResults');
  el.innerHTML = `
    <div class="knowledge-item selected">
      <h3>Context-Injection Preview</h3>
      <div class="badge-row">
        <span class="badge primary">${escapeHtml(data.target)}</span>
        <span class="badge">${data.source_count || 0} Quellen</span>
        <span class="badge">${data.context_chars || 0} Zeichen</span>
        <span class="badge">${data.blocked_local_only_count || 0} local-only blockiert</span>
      </div>
      <p>${escapeHtml(data.rule || '')}</p>
      <pre>${escapeHtml(data.context_text || 'Kein passender Knowledge-Kontext gefunden.')}</pre>
    </div>
  `;
  document.getElementById('rawKnowledge').textContent = JSON.stringify(data, null, 2);
  setStatus('Context-Injection Preview bereit');
}

async function loadGovernanceStatus() {
  try {
    const res = await fetch('/api/gui/knowledge/governance/status');
    const data = await res.json();
    const el = document.getElementById('governanceStatus');
    if (el) el.textContent = data.ok ? 'OK' : `${data.error_count || 0} Fehler`;
  } catch {
    const el = document.getElementById('governanceStatus');
    if (el) el.textContent = '–';
  }
}

async function loadGovernance() {
  setStatus('Prüfe Knowledge Governance ...');
  const res = await fetch('/api/gui/knowledge/governance?limit=500');
  const data = await res.json();
  const el = document.getElementById('governancePanel');
  const issues = data.issues || [];
  if (!issues.length) {
    el.innerHTML = '<div class="knowledge-item selected"><h3>Governance OK</h3><p>Keine Policy-Probleme gefunden.</p></div>';
  } else {
    el.innerHTML = issues.slice(0, 50).map(issue => `
      <div class="knowledge-item">
        <h3>${escapeHtml(issue.code)} <span class="badge ${issue.severity === 'error' ? 'danger' : ''}">${escapeHtml(issue.severity)}</span></h3>
        <p>${escapeHtml(issue.area)} / ${escapeHtml(issue.relative_path)}</p>
        <p>${escapeHtml(issue.message)}</p>
      </div>
    `).join('');
  }
  document.getElementById('rawKnowledge').textContent = JSON.stringify(data, null, 2);
  await loadGovernanceStatus();
  setStatus(`Governance: ${data.error_count || 0} Fehler, ${data.warning_count || 0} Warnungen`);
}

window.addEventListener('DOMContentLoaded', loadKnowledgeBase);
