let selectedArea = 'public';
let selectedPath = '';

function $(id) { return document.getElementById(id); }
function setStatus(text) { $('statusText').textContent = text; }
function esc(v) { return String(v ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function today() { return new Date().toISOString().slice(0, 10); }

async function api(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  $('rawOutput').textContent = JSON.stringify(data, null, 2);
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

function metadataFromForm() {
  return {
    title: $('metaTitle').value.trim(),
    tags: $('metaTags').value.split(',').map(t => t.trim()).filter(Boolean),
    visibility: $('metaVisibility').value,
    cloud_allowed: $('metaCloudAllowed').value === 'true',
    priority: $('metaPriority').value,
    owner: $('metaOwner').value.trim(),
    last_reviewed: $('metaLastReviewed').value,
    summary: $('metaSummary').value.trim(),
  };
}

function fillMetadata(meta) {
  $('metaTitle').value = meta.title || '';
  $('metaTags').value = (meta.tags || []).join(', ');
  $('metaVisibility').value = meta.visibility || selectedArea;
  $('metaCloudAllowed').value = String(Boolean(meta.cloud_allowed));
  $('metaPriority').value = meta.priority || 'normal';
  $('metaOwner').value = meta.owner || 'thomas';
  $('metaLastReviewed').value = meta.last_reviewed || today();
  $('metaSummary').value = meta.summary || '';
  enforceAreaPolicy();
}

function enforceAreaPolicy() {
  const area = $('areaSelect').value;
  $('metaVisibility').value = area;
  $('currentArea').textContent = area;
  if (area === 'private_local_only') {
    $('metaCloudAllowed').value = 'false';
    $('metaCloudAllowed').disabled = true;
    $('policyHint').textContent = 'private_local_only: Cloud immer gesperrt';
  } else {
    $('metaCloudAllowed').disabled = false;
    $('policyHint').textContent = area === 'public' ? 'public: Cloud erlaubt' : 'restricted: Cloud nach Prüfung';
  }
}

async function loadStatus() {
  const data = await api('/api/gui/knowledge/editor/status');
  $('editorEnabled').textContent = data.enabled ? 'aktiv' : 'aus';
}

async function loadTree() {
  setStatus('Lade Wissensbaum ...');
  const data = await api('/api/gui/knowledge/editor/tree');
  const html = (data.areas || []).map(area => {
    const folders = (area.folders || []).map(f => `<button class="badge link tree-item" onclick="selectFolder('${esc(area.name)}','${esc(f.relative_path)}')">📁 ${esc(f.relative_path)}</button>`).join('');
    const files = (area.files || []).map(f => `<button class="badge link tree-item ${area.name===selectedArea && f.relative_path===selectedPath ? 'selected' : ''}" onclick="openFile('${esc(area.name)}','${esc(f.relative_path)}')">📄 ${esc(f.relative_path)}</button>`).join('');
    return `<section><h3>${esc(area.name)}</h3><div class="badge-row"><span class="badge">${esc(area.policy)}</span><span class="badge">${(area.files||[]).length} Dateien</span></div><div class="compact-list">${folders}${files || '<span class="badge">Keine Dateien</span>'}</div></section>`;
  }).join('');
  $('treeList').innerHTML = html;
  setStatus('Bereit');
}

function selectFolder(area, folder) {
  selectedArea = area;
  $('areaSelect').value = area;
  $('relativePath').value = `${folder.replace(/\/$/, '')}/neue-notiz.md`;
  enforceAreaPolicy();
}

async function newMarkdown() {
  selectedArea = $('areaSelect').value;
  selectedPath = '';
  const path = $('relativePath').value.trim() || 'neue-notiz.md';
  const meta = await api(`/api/gui/knowledge/editor/template?area=${encodeURIComponent(selectedArea)}&relative_path=${encodeURIComponent(path)}`);
  fillMetadata(meta);
  $('relativePath').value = path.endsWith('.md') ? path : `${path}.md`;
  $('bodyEditor').value = '# Neue Notiz\n\n';
  $('currentFile').textContent = 'neu';
  renderGovernance(null);
  setStatus('Neue Datei vorbereitet');
}

async function openFile(area, path) {
  selectedArea = area;
  selectedPath = path;
  $('areaSelect').value = area;
  enforceAreaPolicy();
  setStatus(`Lade ${path} ...`);
  const data = await api(`/api/gui/knowledge/editor/files/${encodeURIComponent(area)}/${encodeURIComponent(path)}`);
  $('relativePath').value = data.relative_path;
  fillMetadata(data.metadata || {});
  $('bodyEditor').value = data.body || '';
  $('currentFile').textContent = data.relative_path;
  renderGovernance(data.governance);
  setStatus('Datei geladen');
}

async function saveFile() {
  enforceAreaPolicy();
  const payload = {
    area: $('areaSelect').value,
    relative_path: $('relativePath').value.trim(),
    metadata: metadataFromForm(),
    body: $('bodyEditor').value,
    overwrite: $('overwriteToggle').checked,
  };
  setStatus('Speichere ...');
  try {
    const data = await api('/api/gui/knowledge/editor/files', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    selectedArea = data.area; selectedPath = data.relative_path;
    $('currentFile').textContent = data.relative_path;
    renderGovernance(data.governance);
    await loadTree();
    setStatus('Gespeichert');
  } catch (e) { setStatus(`Fehler: ${e.message}`); }
}

async function createFolder() {
  const area = $('areaSelect').value;
  const rel = prompt('Neuer Ordnerpfad innerhalb des Bereichs:', 'thema');
  if (!rel) return;
  try {
    setStatus('Lege Ordner an ...');
    await api('/api/gui/knowledge/editor/folders', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({area, relative_path: rel}) });
    await loadTree(); setStatus('Ordner angelegt');
  } catch (e) { setStatus(`Fehler: ${e.message}`); }
}

async function moveFile() {
  const sourceArea = selectedArea || $('areaSelect').value;
  const sourcePath = selectedPath || $('relativePath').value.trim();
  const targetArea = prompt('Zielbereich:', $('areaSelect').value);
  if (!targetArea) return;
  const targetPath = prompt('Zielpfad:', sourcePath);
  if (!targetPath) return;
  try {
    setStatus('Verschiebe ...');
    const data = await api('/api/gui/knowledge/editor/move', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({source_area: sourceArea, source_path: sourcePath, target_area: targetArea, target_path: targetPath, overwrite: false}) });
    await loadTree(); await openFile(data.target_area, data.target_path); setStatus('Verschoben');
  } catch (e) { setStatus(`Fehler: ${e.message}`); }
}

async function deleteCurrent() {
  const area = selectedArea || $('areaSelect').value;
  const path = selectedPath || $('relativePath').value.trim();
  if (!path) { setStatus('Keine Datei ausgewählt.'); return; }
  if (!confirm(`Wirklich löschen?\n${area}/${path}`)) return;
  try {
    setStatus('Lösche ...');
    await api('/api/gui/knowledge/editor/delete', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({area, relative_path: path, confirm: true}) });
    selectedPath = ''; $('currentFile').textContent = 'gelöscht'; $('bodyEditor').value = ''; renderGovernance(null);
    await loadTree(); setStatus('Gelöscht');
  } catch (e) { setStatus(`Fehler: ${e.message}`); }
}

function renderGovernance(gov) {
  if (!gov) { $('governanceState').textContent = '–'; $('governancePanel').innerHTML = '<span class="badge">Noch keine Prüfung</span>'; return; }
  const issues = gov.issues || [];
  $('governanceState').textContent = gov.ok ? 'OK' : `${issues.length} Hinweise`;
  $('governancePanel').innerHTML = issues.length ? issues.map(i => `<div class="badge ${i.severity === 'error' ? 'danger' : ''}">${esc(i.severity)} · ${esc(i.code)} · ${esc(i.message)}</div>`).join('') : '<span class="badge primary">Governance OK</span>';
}

function onAreaChange() { enforceAreaPolicy(); }
window.addEventListener('DOMContentLoaded', async () => { enforceAreaPolicy(); await loadStatus(); await loadTree(); await newMarkdown(); });
