const $ = (id) => document.getElementById(id);
let selectedId = null;
function setStatus(text) { $('statusText').textContent = text; }
function raw(data) { $('rawOutput').textContent = JSON.stringify(data, null, 2); }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch])); }
function badge(text, mode='') { return `<span class="badge ${mode}">${escapeHtml(text || '–')}</span>`; }
async function fetchJson(url, options={}) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) return {ok:false, error:data.detail || res.statusText, status:res.status};
  return data;
}
function params() {
  const p = new URLSearchParams({limit:'200', include_reviewed:'true'});
  const q = $('queryInput').value.trim();
  const area = $('areaFilter').value;
  const status = $('statusFilter').value;
  if (q) p.set('query', q);
  if (area) p.set('target_area', area);
  if (status) p.set('status', status);
  return p.toString();
}
async function loadDashboard() {
  setStatus('Lade Import Review ...');
  const data = await fetchJson('/api/obsidian/import-review?' + params());
  raw(data);
  $('candidateCount').textContent = data.candidate_count ?? (data.candidates || []).length;
  $('executionCount').textContent = data.execution_count ?? (data.executions || []).length;
  renderCandidates(data.candidates || []);
  renderExecutions(data.executions || []);
  setStatus('Import Review geladen');
}
async function buildCandidates() {
  setStatus('Erzeuge Import-Kandidaten aus Obsidian ...');
  const q = $('queryInput').value.trim();
  const url = '/api/obsidian/import-candidates/build?limit=50&write=true' + (q ? '&query=' + encodeURIComponent(q) : '');
  const data = await fetchJson(url, {method:'POST'});
  raw(data);
  await loadDashboard();
}
function renderCandidates(items) {
  $('candidateList').innerHTML = items.length ? items.map(item => `
    <button class="item ${item.id === selectedId ? 'selected' : ''}" type="button" onclick="showCandidate('${escapeHtml(item.id)}')">
      <h3>${escapeHtml(item.title)} ${badge(item.priority || 'medium', item.priority === 'high' ? 'primary' : '')}</h3>
      <p>${escapeHtml(item.source_relative_path)}</p>
      <div class="badge-row">${badge(item.target_area)}${badge(item.status || 'pending_review')}${badge('→ ' + (item.suggested_folder || 'obsidian'))}</div>
      <p>${escapeHtml(item.reason || item.summary || '')}</p>
    </button>
  `).join('') : '<div class="item">Keine Import-Kandidaten.</div>';
}
function renderExecutions(items) {
  $('executionList').innerHTML = items.length ? items.slice(0,20).map(item => `
    <div class="item">
      <h3>${escapeHtml(item.candidate_id || 'Import')} ${badge(item.executed_at || '')}</h3>
      <p><strong>Quelle:</strong> ${escapeHtml(item.source_relative_path || '')}</p>
      <p><strong>Ziel:</strong> ${escapeHtml(item.target_area || '')}/${escapeHtml(item.target_relative_path || '')}</p>
    </div>
  `).join('') : '<div class="item">Noch keine Import-Ausführungen.</div>';
}
async function showCandidate(id) {
  selectedId = id;
  setStatus('Lade Kandidat und Import-Plan ...');
  const data = await fetchJson(`/api/obsidian/import-review/${encodeURIComponent(id)}`);
  raw(data);
  const c = data.candidate || {};
  const preview = data.source_preview || {};
  const plan = data.execution_plan || {};
  const errors = plan.errors || [];
  const warnings = plan.warnings || [];
  $('detailPanel').innerHTML = data.found ? `
    <h3>${escapeHtml(c.title)}</h3>
    <div class="badge-row">${badge(c.status || 'pending_review', c.status === 'accepted_for_next_step' ? 'primary' : '')}${badge(c.target_area)}${badge(c.source_relative_path)}</div>
    <div class="meta-grid">
      <div class="meta-box"><span class="label">Zielpfad</span><br><code>${escapeHtml(c.proposed_target_path || '')}</code></div>
      <div class="meta-box"><span class="label">Ausführbar</span><br>${badge(plan.allowed_to_execute ? 'ja' : 'nein', plan.allowed_to_execute ? 'primary' : '')}</div>
      <div class="meta-box"><span class="label">Überschreiben</span><br>${badge(plan.target?.exists ? 'Ziel existiert' : 'frei')}</div>
    </div>
    <h4>Metadaten-Vorschlag</h4>
    <pre>${escapeHtml(JSON.stringify(c.proposed_metadata || {}, null, 2))}</pre>
    <h4>Konflikte / Hinweise</h4>
    ${errors.length ? '<pre class="danger-text">' + escapeHtml(errors.join('\n')) + '</pre>' : ''}
    ${warnings.length ? '<pre>' + escapeHtml(warnings.join('\n')) + '</pre>' : ''}
    ${!errors.length && !warnings.length ? '<p class="muted">Keine Konflikte im aktuellen Plan.</p>' : ''}
    <h4>Quelle Vorschau</h4>
    <pre>${escapeHtml(preview.content_preview || c.summary || '')}</pre>
    <div class="form-row" style="margin-top:.8rem;">
      <select id="decisionSelect">
        <option value="accepted_for_next_step">accept for next step</option>
        <option value="reviewed">reviewed</option>
        <option value="needs_work">needs_work</option>
        <option value="deferred">deferred</option>
        <option value="rejected">rejected</option>
      </select>
      <input id="decisionNote" placeholder="Notiz zur Entscheidung" />
      <button type="button" onclick="saveDecision('${escapeHtml(c.id)}')">Entscheidung speichern</button>
    </div>
    <div class="form-row" style="margin-top:.8rem;">
      <label><input id="overwriteCheck" type="checkbox" /> overwrite erlauben</label>
      <button type="button" onclick="refreshPlan('${escapeHtml(c.id)}')">Plan aktualisieren</button>
      <button class="badge link primary" type="button" onclick="executeImport('${escapeHtml(c.id)}')">Import ausführen</button>
    </div>
    <div id="actionResult" class="result-box"></div>
  ` : '<div class="item">Kandidat nicht gefunden.</div>';
  setStatus(data.found ? 'Kandidat geladen' : 'Nicht gefunden');
  await loadDashboard();
}
async function saveDecision(id) {
  const payload = {decision: $('decisionSelect').value, note: $('decisionNote').value || null, decided_by:'user'};
  setStatus('Speichere Entscheidung ...');
  const data = await fetchJson(`/api/obsidian/import-review/${encodeURIComponent(id)}/decision`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  raw(data);
  setStatus(data.ok ? 'Entscheidung gespeichert' : 'Entscheidung fehlgeschlagen');
  await showCandidate(id);
}
async function refreshPlan(id) {
  const overwrite = $('overwriteCheck')?.checked ? 'true' : 'false';
  const data = await fetchJson(`/api/obsidian/import-review/${encodeURIComponent(id)}/plan?overwrite=${overwrite}`, {method:'POST'});
  raw(data);
  $('actionResult').innerHTML = `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
  setStatus('Plan aktualisiert');
}
async function executeImport(id) {
  if (!confirm('Import nach user_knowledge/ wirklich ausführen? Obsidian wird nicht verändert.')) return;
  const overwrite = Boolean($('overwriteCheck')?.checked);
  setStatus('Führe Import aus ...');
  const data = await fetchJson(`/api/obsidian/import-review/${encodeURIComponent(id)}/execute`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({confirm:true, overwrite, executed_by:'user'})});
  raw(data);
  $('actionResult').innerHTML = data.ok ? `${badge('importiert', 'primary')} <code>${escapeHtml(data.target?.relative_path || '')}</code>` : `<span class="danger-text">${escapeHtml(data.reason || data.error || 'Import fehlgeschlagen')}</span>`;
  setStatus(data.ok ? 'Import abgeschlossen' : 'Import nicht ausgeführt');
  await loadDashboard();
}
window.addEventListener('DOMContentLoaded', loadDashboard);
