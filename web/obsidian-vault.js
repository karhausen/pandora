let selectedInboxPath = null;
function $(id) { return document.getElementById(id); }
function setStatus(text) { $('statusText').textContent = text; }
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
function badge(text, cls='') { return `<span class="badge ${cls}">${escapeHtml(text)}</span>`; }
function raw(data) { $('rawOutput').textContent = JSON.stringify(data, null, 2); }
async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({ok:false, error:'Invalid JSON response'}));
  if (!res.ok) data.http_error = res.status;
  return data;
}
async function loadStatus() {
  setStatus('Prüfe Obsidian-Status ...');
  const data = await fetchJson('/api/obsidian/status');
  $('vaultOk').textContent = data.ok ? 'OK' : 'Nicht bereit';
  $('vaultPath').textContent = data.config?.vault_path || 'nicht konfiguriert';
  $('inboxStatus').textContent = data.inbox_exists ? (data.inbox_writable ? 'bereit' : 'nicht schreibbar') : 'fehlt';
  $('cloudStatus').textContent = data.config?.cloud_allowed ? 'cloud erlaubt' : 'local only';
  raw(data);
  setStatus(data.ok ? 'Obsidian bereit' : (data.issues || []).join(' · ') || 'Nicht bereit');
}
async function ensureInbox() {
  setStatus('Lege Pandora_Inbox an ...');
  const data = await fetchJson('/api/obsidian/ensure-inbox', {method:'POST'});
  raw(data);
  await loadStatus();
  setStatus(data.ok ? 'Inbox geprüft/angelegt' : (data.detail || data.error || 'Fehler'));
}
async function reindexVault() {
  setStatus('Indexiere Vault ...');
  const data = await fetchJson('/api/obsidian/reindex?limit=10000&write=true', {method:'POST'});
  raw(data);
  setStatus(data.ok ? `Index bereit: ${data.file_count || 0} Dateien, ${data.tag_count || 0} Tags` : (data.detail || data.error || 'Fehler'));
}
async function searchVault() {
  const query = $('searchInput').value.trim();
  if (!query) { setStatus('Bitte Suchbegriff eingeben.'); return; }
  setStatus('Suche im Vault ...');
  const data = await fetchJson(`/api/obsidian/search?query=${encodeURIComponent(query)}&limit=30&include_content=false`);
  raw(data);
  const results = data.results || [];
  $('searchResults').innerHTML = results.length ? results.map(item => `
    <button class="item" type="button" onclick='showSearchResult(${JSON.stringify(item).replace(/'/g, '&apos;')})'>
      <h3>${escapeHtml(item.title)} ${badge('Score ' + (item.score ?? 0), 'primary')}</h3>
      <p>${escapeHtml(item.relative_path)}</p>
      <div class="badge-row">${(item.tags || []).slice(0,5).map(t => badge('#'+t)).join('')}${(item.wikilinks || []).slice(0,4).map(l => badge('[['+l+']]')).join('')}</div>
      <p>${escapeHtml(item.excerpt || '')}</p>
    </button>
  `).join('') : '<div class="item">Keine Treffer.</div>';
  setStatus(`${results.length} Treffer`);
}
function showSearchResult(item) {
  $('detailPanel').innerHTML = `
    <h3>${escapeHtml(item.title)}</h3>
    <div class="badge-row">${badge(item.relative_path, 'primary')}${badge(`${item.word_count || 0} Wörter`)}</div>
    <h4>Tags</h4><p>${escapeHtml((item.tags || []).join(', ') || '–')}</p>
    <h4>Wikilinks</h4><p>${escapeHtml((item.wikilinks || []).join(', ') || '–')}</p>
    <h4>Vorschau</h4><pre>${escapeHtml(item.excerpt || '')}</pre>
  `;
}

async function previewContext() {
  const query = $('contextInput').value.trim() || $('searchInput').value.trim();
  const provider = $('contextTarget').value;
  if (!query) { setStatus('Bitte Context-Frage oder Suchbegriff eingeben.'); return; }
  setStatus('Baue Obsidian Context Preview ...');
  const data = await fetchJson(`/api/obsidian/context-preview?query=${encodeURIComponent(query)}&provider_name=${encodeURIComponent(provider)}&limit=5`);
  raw(data);
  const obs = data.obsidian || {};
  const sources = data.sources || [];
  const blocked = data.blocked_obsidian_count || obs.blocked_count || 0;
  $('contextPreview').innerHTML = `
    <div class="item">
      <h3>${escapeHtml(data.cloud_context ? 'Cloud/Company Kontext' : 'Lokaler Kontext')}</h3>
      <div class="badge-row">${badge('Quellen ' + sources.length, 'primary')}${blocked ? badge('blockiert ' + blocked) : ''}${badge(obs.cloud_allowed ? 'cloud erlaubt' : 'local only')}</div>
      <p>${escapeHtml(obs.blocked_reason || data.rule || '')}</p>
    </div>
    ${sources.length ? sources.map(src => `
      <div class="item">
        <h3>${escapeHtml(src.title || src.relative_path)}</h3>
        <p>${escapeHtml(src.relative_path)}</p>
        <div class="badge-row">${(src.tags || []).slice(0,5).map(t => badge('#'+t)).join('')}${badge('Score ' + (src.score ?? 0))}</div>
      </div>
    `).join('') : '<div class="item">Keine Obsidian-Quelle im Kontext. Prüfe Vault-Status oder Cloud-Policy.</div>'}
  `;
  setStatus(`Context Preview: ${sources.length} Quellen, ${blocked} blockiert`);
}

async function exportNote() {
  const tags = $('exportTags').value.split(',').map(t => t.trim()).filter(Boolean);
  const payload = {
    title: $('exportTitle').value.trim(),
    content: $('exportContent').value,
    category: $('exportCategory').value,
    tags,
    suggested_folder: $('suggestedFolder').value.trim() || null,
  };
  if (!payload.title) { setStatus('Titel fehlt.'); return; }
  setStatus('Exportiere nach Pandora_Inbox ...');
  const data = await fetchJson('/api/obsidian/export', {
    method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)
  });
  $('exportResult').innerHTML = data.ok ? `${badge('exportiert', 'primary')} <code>${escapeHtml(data.relative_path)}</code>` : `<span class="danger-text">${escapeHtml(data.detail || data.error || 'Fehler')}</span>`;
  raw(data);
  if (data.ok) await loadInbox();
  setStatus(data.ok ? 'Export abgeschlossen' : 'Export fehlgeschlagen');
}
async function loadInbox() {
  setStatus('Lade Pandora_Inbox ...');
  const status = $('inboxFilter').value;
  const url = `/api/obsidian/inbox/items?limit=200${status ? '&status=' + encodeURIComponent(status) : ''}`;
  const data = await fetchJson(url);
  raw(data);
  const items = data.items || [];
  $('inboxList').innerHTML = items.length ? items.map(item => `
    <button class="item ${item.relative_inbox_path === selectedInboxPath ? 'selected' : ''}" type="button" onclick="showInboxItem('${escapeHtml(item.relative_inbox_path)}')">
      <h3>${escapeHtml(item.title || item.name)} ${badge(item.review_status || 'pending', item.review_status === 'reviewed' ? 'primary' : '')}</h3>
      <p>${escapeHtml(item.relative_inbox_path)}</p>
      <div class="badge-row">${badge(item.category || 'Inbox')}${item.suggested_folder ? badge('→ ' + item.suggested_folder, 'primary') : ''}</div>
    </button>
  `).join('') : '<div class="item">Keine Inbox-Einträge.</div>';
  setStatus(`${items.length} Inbox-Einträge`);
}
async function showInboxItem(path) {
  selectedInboxPath = path;
  setStatus('Lade Inbox-Eintrag ...');
  const data = await fetchJson(`/api/obsidian/inbox/items/${encodeURIComponent(path)}`);
  raw(data);
  const item = data.item || {};
  $('detailPanel').innerHTML = `
    <h3>${escapeHtml(item.title || item.name || path)}</h3>
    <div class="badge-row">${badge(item.review_status || 'pending', 'primary')}${badge(item.relative_inbox_path || path)}${item.suggested_folder ? badge('Vorschlag: ' + item.suggested_folder) : ''}</div>
    <div class="form-row" style="margin: .8rem 0;">
      <select id="markStatus">
        <option value="reviewed">reviewed</option>
        <option value="accepted_for_sorting">accepted_for_sorting</option>
        <option value="needs_revision">needs_revision</option>
        <option value="rejected">rejected</option>
      </select>
      <input id="markNote" placeholder="Notiz zur Prüfung" />
      <button type="button" onclick="markInboxItem('${escapeHtml(path)}')">Markieren</button>
    </div>
    <pre>${escapeHtml(item.content || '')}</pre>
  `;
  await loadInbox();
  setStatus('Inbox-Eintrag geladen');
}
async function markInboxItem(path) {
  const payload = {status: $('markStatus').value, note: $('markNote').value || null, reviewed_by: 'user'};
  setStatus('Speichere Review-Status ...');
  const data = await fetchJson(`/api/obsidian/inbox/items/${encodeURIComponent(path)}/mark`, {
    method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)
  });
  raw(data);
  await loadInbox();
  setStatus(data.ok ? 'Review-Status gespeichert' : 'Speichern fehlgeschlagen');
}
window.addEventListener('DOMContentLoaded', async () => { await loadStatus(); await loadInbox(); });

async function buildImportCandidates() {
  const query = $('importCandidateQuery')?.value?.trim() || '';
  setStatus('Erzeuge Obsidian Import-Kandidaten ...');
  const url = `/api/obsidian/import-candidates/build?limit=50&write=true${query ? '&query=' + encodeURIComponent(query) : ''}`;
  const data = await fetchJson(url, {method:'POST'});
  raw(data);
  await loadImportCandidates();
  setStatus(data.ok === false ? (data.detail || data.error || 'Fehler') : `${data.candidate_count || 0} Import-Kandidaten erzeugt`);
}

async function loadImportCandidates() {
  setStatus('Lade Obsidian Import-Kandidaten ...');
  const area = $('importCandidateArea')?.value || '';
  const status = $('importCandidateStatus')?.value || '';
  const query = $('importCandidateQuery')?.value?.trim() || '';
  const params = new URLSearchParams({limit:'200'});
  if (area) params.set('target_area', area);
  if (status) params.set('status', status);
  if (query) params.set('query', query);
  const data = await fetchJson('/api/obsidian/import-candidates?' + params.toString());
  raw(data);
  const items = data.candidates || [];
  $('importCandidates').innerHTML = items.length ? items.map(item => `
    <button class="item" type="button" onclick="showImportCandidate('${escapeHtml(item.id)}')">
      <h3>${escapeHtml(item.title)} ${badge(item.priority || 'medium', 'primary')}</h3>
      <p>${escapeHtml(item.source_relative_path)}</p>
      <div class="badge-row">${badge(item.target_area || 'target')}${badge(item.status || 'pending_review')}${item.suggested_folder ? badge('→ ' + item.suggested_folder) : ''}</div>
      <p>${escapeHtml(item.reason || item.summary || '')}</p>
    </button>
  `).join('') : '<div class="item">Keine Import-Kandidaten.</div>';
  setStatus(`${items.length} Import-Kandidaten`);
}

async function showImportCandidate(id) {
  setStatus('Lade Import-Kandidat ...');
  const data = await fetchJson(`/api/obsidian/import-candidates/${encodeURIComponent(id)}`);
  raw(data);
  const c = data.candidate || {};
  const preview = data.source_preview || {};
  $('importCandidateDetail').innerHTML = data.found ? `
    <h3>${escapeHtml(c.title)}</h3>
    <div class="badge-row">${badge(c.target_area, 'primary')}${badge(c.status || 'pending_review')}${badge(c.source_relative_path || '')}</div>
    <h4>Vorgeschlagener Zielpfad</h4>
    <p><code>${escapeHtml(c.proposed_target_path || '')}</code></p>
    <h4>Metadaten</h4>
    <pre>${escapeHtml(JSON.stringify(c.proposed_metadata || {}, null, 2))}</pre>
    <h4>Quelle Vorschau</h4>
    <pre>${escapeHtml(preview.content_preview || c.summary || '')}</pre>
    <div class="form-row" style="margin-top:.8rem;">
      <select id="candidateDecision">
        <option value="accepted_for_next_step">accepted_for_next_step</option>
        <option value="reviewed">reviewed</option>
        <option value="needs_work">needs_work</option>
        <option value="deferred">deferred</option>
        <option value="rejected">rejected</option>
      </select>
      <input id="candidateNote" placeholder="Notiz" />
      <button type="button" onclick="markImportCandidate('${escapeHtml(c.id)}')">Entscheidung speichern</button>
    </div>
  ` : '<div class="item">Kandidat nicht gefunden.</div>';
  setStatus(data.found ? 'Import-Kandidat geladen' : 'Nicht gefunden');
}

async function markImportCandidate(id) {
  const payload = {decision: $('candidateDecision').value, note: $('candidateNote').value || null, decided_by:'user'};
  setStatus('Speichere Import-Kandidaten-Entscheidung ...');
  const data = await fetchJson(`/api/obsidian/import-candidates/${encodeURIComponent(id)}/decision`, {
    method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)
  });
  raw(data);
  await loadImportCandidates();
  setStatus(data.ok ? 'Entscheidung gespeichert' : 'Entscheidung fehlgeschlagen');
}

window.addEventListener('DOMContentLoaded', async () => { try { await loadImportCandidates(); } catch (err) {} });
