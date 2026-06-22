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
