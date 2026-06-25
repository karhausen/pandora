const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
let currentData = null;

function stateLabel(state) {
  return {active:'Aktiv', blocked:'Blockiert', finished:'Abgeschlossen', empty:'Leer'}[state] || state || 'unbekannt';
}

async function loadDashboard() {
  const params = new URLSearchParams();
  const state = $('stateFilter')?.value || '';
  const query = $('queryInput')?.value || '';
  if (state) params.set('state', state);
  if (query) params.set('query', query);
  const dash = await fetch('/api/workflow-dashboard').then(r => r.json());
  currentData = dash;
  $('activeCount').textContent = dash.counts?.active ?? 0;
  $('blockedCount').textContent = dash.counts?.blocked ?? 0;
  $('finishedCount').textContent = dash.counts?.finished ?? 0;
  $('openActionCount').textContent = dash.counts?.open_actions ?? 0;

  const list = await fetch('/api/workflow-dashboard/workflows?' + params.toString()).then(r => r.json());
  renderRows(list.workflows || []);
}

function renderRows(rows) {
  $('workflowTable').innerHTML = rows.map(w => {
    const cur = w.current_step || {};
    return `<tr class="${esc(w.state)}">
      <td><strong>${esc(w.workflow_id)}</strong></td>
      <td><span class="status ${esc(w.state)}">${esc(stateLabel(w.state))}</span></td>
      <td>${esc(w.progress_label || '')}</td>
      <td>${esc(cur.title || '–')}<br><small>${esc(cur.action_to_do || '')}</small></td>
      <td>${esc(nextText(w))}</td>
      <td>${esc(w.updated_at || w.created_at || '–')}</td>
      <td><button onclick="showWorkflow('${esc(w.workflow_id)}')">Öffnen</button></td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" class="muted">Keine Workflows gefunden.</td></tr>';
}

function nextText(w) {
  if (w.state === 'blocked') return 'Fehler prüfen';
  if (w.state === 'active') return (w.current_step && w.current_step.action_to_do) || 'Aktuellen Schritt bearbeiten';
  if (w.state === 'finished') return 'Erledigt';
  return 'Keine Aktion';
}

async function showWorkflow(id) {
  const data = await fetch('/api/workflow-dashboard/workflows/' + encodeURIComponent(id)).then(r => r.json());
  $('listView').classList.add('hidden');
  $('detailView').classList.remove('hidden');
  $('detailTitle').textContent = data.workflow_id || id;
  $('detailState').textContent = stateLabel(data.summary?.state);
  $('detailState').className = 'badge status ' + (data.summary?.state || 'active');
  $('nextActionBox').textContent = data.next_user_action || 'Keine nächste Aktion.';
  $('timelineBox').innerHTML = renderTimeline(data.timeline || []);
  $('rawBox').textContent = JSON.stringify(data, null, 2);
}

function renderTimeline(rows) {
  return rows.map(s => `<div class="step ${esc(s.state)}">
    <strong>${esc(s.index)}</strong>
    <div><strong>${esc(s.title)}</strong><small>${esc(s.action_to_do || '')}</small><small>${esc(s.action_id || '')}</small></div>
    <span class="status ${esc(s.state)}">${esc(s.state)}</span>
  </div>`).join('') || '<p class="muted">Keine Timeline vorhanden.</p>';
}

function back() {
  $('detailView').classList.add('hidden');
  $('listView').classList.remove('hidden');
}

$('refreshBtn')?.addEventListener('click', loadDashboard);
$('applyFilters')?.addEventListener('click', loadDashboard);
$('backBtn')?.addEventListener('click', back);
loadDashboard().catch(err => { $('workflowTable').innerHTML = `<tr><td colspan="7">Fehler: ${esc(err.message)}</td></tr>`; });
