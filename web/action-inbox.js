let currentId = null;
const $ = (id) => document.getElementById(id);

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

function esc(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

function statusClass(item) {
  if (item.is_failed) return 'failed';
  if (item.is_done) return 'done';
  return 'open';
}

function renderStatus(item) {
  return `<span class="status ${statusClass(item)}">${esc(item.status)}</span>`;
}

function rowOpen(item) {
  return `<tr class="${item.is_failed ? 'failed' : ''}">
    <td><strong>${esc(item.title)}</strong><br><small>${esc(item.category)}</small></td>
    <td>${esc(item.area)}</td>
    <td>${item.workflow_id ? `<span class="badge">${esc(item.workflow_id)}</span><br><small>${esc(item.workflow_step || "")}</small>` : '<span class="muted">–</span>'}</td>
    <td>${esc(item.action_to_do)}</td>
    <td class="priority-${esc(String(item.priority).toLowerCase())}">${esc(item.priority)}</td>
    <td>${renderStatus(item)}</td>
    <td>${item.last_error ? esc(item.last_error) : '<span class="muted">–</span>'}</td>
    <td>${esc(item.created_at || '–')}</td>
    <td><button data-open="${esc(item.id)}">Öffnen</button></td>
  </tr>`;
}

function rowDone(item) {
  return `<tr class="done">
    <td><strong>${esc(item.title)}</strong><br><small>${esc(item.category)}</small></td>
    <td>${esc(item.area)}</td>
    <td>${renderStatus(item)}</td>
    <td>${esc(item.updated_at || item.created_at || '–')}</td>
    <td><button data-open="${esc(item.id)}">Details</button></td>
  </tr>`;
}

async function loadDashboard() {
  const q = $('queryInput')?.value || '';
  const area = $('areaFilter')?.value || '';
  const data = await fetchJson(`/api/actions/dashboard?limit=1000`);
  let open = data.open_actions || [];
  let done = data.done_actions || [];
  if (q) {
    const needle = q.toLowerCase();
    const filter = item => JSON.stringify(item).toLowerCase().includes(needle);
    open = open.filter(filter); done = done.filter(filter);
  }
  if (area) {
    open = open.filter(i => i.area === area || i.category === area);
    done = done.filter(i => i.area === area || i.category === area);
  }
  $('openCount').textContent = open.length;
  $('failedCount').textContent = open.filter(i => i.is_failed).length;
  $('doneCount').textContent = done.length;
  $('totalCount').textContent = open.length + done.length;
  $('openTable').innerHTML = open.length ? open.map(rowOpen).join('') : '<tr><td colspan="9" class="muted">Keine offenen Actions.</td></tr>';
  $('doneTable').innerHTML = done.length ? done.map(rowDone).join('') : '<tr><td colspan="5" class="muted">Noch keine erledigten Actions.</td></tr>';
  document.querySelectorAll('[data-open]').forEach(btn => btn.addEventListener('click', () => openDetail(btn.dataset.open)));
}

function renderWorkflow(workflow) {
  const rows = (workflow && workflow.timeline) || [];
  if (!rows.length) return '<p class="muted">Noch keine Workflow-Kette vorhanden.</p>';
  return `<div class="workflow-id">${esc(workflow.workflow_id || '')} · Schritt ${esc(workflow.current_step || '')}/${esc(workflow.total_steps || '')}</div>` +
    rows.map(r => `<div class="wf-step ${esc(r.state)}"><span>${esc(r.index)}.</span><strong>${esc(r.title)}</strong><em>${esc(r.state)}</em></div>`).join('');
}

function kv(obj) {
  return Object.entries(obj || {}).map(([k,v]) => `<div class="key">${esc(k)}</div><div>${esc(v)}</div>`).join('');
}

async function openDetail(id) {
  currentId = id;
  history.replaceState(null, '', `/action-inbox/${encodeURIComponent(id)}`);
  const data = await fetchJson(`/api/actions/${encodeURIComponent(id)}`);
  if (!data.found) throw new Error('Action nicht gefunden');
  const action = data.action;
  $('listView').classList.add('hidden');
  $('detailView').classList.remove('hidden');
  $('detailTitle').textContent = action.title;
  $('detailStatus').textContent = action.status;
  $('detailStatus').className = `badge ${action.is_failed ? 'danger' : action.is_done ? '' : 'primary'}`;
  $('summaryBox').innerHTML = kv(data.summary);
  $('reasonBox').textContent = data.reason || 'Keine Begründung vorhanden.';
  $('workflowBox').innerHTML = renderWorkflow(data.workflow);
  $('planBox').textContent = JSON.stringify(data.planned_action || {}, null, 2);
  $('errorsBox').innerHTML = (data.errors || []).length ? data.errors.map(e => `<div class="error-entry"><strong>${esc(e.source || 'error')}</strong><br>${esc(e.message)}</div>`).join('') : '<p class="muted">Keine Fehler gemeldet.</p>';
  $('logsBox').innerHTML = (data.logs || []).map(l => `<div class="log-entry"><strong>${esc(l.time || 'ohne Zeit')}</strong> <span class="muted">${esc(l.level || '')}</span><br>${esc(l.message || JSON.stringify(l))}${l.note ? `<br><em>${esc(l.note)}</em>` : ''}</div>`).join('');
  $('artifactsBox').innerHTML = (data.artifacts || []).map(a => `<div class="artifact"><strong>${esc(a.label)}</strong><br><code>${esc(a.path)}</code><br><span class="muted">${esc(a.kind)}</span></div>`).join('');
  $('rawBox').textContent = JSON.stringify({content: data.content, review_state: data.review_state}, null, 2);
}

async function saveDecision() {
  if (!currentId) return;
  const payload = { decision: $('decisionSelect').value, note: $('decisionNote').value || null, decided_by: 'action-inbox' };
  await fetchJson(`/api/actions/${encodeURIComponent(currentId)}/decision`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
  });
  await openDetail(currentId);
}

function back() {
  currentId = null;
  history.replaceState(null, '', '/action-inbox');
  $('detailView').classList.add('hidden');
  $('listView').classList.remove('hidden');
  loadDashboard();
}

window.addEventListener('DOMContentLoaded', async () => {
  $('refreshBtn').addEventListener('click', loadDashboard);
  $('applyFilters').addEventListener('click', loadDashboard);
  $('backBtn').addEventListener('click', back);
  $('decisionBtn').addEventListener('click', saveDecision);
  const path = decodeURIComponent(location.pathname || '');
  if (path.startsWith('/action-inbox/') && path.length > '/action-inbox/'.length) {
    await loadDashboard();
    await openDetail(path.substring('/action-inbox/'.length));
  } else {
    await loadDashboard();
  }
});
