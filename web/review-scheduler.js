async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  try { const data = JSON.parse(text); if (!response.ok) return {ok:false, error:data.detail || data}; return data; }
  catch { return {ok: response.ok, text}; }
}
function show(id, data) { document.getElementById(id).textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2); }
function card(label, value, cls='') { return `<div class="summary ${cls}"><div class="label">${label}</div><div class="value">${value}</div></div>`; }
async function loadStatus() {
  const data = await api('/api/review-scheduler/status');
  show('statusBox', data);
  const cfg = data.config || {};
  const due = data.due || {};
  document.getElementById('summaryCards').innerHTML = [
    card('Enabled', cfg.enabled ? 'Ja' : 'Nein', cfg.enabled ? 'ok' : 'warn'),
    card('Zeit', cfg.time || '02:00'),
    card('Due', due.due ? 'Ja' : 'Nein', due.due ? 'warn' : 'ok'),
    card('Runs', data.run_count || 0)
  ].join('');
  await loadHistory();
}
async function runManual() {
  const limit = Number(document.getElementById('manualLimit').value || 200);
  const write = document.getElementById('manualWrite').checked;
  const create_actions = document.getElementById('manualActions').checked;
  show('manualBox', {running:true, limit, write, create_actions});
  const data = await api('/api/review-scheduler/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({limit, write, create_actions})});
  show('manualBox', data);
  await loadStatus();
}
async function runIfDue(force) {
  show('dueBox', {running:true, force});
  const data = await api('/api/review-scheduler/run-if-due', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({force})});
  show('dueBox', data);
  await loadStatus();
}
async function loadHistory() {
  const data = await api('/api/review-scheduler/history?limit=50');
  if (!data.runs || data.runs.length === 0) { document.getElementById('historyList').textContent = 'Keine Scheduler-Läufe.'; return; }
  document.getElementById('historyList').innerHTML = data.runs.map(r => `
    <div class="item">
      <div><strong>${r.trigger || 'run'}</strong><small>${r.ts || ''} · ${r.report_id || 'kein Report'} · ${r.recommendation_count || 0} Empfehlungen</small></div>
      <span class="pill">${r.ok ? 'OK' : 'CHECK'}</span>
    </div>`).join('');
}
loadStatus();
