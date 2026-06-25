async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  try { const data = JSON.parse(text); if (!response.ok) return {ok:false, error:data.detail || data}; return data; }
  catch { return {ok: response.ok, text}; }
}
function show(id, data) { document.getElementById(id).textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2); }
function summaryCard(label, value, cls="") { return `<div class="summary ${cls}"><div class="label">${label}</div><div class="value">${value}</div></div>`; }
async function loadDashboard() {
  const data = await api('/api/night-review/status');
  show('dashboardBox', data);
  const open = data.open_recommendation_count || 0;
  const recs = data.recommendation_count || 0;
  const reports = data.report_count || 0;
  document.getElementById('summaryCards').innerHTML = [
    summaryCard('Reports', reports, reports ? 'ok' : 'warn'),
    summaryCard('Empfehlungen', recs, recs ? 'warn' : 'ok'),
    summaryCard('Offen', open, open ? 'warn' : 'ok'),
    summaryCard('Auto Changes', data.safety?.auto_execute ? 'Ja' : 'Nein', data.safety?.auto_execute ? 'danger' : 'ok')
  ].join('');
  await loadRecommendations();
  await loadReports();
}
async function runNightReview() {
  const limit = Number(document.getElementById('runLimit').value || 200);
  const write = document.getElementById('writeRun').checked;
  const create_actions = document.getElementById('createActions').checked;
  show('runBox', {running:true, limit, write, create_actions});
  const data = await api('/api/night-review/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({limit, write, create_actions})});
  show('runBox', data);
  await loadDashboard();
}
async function loadRecommendations() {
  const data = await api('/api/night-review/recommendations?limit=100');
  if (!data.recommendations || data.recommendations.length === 0) { document.getElementById('recommendationList').textContent = 'Keine offenen Empfehlungen.'; return; }
  document.getElementById('recommendationList').innerHTML = data.recommendations.map(r => `
    <div class="item" onclick='show("detailBox", ${JSON.stringify(JSON.stringify(r)).replaceAll("'", "&apos;")})'>
      <div><strong>${r.title || r.id}</strong><small>${r.area || ''} · ${r.priority || ''} · ${r.status || ''}</small></div>
      <button onclick="event.stopPropagation(); decideRecommendation('${encodeURIComponent(r.id)}','reviewed')">Reviewed</button>
    </div>`).join('');
}
async function decideRecommendation(encodedId, decision) {
  const data = await api(`/api/night-review/recommendations/${encodedId}/decision`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({decision})});
  show('detailBox', data);
  await loadDashboard();
}
async function loadReports() {
  const data = await api('/api/night-review/reports?limit=25');
  if (!data.reports || data.reports.length === 0) { document.getElementById('reportList').textContent = 'Keine Reports.'; return; }
  document.getElementById('reportList').innerHTML = data.reports.map(r => `
    <div class="item" onclick="showReport('${encodeURIComponent(r.id)}')">
      <div><strong>${r.title || r.id}</strong><small>${r.created_at || ''} · ${r.recommendation_count || 0} Empfehlungen</small></div>
      <span class="pill">${r.status || 'available'}</span>
    </div>`).join('');
}
async function showReport(encodedId) { const data = await api(`/api/night-review/reports/${encodedId}`); show('detailBox', data); }
loadDashboard();
