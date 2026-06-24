async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return await response.json();
}
function show(id, data) { document.getElementById(id).textContent = JSON.stringify(data, null, 2); }
async function loadLearning() {
  const [status, metrics, patterns, events, insights] = await Promise.all([
    fetchJson('/api/learning/status'),
    fetchJson('/api/learning/metrics'),
    fetchJson('/api/learning/patterns'),
    fetchJson('/api/learning/events?limit=50'),
    fetchJson('/api/learning/insights')
  ]);
  show('statusBox', status);
  show('metricsBox', metrics);
  show('patternsBox', patterns);
  renderInsights(insights.insights || insights.insights === undefined ? (insights.insights || insights.insights || []) : []);
  const body = document.getElementById('eventsBody');
  body.innerHTML = '';
  for (const event of (events.events || []).slice().reverse()) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${event.event_type || ''}</td><td>${event.source || ''}</td><td>${event.area || ''}</td><td>${event.result || ''}</td><td>${event.title || ''}</td>`;
    body.appendChild(tr);
  }
}
async function collectLearning() {
  const result = await fetchJson('/api/learning/collect', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({limit: 500})});
  alert(`Gesammelt: ${result.written_count || 0}`);
  await loadLearning();
}
async function rebuildLearning() {
  const result = await fetchJson('/api/learning/rebuild', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({limit: 500})});
  alert(`Rebuild OK: ${result.collection?.written_count || 0} neue Events`);
  await loadLearning();
}
loadLearning().catch(err => { document.body.insertAdjacentHTML('beforeend', `<pre>${err}</pre>`); });

function renderInsights(insights) {
  const box = document.getElementById('insightsBox');
  if (!box) return;
  box.innerHTML = '';
  const rows = Array.isArray(insights) ? insights : [];
  if (!rows.length) { box.textContent = 'Keine Insights vorhanden. Rebuild ausführen.'; return; }
  for (const item of rows) {
    const div = document.createElement('article');
    div.className = 'mini-card';
    div.innerHTML = `<h3>${item.title || item.id}</h3><p>${item.summary || ''}</p><p><span class="badge">${item.priority || 'medium'}</span> <span class="badge">${item.status || 'pending'}</span> <span class="badge">${item.insight_type || ''}</span></p><button onclick="decideInsight('${item.id}', 'reviewed')">Als geprüft markieren</button>`;
    box.appendChild(div);
  }
}
async function rebuildInsights() {
  const result = await fetchJson('/api/learning/insights?rebuild=true', {method:'GET'});
  alert(`Insights: ${result.insight_count || 0}`);
  await loadLearning();
}
async function decideInsight(id, decision) {
  await fetchJson(`/api/learning/insights/${encodeURIComponent(id)}/decision`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({decision})});
  await loadLearning();
}
