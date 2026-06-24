async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return await response.json();
}
function show(id, data) { document.getElementById(id).textContent = JSON.stringify(data, null, 2); }
async function loadLearning() {
  const [status, metrics, patterns, events] = await Promise.all([
    fetchJson('/api/learning/status'),
    fetchJson('/api/learning/metrics'),
    fetchJson('/api/learning/patterns'),
    fetchJson('/api/learning/events?limit=50')
  ]);
  show('statusBox', status);
  show('metricsBox', metrics);
  show('patternsBox', patterns);
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
