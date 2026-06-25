let currentData = null;

async function loadDashboard() {
  const res = await fetch('/api/gui/guided-improvement/dashboard?limit=200');
  const data = await res.json();
  currentData = data;
  renderDashboard(data);
}

async function rebuildRecommendations() {
  const res = await fetch('/api/gui/guided-improvement/rebuild', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({write: true, limit: 200})
  });
  const data = await res.json();
  document.getElementById('rawBox').textContent = JSON.stringify(data, null, 2);
  await loadDashboard();
}

function renderDashboard(data) {
  const status = data.status || {};
  const counts = status.counts || {};
  document.getElementById('headlineText').textContent = `${counts.open || 0} offen, ${counts.total || 0} gesamt`;
  document.getElementById('summaryCards').innerHTML = `
    <div class="mini-card"><strong>${counts.open || 0}</strong><span>Offen</span></div>
    <div class="mini-card"><strong>${counts.total || 0}</strong><span>Gesamt</span></div>
    <div class="mini-card"><strong>${Object.keys(status.by_type || {}).length}</strong><span>Typen</span></div>
  `;
  const rows = ((data.recommendations || {}).recommendations || []);
  document.getElementById('recommendationList').innerHTML = rows.length ? rows.map(renderRecommendation).join('') : '<p class="muted">Keine offenen Vorschläge.</p>';
  document.getElementById('rawBox').textContent = JSON.stringify(data, null, 2);
}

function renderRecommendation(item) {
  const id = escapeHtml(item.id || '');
  return `
    <article class="recommendation-item">
      <h3>${escapeHtml(item.title || id)}</h3>
      <p>${escapeHtml(item.summary || '')}</p>
      <div class="recommendation-meta">
        <span class="badge">${escapeHtml(item.area || 'Unknown')}</span>
        <span class="badge">${escapeHtml(item.improvement_type || '')}</span>
        <span class="badge">${escapeHtml(item.priority || 'medium')}</span>
        <span class="badge">${escapeHtml(item.status || 'pending')}</span>
      </div>
      <div class="row">
        <button onclick="showRecommendation('${id}')">Details</button>
        <button onclick="decideRecommendation('${id}', 'accepted_for_next_step')">Nächsten Schritt erlauben</button>
        <button onclick="decideRecommendation('${id}', 'reviewed')">Als geprüft markieren</button>
        <button onclick="decideRecommendation('${id}', 'rejected')">Ablehnen</button>
      </div>
    </article>
  `;
}

async function showRecommendation(id) {
  const res = await fetch(`/api/guided-improvement/recommendations/${encodeURIComponent(id)}`);
  const data = await res.json();
  document.getElementById('detailBox').textContent = JSON.stringify(data.recommendation || data, null, 2);
}

async function decideRecommendation(id, decision) {
  const note = decision === 'accepted_for_next_step' ? 'Nächster kontrollierter Schritt erlaubt.' : '';
  const res = await fetch(`/api/guided-improvement/recommendations/${encodeURIComponent(id)}/decision`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({decision, note})
  });
  const data = await res.json();
  document.getElementById('detailBox').textContent = JSON.stringify(data, null, 2);
  await loadDashboard();
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}

loadDashboard().catch(err => {
  document.getElementById('headlineText').textContent = 'Fehler beim Laden';
  document.getElementById('rawBox').textContent = String(err);
});
