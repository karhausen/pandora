function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

async function loadStatus() {
  const box = document.getElementById('queueStatus');
  const res = await fetch('/api/proposal-queue/status');
  const data = await res.json();
  const stats = data.stats || {};
  box.innerHTML = `<strong>Version ${esc(data.version)}</strong> · ${esc(stats.total || 0)} Proposals · ${esc(stats.high_priority || 0)} hohe Priorität · Modus: ${esc(data.mode)} · Aktiviert Änderungen: ${data.activates_changes ? 'ja' : 'nein'}`;
}

async function loadItems() {
  const target = document.getElementById('queueItems');
  const params = new URLSearchParams();
  params.set('limit', '100');
  const q = document.getElementById('queryInput').value.trim();
  const status = document.getElementById('statusFilter').value;
  const type = document.getElementById('typeFilter').value;
  if (q) params.set('query', q);
  if (status) params.set('status', status);
  if (type) params.set('type', type);
  const res = await fetch(`/api/proposal-queue/items?${params.toString()}`);
  const data = await res.json();
  const items = data.items || [];
  if (!items.length) {
    target.innerHTML = `<div class="empty">Keine Proposals in der Queue. Priorisierte Kandidaten können kontrolliert importiert werden.</div>`;
    return;
  }
  target.innerHTML = items.map((item) => `
    <article class="queue-card">
      <h2>${esc(item.title)}</h2>
      <div class="meta">
        <span class="badge">${esc(item.proposal_type)}</span>
        <span class="badge">Status: ${esc(item.queue_status)}</span>
        <span class="badge">Lifecycle: ${esc(item.lifecycle_status)}</span>
        <span class="badge">Priorität: ${esc(item.priority)}</span>
        <span class="badge">Confidence: ${Math.round((item.confidence || 0) * 100)}%</span>
        <span class="badge">Risk: ${esc(item.risk)}</span>
      </div>
      <p class="description">${esc(item.description)}</p>
      <small>${esc(item.proposal_id)} · Quelle: ${esc(item.source)} · ${esc(item.created_at)}</small>
    </article>
  `).join('');
}

async function importPrioritized() {
  const res = await fetch('/api/proposal-queue/import-prioritized?limit=50&min_priority=60', { method: 'POST' });
  await res.json();
  await loadStatus();
  await loadItems();
}

async function refresh() {
  try {
    await loadStatus();
    await loadItems();
  } catch (error) {
    document.getElementById('queueStatus').textContent = `Proposal Queue konnte nicht geladen werden: ${error}`;
  }
}

document.getElementById('refreshBtn').addEventListener('click', refresh);
document.getElementById('importBtn').addEventListener('click', importPrioritized);
document.getElementById('queryInput').addEventListener('input', () => setTimeout(refresh, 0));
document.getElementById('statusFilter').addEventListener('change', refresh);
document.getElementById('typeFilter').addEventListener('change', refresh);
refresh();
