async function getJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}
function esc(value) { return String(value ?? '').replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s])); }
function badge(score) {
  const cls = score >= 75 ? 'good' : score >= 50 ? 'warn' : 'bad';
  return `<span class="badge ${cls}">${esc(score)}</span>`;
}
async function load() {
  const [status, health, reviews] = await Promise.all([
    getJson('/api/tool-evolution/status'),
    getJson('/api/tool-evolution/health'),
    getJson('/api/tool-evolution/reviews')
  ]);
  document.getElementById('summary').innerHTML = `
    <div class="card"><div class="label">Tools</div><div class="value">${esc(status.tool_count)}</div></div>
    <div class="card"><div class="label">Ø Health</div><div class="value">${esc(status.average_health_score)}</div></div>
    <div class="card"><div class="label">Unhealthy</div><div class="value">${esc(status.unhealthy_count)}</div></div>
    <div class="card"><div class="label">Proposal Candidates</div><div class="value">${esc(status.proposal_candidate_count)}</div></div>
  `;
  const rows = (health.tools || []).map(t => `
    <tr><td><strong>${esc(t.tool_id)}</strong><br><span class="muted">${esc(t.module)}</span></td><td>${esc(t.status)}</td><td>${badge(t.health_score)} ${esc(t.grade)}</td><td>${esc(t.executions)}</td><td>${esc((t.issues || []).join(', ') || '—')}</td></tr>
  `).join('');
  document.getElementById('health').innerHTML = `<table><thead><tr><th>Tool</th><th>Status</th><th>Health</th><th>Runs</th><th>Issues</th></tr></thead><tbody>${rows || '<tr><td colspan="5">Keine Tools gefunden.</td></tr>'}</tbody></table>`;
  const reviewRows = (reviews.reviews || []).map(r => `
    <tr><td><strong>${esc(r.title)}</strong><br><span class="muted">${esc(r.recommendation)}</span></td><td>${esc(r.severity)}</td><td>${badge(r.health_score)}</td><td>${esc((r.issues || []).join(', '))}</td></tr>
  `).join('');
  document.getElementById('reviews').innerHTML = `<table><thead><tr><th>Review</th><th>Severity</th><th>Health</th><th>Issues</th></tr></thead><tbody>${reviewRows || '<tr><td colspan="4">Keine Review-Kandidaten.</td></tr>'}</tbody></table>`;
}
document.getElementById('refresh').addEventListener('click', load);
load().catch(err => { document.getElementById('summary').innerHTML = `<div class="card">Fehler: ${esc(err.message)}</div>`; });
