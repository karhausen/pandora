async function j(url){ const r = await fetch(url); return await r.json(); }
function pct(v){ return `${Math.round((v || 0) * 100)}%`; }
function esc(s){ return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
async function load(){
  const status = await j('/api/learning/status');
  const patterns = await j('/api/learning/patterns');
  const history = await j('/api/learning/history?limit=25');
  const stats = status.stats || {};
  document.getElementById('status').textContent = status.ok ? 'OK' : 'Fehler';
  document.getElementById('total').textContent = stats.total_decisions ?? 0;
  document.getElementById('acceptance').textContent = pct(stats.acceptance_rate);
  document.getElementById('patterns').innerHTML = (patterns.patterns || []).length ? patterns.patterns.map(p => `<div class="item"><b>${esc(p.proposal_type)}</b> · ${esc(p.label)}<br><small>${esc(p.recommendation)} · Confidence ${pct(p.confidence)}</small></div>`).join('') : '<small>Noch keine belastbaren Muster. Mehr Entscheidungen nötig.</small>';
  document.getElementById('history').innerHTML = (history.decisions || []).length ? history.decisions.map(d => `<div class="item"><b>${esc(d.title)}</b><br>${esc(d.proposal_type)} · ${esc(d.decision)} · ${esc(d.resulting_status)}<br><small>${esc(d.decided_at)} · ${esc(d.decided_by)}</small></div>`).join('') : '<small>Noch keine Entscheidungen gespeichert.</small>';
}
load().catch(err => { document.body.insertAdjacentHTML('beforeend', `<pre>${esc(err)}</pre>`); });
