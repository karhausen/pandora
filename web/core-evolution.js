async function j(url){const r=await fetch(url); if(!r.ok) throw new Error(url); return await r.json();}
function esc(x){return String(x??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function load(){
  const [status, health, refs] = await Promise.all([j('/api/core-evolution/status'), j('/api/core-evolution/health'), j('/api/core-evolution/refactoring')]);
  document.getElementById('summary').innerHTML = [
    ['Health', health.health_score], ['Grade', health.grade], ['Core Files', health.core_file_count], ['Risk Hotspots', health.risk_hotspot_count], ['Candidates', refs.count], ['Mode', status.mode]
  ].map(([l,v])=>`<div class="card"><div class="value">${esc(v)}</div><div class="label">${esc(l)}</div></div>`).join('');
  document.getElementById('candidates').innerHTML = (refs.candidates||[]).slice(0,30).map(c=>`<div class="item"><strong>${esc(c.title)}</strong><div class="meta"><span class="pill ${esc(c.severity)}">${esc(c.severity)}</span> Priority ${esc(c.priority)} · Risk ${esc(c.risk)} · ${esc(c.source)}</div><p>${esc(c.recommendation)}</p></div>`).join('') || '<p>Keine Kandidaten.</p>';
  document.getElementById('hotspots').innerHTML = (health.risk_hotspots||[]).slice(0,30).map(h=>`<div class="item"><strong>${esc(h.relative_path)}</strong><div class="meta">Risk ${esc(h.risk_score)} · Complexity ${esc(h.complexity_score)} · Lines ${esc(h.lines)}</div><pre>${esc((h.issues||[]).join('\n') || 'Keine Issues')}</pre></div>`).join('') || '<p>Keine Hotspots.</p>';
}
document.getElementById('refresh').addEventListener('click', load);
load().catch(e=>{document.body.insertAdjacentHTML('beforeend', `<pre>${esc(e.stack||e)}</pre>`)});
