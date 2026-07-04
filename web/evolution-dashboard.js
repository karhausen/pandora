async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return await res.json();
}
function el(tag, cls, text) { const n=document.createElement(tag); if(cls) n.className=cls; if(text!==undefined) n.textContent=text; return n; }
function pick(obj, path, fallback='—') { try { return path.split('.').reduce((a,k)=>a && a[k], obj) ?? fallback; } catch { return fallback; } }
async function load() {
  const cards = document.getElementById('cards');
  const modules = document.getElementById('modules');
  const queue = document.getElementById('queue');
  const learning = document.getElementById('learning');
  const timeline = document.getElementById('timeline');
  cards.innerHTML = modules.innerHTML = timeline.innerHTML = '';
  const [status, summary, timelineData] = await Promise.all([
    getJson('/api/evolution-dashboard/status'),
    getJson('/api/evolution-dashboard/summary'),
    getJson('/api/evolution-dashboard/timeline?limit=25')
  ]);
  const cardData = [
    ['Health', `${status.overall_health_score}%`],
    ['Module OK', `${status.modules_ok}/${status.modules_total}`],
    ['Queue', pick(summary, 'proposal_queue.data.total', JSON.stringify(pick(summary, 'proposal_queue.data.stats', {})))],
    ['Learning', pick(summary, 'decision_learning.data.stats.total_decisions', 0)]
  ];
  for (const [label, value] of cardData) { const c=el('article','card'); c.append(el('div','label',label), el('div','value',String(value))); cards.append(c); }
  for (const m of status.modules || []) { const d=el('div', `module ${m.ok?'ok':'fail'}`); d.append(el('strong','',m.name), el('div','',`Status: ${m.ok?'OK':'Fehler'}`), el('small','',m.mode || m.kind || '')); modules.append(d); }
  queue.textContent = JSON.stringify(summary.proposal_queue?.data || summary.proposal_queue, null, 2);
  learning.textContent = JSON.stringify(summary.decision_learning?.data || summary.decision_learning, null, 2);
  for (const ev of timelineData.events || []) { const item=el('div','event'); item.append(el('small','',`${ev.timestamp || ''} · ${ev.source || ''}`), el('strong','',ev.title || 'Evolution Event'), el('div','', ev.status ? `Status: ${ev.status}` : '')); timeline.append(item); }
}
document.getElementById('refresh').addEventListener('click', () => load().catch(err => alert(err.message)));
load().catch(err => { document.getElementById('cards').textContent = err.message; });
