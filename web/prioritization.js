async function getJson(url){const r=await fetch(url);return await r.json();}
function pretty(x){return JSON.stringify(x,null,2)}
async function loadPriority(){
  try{
    const [status,health,weights,result]=await Promise.all([getJson('/api/prioritization/status'),getJson('/api/prioritization/health'),getJson('/api/prioritization/weights'),getJson('/api/prioritization/prioritize?limit=500')]);
    document.getElementById('statusPanel').innerHTML=`<strong>${status.kind}</strong> · Version ${status.version} · Proposals: ${status.creates_proposals?'ja':'nein'} · Änderungen: ${status.activates_changes?'ja':'nein'}`;
    document.getElementById('healthBox').textContent=pretty(health);
    document.getElementById('weightsBox').textContent=pretty(weights.weights);
    const box=document.getElementById('queueBox'); box.innerHTML='';
    if(!result.queue.length){box.innerHTML='<p class="muted">Noch keine Kandidaten. Dafür braucht Pandora gespeicherte Observation Events und erkennbare Patterns.</p>';return;}
    for(const c of result.queue){const div=document.createElement('div'); div.className='candidate'; div.innerHTML=`<strong>${c.title}</strong><br><span class="tag">${c.candidate_type}</span><span class="tag">${c.score.level}</span><span class="tag">Score ${Number(c.score.total_score).toFixed(1)}</span><p>${c.description}</p><p class="muted">${c.score.explanation}</p><pre>${pretty(c.score.factors)}</pre>`; box.appendChild(div);}
  }catch(e){document.getElementById('statusPanel').textContent=`Prioritization konnte nicht geladen werden: ${e}`;}
}
loadPriority();
