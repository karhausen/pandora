async function getJson(url){const r=await fetch(url);return await r.json();}
function pretty(x){return JSON.stringify(x,null,2)}
async function loadPattern(){
  try{
    const [status,health,stats,result]=await Promise.all([getJson('/api/pattern/status'),getJson('/api/pattern/health'),getJson('/api/pattern/statistics'),getJson('/api/pattern/detect?limit=500')]);
    document.getElementById('statusPanel').innerHTML=`<strong>${status.kind}</strong> · Version ${status.version} · ${status.detectors.length} Detektoren · Proposals: ${status.creates_proposals ? 'ja' : 'nein'}`;
    document.getElementById('healthBox').textContent=pretty(health);
    document.getElementById('statsBox').textContent=pretty(stats);
    const box=document.getElementById('patternsBox'); box.innerHTML='';
    if(!result.patterns.length){box.innerHTML='<p class="muted">Noch keine Muster erkannt. Dafür braucht Pandora genügend Observation Events.</p>';return;}
    for(const p of result.patterns){
      const div=document.createElement('div'); div.className='pattern';
      div.innerHTML=`<strong>${p.title}</strong><br><span class="tag">${p.pattern_type}</span><span class="tag">Confidence ${p.confidence}</span><span class="tag">Trend ${p.trend}</span><span class="tag">${p.severity}</span><p>${p.description}</p><pre>${pretty(p.evidence)}</pre><p class="muted">${p.recommendation_hint||''}</p>`;
      box.appendChild(div);
    }
  }catch(e){document.getElementById('statusPanel').textContent=`Pattern Recognition konnte nicht geladen werden: ${e}`;}
}
loadPattern();
