async function getJson(url){const r=await fetch(url);return await r.json();}
function pretty(x){return JSON.stringify(x,null,2)}
async function loadObservation(){
  try{
    const [status,health,stats,events]=await Promise.all([getJson('/api/observation/status'),getJson('/api/observation/health'),getJson('/api/observation/statistics'),getJson('/api/observation/events?limit=25')]);
    document.getElementById('statusPanel').innerHTML=`<strong>${status.kind}</strong> · Version ${status.version} · ${status.components.length} Komponenten · Proposals: ${status.creates_proposals ? 'ja' : 'nein'}`;
    document.getElementById('healthBox').textContent=pretty(health);
    document.getElementById('statsBox').textContent=pretty(stats);
    const box=document.getElementById('eventsBox'); box.innerHTML='';
    if(!events.events.length){box.innerHTML='<p class="muted">Noch keine Events gespeichert.</p>';return;}
    for(const ev of events.events){const div=document.createElement('div');div.className='event';div.innerHTML=`<strong>${ev.component}</strong> · ${ev.event_type} · ${ev.success?'OK':'Fehler'}<br><span class="muted">${ev.timestamp}</span><br>${ev.message||''}`;box.appendChild(div);}
  }catch(e){document.getElementById('statusPanel').textContent=`Observation konnte nicht geladen werden: ${e}`;}
}
loadObservation();
