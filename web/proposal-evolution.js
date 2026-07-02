async function loadJson(url, options) {
  const res = await fetch(url, options);
  return await res.json();
}
async function refresh() {
  document.getElementById('status').textContent = JSON.stringify(await loadJson('/api/proposal-evolution/status'), null, 2);
  document.getElementById('history').textContent = JSON.stringify(await loadJson('/api/proposal-evolution/history'), null, 2);
}
document.getElementById('refresh').addEventListener('click', refresh);
refresh();
