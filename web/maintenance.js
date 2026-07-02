async function loadMaintenance() {
  const statusBox = document.getElementById("maintenanceStatus");
  const grid = document.getElementById("maintenanceGrid");
  try {
    const response = await fetch("/api/gui/user-simplification/status");
    const data = await response.json();
    statusBox.textContent = `${data.mode}: User-Seite bleibt Chat-first, Maintenance ist der einzige Verwaltungs-Einstieg.`;
    grid.innerHTML = "";
    for (const item of data.maintenance_sections || []) {
      const card = document.createElement("a");
      card.className = "maintenance-card";
      card.href = item.href;
      card.innerHTML = `<strong>${item.label}</strong><span>${item.purpose}</span>`;
      grid.appendChild(card);
    }
  } catch (error) {
    statusBox.textContent = `Maintenance-Struktur konnte nicht geladen werden: ${error}`;
  }
}

loadMaintenance();
