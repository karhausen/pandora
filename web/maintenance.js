function riskLabel(risk) {
  const labels = {
    read_only: "Nur anzeigen",
    human_approval: "Freigabe nötig",
    controlled_write: "Schreibt kontrolliert",
    controlled_activation: "Aktivierung",
    configuration: "Konfiguration",
    controlled_run: "Manueller Lauf",
  };
  return labels[risk] || risk || "Status";
}

function renderNavigation(groups) {
  const nav = document.getElementById("maintenanceNav");
  nav.innerHTML = "";
  for (const group of groups) {
    const link = document.createElement("a");
    link.href = `#${group.id}`;
    link.innerHTML = `<strong>${group.title}</strong><span>${group.link_count} Bereiche</span>`;
    nav.appendChild(link);
  }
}

function renderGroups(groups) {
  const target = document.getElementById("maintenanceGroups");
  target.innerHTML = "";
  for (const group of groups) {
    const section = document.createElement("section");
    section.className = "maintenance-group";
    section.id = group.id;
    section.innerHTML = `
      <div class="group-header">
        <div>
          <span class="eyebrow">${group.intent}</span>
          <h2>${group.title}</h2>
          <p>${group.description}</p>
        </div>
      </div>
      <div class="maintenance-grid"></div>
    `;
    const grid = section.querySelector(".maintenance-grid");
    for (const item of group.links || []) {
      const card = document.createElement("a");
      card.className = "maintenance-card";
      card.href = item.href;
      card.innerHTML = `
        <div class="card-topline">
          <strong>${item.label}</strong>
          ${item.badge ? `<span class="badge">${item.badge}</span>` : ""}
        </div>
        <span class="purpose">${item.purpose}</span>
        <span class="risk">${riskLabel(item.risk)}</span>
      `;
      grid.appendChild(card);
    }
    target.appendChild(section);
  }
}

async function loadMaintenanceCenter() {
  const statusBox = document.getElementById("maintenanceStatus");
  try {
    const response = await fetch("/api/gui/maintenance-center/status");
    const data = await response.json();
    statusBox.innerHTML = `<strong>MVP ${data.version}</strong> · ${data.group_count} strukturierte Gruppen · ${data.link_count} Einstiege · Startpunkt: <a href="${data.primary_path}">Operations Cockpit</a>`;
    renderNavigation(data.groups || []);
    renderGroups(data.groups || []);
  } catch (error) {
    statusBox.textContent = `Maintenance Center konnte nicht geladen werden: ${error}`;
  }
}

loadMaintenanceCenter();
