let currentSessionId = localStorage.getItem("pandora_session_id") || null;
let activeChatRoute = null;
let selectedProposalId = null;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  try {
    const data = JSON.parse(text);
    if (!response.ok) {
      return { success: false, error: data.detail || data.error || text };
    }
    return data;
  } catch {
    return { success: response.ok, answer: text };
  }
}

function addMessage(role, text) {
  const chat = document.getElementById("chat");
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.innerHTML = `<div class="role">${role === "user" ? "Du" : "Pandora"}</div><div class="answer"></div>`;
  item.querySelector(".answer").textContent = text;
  chat.prepend(item);
  chat.scrollTop = 0;
}

function clearChat() {
  document.getElementById("chat").innerHTML = "";
}

function setBusy(isBusy) {
  document.getElementById("runButton").disabled = isBusy;
  document.getElementById("statusText").textContent = isBusy ? "Pandora arbeitet..." : "Bereit";
}

function normalizeCoordinatorDetails(result) {
  return {
    route: result.route || result?.decision?.route || null,
    reason: result?.decision?.reason || null,
    confidence: result?.decision?.confidence ?? null,
    provider_name: result?.decision?.provider_name || null,
    model: result?.decision?.model || null,
    session_id: result.session_id || null,
    success: result.success ?? null,
    error: result.error || null
  };
}

function showDetails(result) {
  const details = document.getElementById("details");
  if (details) details.classList.remove("hidden");

  const decisionBox = document.getElementById("decisionBox");
  const planBox = document.getElementById("planBox");
  const executionBox = document.getElementById("executionBox");

  if (decisionBox) {
    decisionBox.textContent = JSON.stringify(normalizeCoordinatorDetails(result), null, 2);
  }

  if (planBox) {
    planBox.textContent = JSON.stringify(result.plan || {}, null, 2);
  }

  if (executionBox) {
    executionBox.textContent = JSON.stringify(result.execution || {}, null, 2);
  }
}

async function ensureSession() {
  if (currentSessionId) return currentSessionId;
  const created = await api("/chat/sessions", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({title: "Neue Unterhaltung"})
  });
  currentSessionId = created.session_id;
  localStorage.setItem("pandora_session_id", currentSessionId);
  await loadSessions();
  return currentSessionId;
}

async function newSession() {
  const created = await api("/chat/sessions", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({title: "Neue Unterhaltung"})
  });
  currentSessionId = created.session_id;
  localStorage.setItem("pandora_session_id", currentSessionId);
  clearChat();
  document.getElementById("details").classList.add("hidden");
  await loadSessions();
}

async function loadSessions() {
  const data = await api("/chat/sessions");
  const select = document.getElementById("sessionSelect");
  if (!select || !data.sessions) return;

  select.innerHTML = "";
  for (const session of data.sessions) {
    const option = document.createElement("option");
    option.value = session.session_id;
    option.textContent = `${session.title || "Unterhaltung"} (${session.message_count})`;
    if (session.session_id === currentSessionId) option.selected = true;
    select.appendChild(option);
  }
}

async function switchSession() {
  const select = document.getElementById("sessionSelect");
  currentSessionId = select.value;
  localStorage.setItem("pandora_session_id", currentSessionId);
  await loadCurrentSession();
}

async function loadCurrentSession() {
  if (!currentSessionId) return;
  const session = await api(`/chat/sessions/${currentSessionId}`);

  if (session.error || session.success === false || !session.session_id) {
    localStorage.removeItem("pandora_session_id");
    currentSessionId = null;
    clearChat();
    await loadSessions();
    return;
  }

  clearChat();
  const messages = [...(session.messages || [])].reverse();
  for (const message of messages) {
    addMessage(message.role, message.content);
  }
}

function renderActiveChatRoute(status) {
  activeChatRoute = status.active_chat_route || null;
  const providerBox = document.getElementById("activeChatProvider");
  const modelBox = document.getElementById("activeChatModel");
  if (!providerBox || !modelBox) return;

  if (!activeChatRoute) {
    providerBox.textContent = "Routing unbekannt";
    modelBox.textContent = "Bitte im LLM & Profile Center prüfen";
    return;
  }

  providerBox.textContent = activeChatRoute.provider_name || "unbekannt";
  const model = activeChatRoute.model || "kein Modell gesetzt";
  const source = activeChatRoute.resolved_from || "Routing";
  modelBox.textContent = `${model} · ${source}`;
}


function proposalStatusClass(status) {
  const normalized = String(status || "").toLowerCase();
  if (["validated", "approved", "installed"].includes(normalized)) return `status ${normalized}`;
  if (["failed", "rejected"].includes(normalized)) return `status danger`;
  return "status";
}

function extractProposalId(result) {
  return result?.execution?.proposal_id
    || result?.execution?.tool_development?.proposal?.id
    || result?.execution?.tool_development?.proposal_id
    || result?.proposal_id
    || null;
}

function openWorkflow(message) {
  const workflow = document.getElementById("toolWorkflow");
  const hint = document.getElementById("proposalHint");
  if (workflow) workflow.classList.remove("hidden");
  if (hint && message) hint.textContent = message;
}

async function loadProposals(selectId = selectedProposalId) {
  const workflow = document.getElementById("toolWorkflow");
  if (workflow) workflow.classList.remove("hidden");

  const data = await api("/tool-proposals");
  const proposals = data.tool_proposals || [];
  const list = document.getElementById("proposalList");
  if (!list) return;

  list.innerHTML = "";
  if (!proposals.length) {
    list.innerHTML = '<div class="empty">Keine Proposals vorhanden.</div>';
    return;
  }

  for (const proposal of proposals) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `proposal-item ${proposal.id === selectId ? "active" : ""}`;
    item.innerHTML = `
      <span class="proposal-title">${proposal.capability || proposal.id}</span>
      <span class="proposal-id">${proposal.id}</span>
      <span class="${proposalStatusClass(proposal.status)}">${proposal.status}</span>
    `;
    item.addEventListener("click", () => showProposal(proposal.id));
    list.appendChild(item);
  }

  if (selectId) {
    await showProposal(selectId, false);
  }
}

function updateWorkflowButtons(proposal) {
  const approve = document.getElementById("approveButton");
  const install = document.getElementById("installButton");
  const reject = document.getElementById("rejectButton");
  const status = proposal?.status;

  if (approve) approve.disabled = status !== "VALIDATED";
  if (install) install.disabled = status !== "APPROVED";
  if (reject) reject.disabled = status === "INSTALLED" || !status;
}

async function showProposal(proposalId, refreshList = true) {
  selectedProposalId = proposalId;
  const data = await api(`/tool-proposals/${proposalId}`);
  const proposal = data.proposal || data;

  const summary = document.getElementById("proposalSummary");
  if (summary) {
    summary.innerHTML = `
      <div><strong>${proposal.capability || proposal.id}</strong></div>
      <div>ID: ${proposal.id}</div>
      <div>Status: <span class="${proposalStatusClass(proposal.status)}">${proposal.status}</span></div>
      <div>Risiko: ${proposal.risk || "unbekannt"}</div>
    `;
  }

  const box = document.getElementById("proposalBox");
  if (box) box.textContent = JSON.stringify(proposal, null, 2);
  updateWorkflowButtons(proposal);

  if (refreshList) await loadProposals(proposalId);
}

async function approveSelectedProposal() {
  if (!selectedProposalId) return;
  const result = await api(`/tool-proposals/${selectedProposalId}/approve`, { method: "POST" });
  document.getElementById("proposalHint").textContent = result.success ? "Proposal approved." : `Approve fehlgeschlagen: ${result.error || "unbekannt"}`;
  await loadProposals(selectedProposalId);
}

async function rejectSelectedProposal() {
  if (!selectedProposalId) return;
  const result = await api(`/tool-proposals/${selectedProposalId}/reject`, { method: "POST" });
  document.getElementById("proposalHint").textContent = result.success ? "Proposal rejected." : `Reject fehlgeschlagen: ${result.error || "unbekannt"}`;
  await loadProposals(selectedProposalId);
}

async function installSelectedProposal() {
  if (!selectedProposalId) return;
  const result = await api(`/tool-proposals/${selectedProposalId}/install`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({})
  });
  document.getElementById("proposalHint").textContent = result.activated
    ? `Tool installiert: ${result.tool_id}`
    : `Installation fehlgeschlagen: ${result.error || "unbekannt"}`;
  await loadProposals(selectedProposalId);
}

async function runPandora() {
  const input = document.getElementById("taskInput");
  const task = input.value.trim();
  if (!task) return;

  await ensureSession();
  addMessage("user", task);
  setBusy(true);

  const result = await api("/coordinator/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      task,
      session_id: currentSessionId,
      save: true
    })
  });

  if (result.success) {
    currentSessionId = result.session_id;
    localStorage.setItem("pandora_session_id", currentSessionId);
    addMessage("assistant", result.answer || "Erledigt.");
    showDetails(result);
    const proposalId = extractProposalId(result);
    if (proposalId) {
      selectedProposalId = proposalId;
      openWorkflow(`Neuer Tool-Vorschlag erkannt: ${proposalId}`);
      await loadProposals(proposalId);
    }
    await loadSessions();
  } else {
    addMessage("assistant", `Fehler: ${result.error || "Unbekannter Fehler"}`);
    showDetails(result);
  }

  setBusy(false);
}

async function loadUserStatus() {
  const status = await api("/user/status");
  renderActiveChatRoute(status);
  document.getElementById("statusText").textContent = status.ready ? "Bereit" : "Nicht bereit";
}

async function boot() {
  await loadUserStatus();
  await loadSessions();
  if (currentSessionId) {
    await loadCurrentSession();
  }
}

boot();
