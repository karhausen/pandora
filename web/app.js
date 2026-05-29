async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  try { return JSON.parse(text); } catch { return text; }
}
function show(id, data) {
  document.getElementById(id).textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}
async function loadStatus() {
  const data = await api("/status");
  document.getElementById("statusBadge").textContent = data.status ? `${data.status} · ${data.version}` : "Status unbekannt";
}
async function loadHeartbeat(){ show("heartbeatBox", await api("/heartbeat")); }
async function loadTools(){ show("toolsBox", await api("/tools")); }
async function loadSkills(){ show("skillsBox", await api("/skills")); }
async function loadJournal(){ show("journalBox", await api("/agent/journal")); }
async function loadToolProposals(){ show("toolProposalsBox", await api("/tool-proposals")); }
async function loadSkillProposals(){ show("skillProposalsBox", await api("/skill-proposals")); }
async function loadCapabilityWorkflows(){ show("capabilityWorkflowsBox", await api("/capabilities/workflows")); }
async function runAgent() {
  const task = document.getElementById("taskInput").value;
  const provider_name = document.getElementById("providerSelect").value;
  show("agentResult", {running:true, task, provider_name});
  const result = await api("/agent/run", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({task, provider_name})
  });
  show("agentResult", result);
  await loadJournal();
}
async function runLearning(){
  show("learningBox", await api("/learning/run", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({limit:200})}));
}
async function loadRecommendations(){ show("learningBox", await api("/learning/recommendations")); }
async function loadRankings(){ show("learningBox", await api("/learning/rankings")); }
async function loadGovernance(){ show("governanceBox", await api("/governance/check")); }
async function loadChangelog(){ show("governanceBox", await api("/changelog")); }
async function boot(){
  await loadStatus();
  await Promise.all([loadHeartbeat(), loadTools(), loadSkills(), loadJournal(), loadToolProposals(), loadSkillProposals(), loadCapabilityWorkflows(), loadRecommendations()]);
}
boot();


async function generateTool() {
  const capability = document.getElementById("toolCapabilityInput").value;
  const provider_name = document.getElementById("toolGenerationProvider").value;
  const run_tests = !document.getElementById("toolGenerationNoTests").checked;

  show("toolGenerationBox", {running: true, capability, provider_name, run_tests});

  const result = await api("/tool-generation/generate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      capability,
      provider_name,
      max_attempts: 2,
      run_tests
    })
  });

  show("toolGenerationBox", result);
  await loadToolProposals();
}

async function loadToolGenerationLogs() {
  show("toolGenerationBox", await api("/tool-generation/logs"));
}


async function loadCoreStatus() {
  show("coreStatusBox", await api("/core/status"));
}

async function runCoreSmoke() {
  show("coreStatusBox", {running: true, action: "core-smoke"});
  show("coreStatusBox", await api("/core/smoke", {method: "POST"}));
}

async function createCoreSnapshot() {
  show("coreStatusBox", {running: true, action: "core-snapshot"});
  show("coreStatusBox", await api("/core/snapshot", {method: "POST"}));
}


async function runRealityCheck() {
  show("realityCheckBox", {running: true, iterations: 3});
  show("realityCheckBox", await api("/reality-check/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({iterations: 3, delay: 0, run_pytest: false})
  }));
}

async function loadRealityReport() {
  show("realityCheckBox", await api("/reality-check/report"));
}

async function loadRealityLogs() {
  show("realityCheckBox", await api("/reality-check/logs"));
}


async function runPlanner() {
  const task = document.getElementById("plannerTaskInput").value;
  show("plannerBox", {running: true, task});
  show("plannerBox", await api("/planner/plan", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({task, provider_name: "mock", save: true})
  }));
}

async function loadPlannerPlans() {
  show("plannerBox", await api("/planner/plans"));
}


async function runPlannerWorker() {
  const task = document.getElementById("workerTaskInput").value;
  show("workerBox", {running: true, task});
  show("workerBox", await api("/planner-worker/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({task, provider_name: "mock", save: true})
  }));
}

async function loadWorkerExecutions() {
  show("workerBox", await api("/worker/executions"));
}
