console.log('Pandora GUI loaded'); async function loadStatus(){}

async function loadRecommendations() {
    const data = await api("/learning/recommendations");
    console.log("Recommendations", data);
}
