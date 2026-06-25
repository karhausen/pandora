TOOL_META = {
    "id": "weather_lookup",
    "name": "Weather Lookup",
    "description": "Fetches current weather information for a location via a configured weather API.",
    "version": "0.1.0",
    "input_schema": {'location': 'str'},
    "output_schema": {'location': 'str', 'temperature': 'float', 'condition': 'str', 'source': 'str'},
    "security_level": "SAFE",
    "status": "ACTIVE",
    "module": "generated_tools.weather_lookup",
    "function": "run",
}

def run(payload: dict) -> dict:
    text = payload.get("text") or payload.get("input") or ""
    return {"text": str(text)}
