TOOL_META = {"id":"uppercase","name":"Uppercase","description":"Converts text to uppercase.","version":"0.1.0","input_schema":{"text":"str"},"output_schema":{"text":"str"},"security_level":"SAFE","status":"ACTIVE","module":"tools.uppercase","function":"run"}
def run(payload: dict) -> dict:
    return {"text": str(payload.get("text", "")).upper()}
