from generated_tools.weather_lookup import run

def test_weather_lookup():
    result = run({"text": "hello"})
    assert isinstance(result, dict)
