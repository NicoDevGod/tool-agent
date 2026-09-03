"""
Tool implementations the agent can call, plus their JSON schemas in the format
Groq/OpenAI-compatible chat completion APIs expect for function calling.
"""

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

# Wikipedia's API rejects requests without a descriptive User-Agent (returns
# a 403 "please set a user-agent" error) as part of their robot policy.
_WIKIPEDIA_HEADERS = {"User-Agent": "tool-agent-demo/1.0 (https://github.com/NicoDevGod/tool-agent)"}

# ---------- 1. Calculator ----------

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _eval_node(node):
    """Recursively evaluates an arithmetic AST node, allowing only numbers and
    the operators above. This is deliberately NOT a call to eval()/exec() —
    the LLM constructs this expression, so a raw eval() would be an arbitrary
    code execution vulnerability."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


# ---------- 2. Weather (Open-Meteo, no API key needed) ----------


def get_weather(city: str) -> str:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    ).json()
    results = geo.get("results")
    if not results:
        return f"Could not find a location named '{city}'."

    place = results[0]
    lat, lon = place["latitude"], place["longitude"]

    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current": "temperature_2m,wind_speed_10m,weather_code"},
        timeout=10,
    ).json()
    current = forecast.get("current", {})

    label = f"{place['name']}, {place.get('country', '')}".strip(", ")
    return (
        f"Weather in {label}: {current.get('temperature_2m')}°C, "
        f"wind {current.get('wind_speed_10m')} km/h."
    )


# ---------- 3. Wikipedia search (no API key needed) ----------


def search_wikipedia(query: str) -> str:
    # The summary endpoint needs an exact page title, but the LLM passes a
    # loose query ("Astro web framework", not "Astro (web framework)"), so
    # resolve the best-matching title with the search API first.
    search = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
        headers=_WIKIPEDIA_HEADERS,
        timeout=10,
    ).json()
    results = search.get("query", {}).get("search")
    if not results:
        return f"No Wikipedia article found for '{query}'."
    title = results[0]["title"]

    resp = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
        headers=_WIKIPEDIA_HEADERS,
        timeout=10,
    )
    if resp.status_code != 200:
        return f"No Wikipedia article found for '{query}'."
    extract = resp.json().get("extract")
    return extract or f"No summary available for '{query}'."


# ---------- 4. Current time (local, no API needed) ----------


def get_current_time(timezone: str = "UTC") -> str:
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime(f"%Y-%m-%d %H:%M:%S ({timezone})")
    except Exception:
        return f"Unknown timezone: '{timezone}'. Try e.g. 'America/Santiago' or 'UTC'."


# ---------- Schemas + dispatch table ----------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression (+ - * / % **, parentheses).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '23 * (4 + 5)'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'La Serena'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Get a short summary of a topic from Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to look up, e.g. 'Astro (web framework)'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in a given IANA timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name, e.g. 'America/Santiago'. Defaults to UTC.",
                    },
                },
                "required": [],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "search_wikipedia": search_wikipedia,
    "get_current_time": get_current_time,
}
