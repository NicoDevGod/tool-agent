"""
The agent loop: the core mechanic behind every "AI agent" — an LLM that can
decide to call functions, see their results, and decide again (possibly
calling more tools) until it has enough information to answer in plain text.

This talks to Groq's chat completions API directly (no LangChain), so the
tool-calling protocol itself is visible instead of hidden behind a framework.
"""

import json

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools: a calculator, current "
    "weather lookup, Wikipedia search, and current time by timezone. Use a "
    "tool whenever the question needs live data or precise computation instead "
    "of guessing. Answer directly, without tools, for anything else. "
    "If a tool call doesn't return useful information, do not retry it with "
    "slightly reworded arguments — answer from your own knowledge instead, "
    "and say the lookup didn't find anything if that's relevant."
)

MAX_STEPS = 5


def run_agent(client, model, history, on_tool_call=None):
    """Runs the agent loop for one user turn.

    `history` is the full list of {"role", "content"} messages so far,
    including the latest user message. Returns the final assistant text.
    `on_tool_call(name, arguments, result)` is called once per tool
    invocation, useful for showing the agent's steps in a UI.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content

        # The assistant's tool-call request has to go back into the
        # conversation before the tool results, or the API rejects the next
        # message as out of order. Built manually (not message.model_dump())
        # because the response object carries extra fields (e.g.
        # "annotations") that Groq's API rejects when echoed back as input.
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            function = TOOL_FUNCTIONS.get(name)
            result = function(**arguments) if function else f"Unknown tool: {name}"

            if on_tool_call:
                on_tool_call(name, arguments, result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    return "I couldn't finish reasoning about this within the allowed number of steps."
