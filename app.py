import os

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

from agent import run_agent

load_dotenv()

MODEL = "openai/gpt-oss-20b"


def make_chat_fn(client):
    def chat(message, history):
        steps = []

        def on_tool_call(name, arguments, result):
            args_str = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
            steps.append(f"🔧 `{name}({args_str})` → {result}")

        messages = [*history, {"role": "user", "content": message}]
        answer = run_agent(client, MODEL, messages, on_tool_call=on_tool_call)

        if steps:
            trace = "\n".join(steps)
            return f"{trace}\n\n{answer}"
        return answer

    return chat


def main():
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    demo = gr.ChatInterface(
        fn=make_chat_fn(client),
        type="messages",
        title="Tool-Use Agent",
        description=(
            "An agent that can call tools when it needs live data or exact "
            "computation: a calculator, current weather, Wikipedia search, "
            "and current time by timezone. Tool calls are shown above each "
            "answer so you can see the agent's reasoning steps."
        ),
        examples=[
            "What's 23 times (47 + 8), divided by 3?",
            "What's the weather like in La Serena right now?",
            "What is Astro (the web framework)? Also, what time is it in Chile?",
        ],
    )
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
