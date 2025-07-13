from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from dotenv import load_dotenv
import os
import chainlit as cl

# Load environment variables (like API key)
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")



# === Gemini API Client (OpenAI-Compatible Wrapper) ===
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/"
)

# === Model ===
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",  # or "gemini-2.0-pro" if you're using latest
    openai_client=client,
)

# === Agent ===
agent = Agent(
    name="Translator Agent",
    instructions=(
        "You are a Translator Agent. You translate Urdu info english and english into urdu"
    ),
    model=model,)

# === Run Config ===
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# === Chainlit Event: On Chat Start ===
@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content=f"Hello! I am {agent.name}. Ask me about Muhammad Uzair.").send()

# === Chainlit Event: On Message ===
@cl.on_message
async def handle_message(message):
    history = cl.user_session.get('history')
    history.append({"role": "user", "content": message.content})
    
    result = await Runner.run(
        agent,
        input=history,
        run_config=config
    )
    
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    
    await cl.Message(content=result.final_output).send()
