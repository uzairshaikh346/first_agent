from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from dotenv import load_dotenv
import os
import chainlit as cl

# Load environment variables (like API key)
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# === Tool Function ===
@function_tool
def getInfoOfUzair(Name: str) -> str:
    """Provides professional details of Muhammad Uzair."""
    if Name.lower() != "muhammad uzair":
        return "Sorry, I only provide info about Muhammad Uzair."
    
    return (
        "Muhammad Uzair is a skilled Full Stack Developer from Karachi. "
        "He has experience with Next.js, React, Tailwind CSS, AI integration, and cloud-native development. "
        "LinkedIn: https://www.linkedin.com/in/muhammad-uzair-2526732a6/"
    )

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
    name="Uzair Info Provider",
    instructions=(
        "You are an assistant that provides accurate and professional details about Muhammad Uzair. "
        "Use the provided tools to help answer user queries."
    ),
    model=model,
    tools=[getInfoOfUzair],
)

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
