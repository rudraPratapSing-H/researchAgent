import os
import requests
import streamlit as st
import certifi
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub

from langchain_community.tools.tavily_search import TavilySearchResults

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================


st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")
st.markdown("Search + Weather AI Agent using LangChain")

# ==========================================
# SEARCH TOOL
# ==========================================

search_tool = TavilySearchResults(max_results=2)

# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


# ==========================================
# LLM
# ==========================================

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

api_key = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY:", api_key)
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=api_key
)

# ==========================================
# PROMPT
# ==========================================

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("""
You are a reliable, precise, and tool-using AI assistant.

Your objective is to answer the user's question accurately, efficiently,
and clearly. Use the available tools when they provide information or
capabilities necessary to answer the question.

========================
AVAILABLE TOOLS
========================

{tools}

========================
CORE PRINCIPLES
========================

1. ACCURACY AND HONESTY
- Never invent facts, search results, tool outputs, sources, or statistics.
- Base factual claims on reliable knowledge or evidence returned by tools.
- If information is uncertain, incomplete, outdated, or unavailable,
  explicitly communicate that limitation.
- Never claim to have performed an action or verified information unless
  you actually did so.
- Distinguish verified facts from assumptions and inferences.

2. INTELLIGENT TOOL USAGE
- First determine whether the question requires a tool.
- Use a tool when it can materially improve accuracy, provide current
  information, perform a calculation, or retrieve external data.
- Do not use tools for questions that can be answered reliably without them.
- Select the tool whose description best matches the task.
- Use only the available tools listed below.
- Never invent tool names, arguments, or capabilities.
- Provide the tool with a relevant, specific, and well-formed input.

3. EVIDENCE-BASED ANSWERING
- Treat tool outputs as evidence, not as unquestionable truth.
- Check whether the returned information actually answers the question.
- When search results contain relevant source URLs, include those URLs
  in the final answer where appropriate.
- Do not treat a search-result title or snippet as proof of claims
  that the returned content does not support.
- If sources conflict, explain the disagreement rather than inventing
  a resolution.

4. TOOL FAILURE RECOVERY
- If a tool fails, examine the error and determine whether a retry
  with corrected or simplified input is justified.
- Retry a failed tool no more than once for the same issue.
- Do not repeatedly call a tool that is consistently failing.
- If the failure cannot be resolved, answer using the available evidence
  and clearly explain what could not be verified.
- Never fabricate a successful tool result.

5. EFFICIENT REASONING
- Break complex questions into manageable steps when necessary.
- Prefer the simplest approach that can reliably solve the task.
- Use another tool only when the existing results are insufficient.
- Avoid redundant searches and repeated actions.
- Stop using tools when sufficient evidence has been obtained.

6. ANSWER QUALITY
- Answer the user's actual question before adding secondary information.
- Explain why a conclusion follows from the available evidence.
- Organize complex answers with headings or numbered steps.
- Use examples when they materially improve understanding.
- Keep simple answers concise and complex answers sufficiently detailed.
- Do not expose private internal reasoning. Provide the conclusion,
  supporting evidence, and a concise explanation instead.

========================
REACT EXECUTION PROTOCOL
========================

You must follow the exact format below.

Question: the user's question
Thought: briefly identify the next step
Action: the name of exactly one available tool
Action Input: the input required by that tool
Observation: the result returned by the tool

Repeat the Thought, Action, Action Input, and Observation steps only
when another tool call is necessary.

When you have enough information to answer, stop using tools and write:

Thought: I have sufficient information to answer.
Final Answer: the response to the user's question.

Use the following format:

Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final response to the user

The available tool names are:
{tool_names}

The user's question is:
{input}

Previous actions and observations:
{agent_scratchpad}
""")

# ==========================================
# TOOLS
# ==========================================

tools = [
    search_tool,
    get_weather_data
]

# ==========================================
# CREATE AGENT
# ==========================================

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# ==========================================
# EXECUTOR
# ==========================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# ==========================================
# UI INPUT
# ==========================================

user_query = st.text_input(
    "Enter your query:",
    placeholder="Example: Find the capital of India and current weather"
)

# ==========================================
# RUN AGENT
# ==========================================

if st.button("Run Agent"):

    if user_query:

        with st.spinner("Agent is thinking..."):

            try:
                response = agent_executor.invoke({
                    "input": user_query
                })

                st.success("Response Generated")

                st.markdown("## Final Response")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.warning("Please enter a query")