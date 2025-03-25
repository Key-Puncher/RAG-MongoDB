from typing import List
from dotenv import dotenv_values

from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool

config = dotenv_values(".env")

OPENAI_KEY = config["OPENAI_KEY"]


def create_agent(agent_func):
    query_tool = FunctionTool.from_defaults(name="get_nrma_info", fn=agent_func)

    llm = OpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_KEY)

    agent_worker = FunctionCallingAgentWorker.from_tools(
        [query_tool], llm=llm, verbose=True
    )
    agent = AgentRunner(agent_worker)
    return agent
