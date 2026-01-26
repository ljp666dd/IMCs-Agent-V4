import asyncio
from typing import Sequence
from autogen_core import CancellationToken
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination, HandoffTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage, ToolCallExecutionEvent
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat, Swarm
from autogen_agentchat.ui import Console
from autogen_agentchat.base import Handoff
from autogen_ext.models.openai import OpenAIChatCompletionClient
from backend.constant import MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, init_model
from backend.tools import hybird_retrieval_from_knowledge_base


async def _single_agent_answer_with_rag(user_query:str, model: str = MODEL):
    model_client = init_model(model)
    try:
        assistant = AssistantAgent(
            name="assistant",
            system_message="""You are a helpful assistant. You can call tools to help user.""",
            model_client=model_client,
            tools=[hybird_retrieval_from_knowledge_base],
            reflect_on_tool_use=True, # Set to True to have the model reflect on the tool use, set to False to return the tool call result directly.
        )

        response = await assistant.on_messages([TextMessage(content=user_query, source="user")], CancellationToken())
        return response.chat_message.content
    except:
        return "Sorry, I am not able to answer your question."
    # print("Assistant:", response.chat_message.content)


async def _single_agent_answer_with_rag_cot(user_query:str, model: str = MODEL):
    model_client = init_model(model)

    assistant = AssistantAgent(
        name="assistant",
        system_message="""You are a helpful assistant. You can call tools to help user. Using chain of thought (CoT) when answering questions.""",
        model_client=model_client,
        tools=[hybird_retrieval_from_knowledge_base],
        reflect_on_tool_use=True, # Set to True to have the model reflect on the tool use, set to False to return the tool call result directly.
    )
    response = await assistant.on_messages([TextMessage(content=user_query + "\nLet's think step by step:", source="user")], CancellationToken())

    if model == "deepseek-r1":
        rag_assistant = AssistantAgent(
            name="rag_assistant",
            system_message="""You are a helpful assistant. You can call tools to help user. Using chain of thought (CoT) when answering questions.""",
            model_client=init_model('gpt-4o'),
            tools=[hybird_retrieval_from_knowledge_base],
            reflect_on_tool_use=False, # Set to True to have the model reflect on the tool use, set to False to return the tool call result directly.
        )
        rag_res = await rag_assistant.on_messages([TextMessage(content=user_query + "\nLet's think step by step:", source="user")], CancellationToken())
        response = await assistant.on_messages([TextMessage(content=rag_res.chat_message.content + "Following the above information to answer this question:" + user_query + "\nLet's think step by step:", source="user")], CancellationToken())

    return response.chat_message.content
    # print("Assistant:", response.chat_message.content)


async def main(model: str = MODEL):
    model_client = OpenAIChatCompletionClient(
        model=model,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    )

    assistant = AssistantAgent(
        name="assistant",
        system_message="""You are a helpful assistant. You can call tools to help user.""",
        model_client=model_client,
        tools=[hybird_retrieval_from_knowledge_base],
        reflect_on_tool_use=True, # Set to True to have the model reflect on the tool use, set to False to return the tool call result directly.
    )

    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        response = await assistant.on_messages([TextMessage(content=user_input, source="user")], CancellationToken())
        print("Assistant:", response.chat_message.content)


if __name__ == "__main__":
    # asyncio.run(main())

    answer = asyncio.run(_single_agent_answer_with_rag("how to synthesis CsPbBr3 nanocubes at room temperature?", model="gpt-4o"))
    # answer = single_agent_answer_with_rag("how to synthesis CsPbBr3 nanocubes at room temperature?", model="gpt-4o")
    print(answer)
    pass
