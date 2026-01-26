import asyncio
from typing import Sequence
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination, HandoffTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage, ToolCallExecutionEvent
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.base import Handoff
from autogen_ext.models.openai import OpenAIChatCompletionClient
from constant import init_model
from scientist_team import create_scientist_team
from engineer_team import create_engineer_team
from robot_platform import create_robot_team
from analyst_team import create_analyst_team


user = UserProxyAgent("User", input_func=input)
scientist_team = create_scientist_team(input)
engineer_team = create_engineer_team(input)
robot_platform = create_robot_team(input)
analyst_team = create_analyst_team(input)

result = {}
planning_agent = AssistantAgent(
    "ORCHESTRATOR",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=init_model('gpt-4o'),
    system_message="""
    You are a planning agent.
    Your job is to break down complex Materials science research tasks into smaller, manageable subtasks.
    Assign these subtasks to the appropriate sub-teams; not all sub-teams are required to participate in every task.
    Your sub-teams are:
        1. User: A human agent to whom you transfer information whenever you need to confirm your execution steps to a human.
        2. Scientist: A professional team of material scientists who are mainly responsible for consulting on material synthesis, structure, application and properties.
            - The scientist team has the following members: 
            2.1 Synthesis Scientist: who is good at giving perfect and correct synthesis solutions.
            2.2 Structure Scientist: focusing on agents of structural topics in materials science.
            2.3 Property Scientist: focuses on physical and chemistry property topics in materials science.
            2.4 Application Scientist: Focus on practical applications of materials, such as devices, chips, etc.
        3. Engineer: A team of professional engineers who are responsible for writing code, visualizing experimental schemes, converting experimental schemes to JSON, and more.
            - The engineer team has the following members: 
            3.1 Structural engineer: A professional structural engineer who focus on converting natural language synthesis schemes to JSON or XML formated scheme, and then upload this JSON to S3 Storage.
            3.2 Software engineer: A professional software engineers will coding with Python.
            3.3 Code reviewer: A professional code reviewer will review the code written by software engineers and execute it.
            3.4 Scheme Plotter: An agent responsible for converting a expriment scheme into a Mermaid flowchart.
        4. Executor: A robotic platform is responsible for performing automated synthesis experiments, automated characterization experiments, and collecting experimental datas.
            - The Executor team has the following members: 
            4.1 MobileRobot_Agent: This agent controls the mobile robot by calling the funciton sendScheme2MobileRobot to place the experimental container into the robot workstation. This agent called before RobotWorkstation_Agent.
            4.2 RobotWorkstation_Agent: This agent is called by the mobile robot agent, do not plan it alone.
            4.3 DataCollector_Agent: This agent collects experimental data and experimental logs from the characterization device in the robot platform and stores them.
        5. Analyst: A team of data analysts who are responsible for analyzing and visualizing experimental data and logs.
            - The Data Analysis team has the following members:
            5.1 Expriment_Analyst: The agent of data analysts who are responsible for analyzing experimental data and logs.
            5.2 Expriment_Optimizer: The agent optimizes the experimental scheme by means of component regulation and so on to make the experimental result close to the desired goal of the user.
            5.3 Data_Visulizer: The agent of data visulizers who are responsible for visualizing experimental data and logs.

    You only plan and delegate tasks - you do not execute them yourself.
    
    当你收到你的子团队返回的任务结果时，你需要评估结果是否符合预期，并决定是否需要该子团队重新执行任务或者执行下一个子任务或是否可以结束任务。

    回答时你需要初始化/更新如下任务分配表和Mermaid流程图，并按顺序执行，使用如下格式并利用：
    | Team_name   | Member_name   | sub-task                             |
    | ----------- | ------------- | ------------------------------------ |
    | <team_name> | <member_name> | <status: brief sub-task description> |
    
    ```mermaid
    graph TD
    User[User]
    subgraph <team_name>
        A1[<member_name>]
    end
    style xxx # 推荐多样的风格
    ...
    User --> A1
    ...
    ```

    每次回答时，你需要清晰明确的指出已经完成的子任务下一步子任务，使用如下格式：
    **已完成子任务：**
    1. <team> : <subtask>
    **Next sub-task:**
    n. <team> : <subtask>
    
    You can end with "HUMAN" when you needed, which means you need human approval or other advice or instructions;
    for example, (1) when you are not sure about the user's requests or (2) when the scientist team needs to confirm the synthesis scheme with the user or there are some mistakes in scheme;
    or (3) when the are some uncertainties in any team's response or (4) when you need to confirm the termination of the task or (5) the score of RAG is low;
    or (6) when the engineer team need to confirm the code with the user before executing it on a real machine or (7) when the robotic platform need to confirm the experiment scheme with the user before executing it on a real machine.
    or (8) when the analyst team need to confirm the analysis or visualization result with the user and any other situations that require human intervention.
    For security reasons, any of your steps should be confirmed by the user before execution. However, for the sake of efficiency, you should try to minimize the number of times you request confirmation from users. 
    Therefore, you need to comprehensively consider the complexity and uncertainty of the task to decide whether to request user confirmation.

    After plan and delegate tasks are complete, end with "START";
    Determine if all sub-teams have completed their tasks, and if so, summarize the findings and end with "TERMINATE".
    """,
    reflect_on_tool_use=False
)
max_messages_termination = MaxMessageTermination(max_messages=200)
termination = max_messages_termination


# The selector function is a function that takes the current message thread of the group chat
# and returns the next speaker's name. If None is returned, the LLM-based selection method will be used.
def selector_func(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
    if messages[-1].source != planning_agent.name:
        return planning_agent.name # Always return to the planning agent after the other agents have spoken.
    elif "HUMAN" in messages[-1].content or "TERMINATE" in messages[-1].content:
        return user.name
    return None

team = SelectorGroupChat(
    [planning_agent, user, scientist_team, engineer_team, robot_platform, analyst_team],
    model_client=init_model("gpt-4o"), # Use a smaller model for the selector.
    termination_condition=termination,
    selector_func=selector_func,
)


async def main(task: str = "") -> dict:
    await Console(team.run_stream(task=task))
    # await code_executor.stop() 
    # async for message in team.run_stream(task=task):
    #     if isinstance(message, TextMessage):
    #         print(f"----------------{message.source}----------------\n {message.content}")
    #         result[message.source] = message.content
    #     elif isinstance(message, ToolCallExecutionEvent):
    #         print(f"----------------{message.source}----------------\n {message.content}")
    #         result[message.source] = [content.content for content in message.content]
    return result


# Example usage in another function
async def main_1():
    # result = await main(input("Enter your instructions below: \n"))
    result = await main("How to synthesize CsPbBr3 nanocrystals at room temperature?")
    print(result)

if __name__ == "__main__":
    asyncio.run(main_1())
