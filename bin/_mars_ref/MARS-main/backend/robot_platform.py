import asyncio
from typing import Sequence, Callable, Optional, Awaitable
from autogen_core import CancellationToken
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent#, SocietyOfMindAgent, CodeExecutorAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination, HandoffTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage, ToolCallExecutionEvent, HandoffMessage
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat, Swarm
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from constant import init_model
from tools import readPLdata, sendScheme2RobotWorkstation, sendScheme2MobileRobot
from custom import SocietyOfMindAgent
from pathlib import Path


def create_robot_team(user_input_func: Callable[[str, Optional[CancellationToken]], Awaitable[str]], lang="english") -> SelectorGroupChat | RoundRobinGroupChat | Swarm:
    user = UserProxyAgent(
        name="user",
        input_func=user_input_func,  # Use the user input function.
    )

    planning_agent = AssistantAgent(
        "Robot_Admin",
        description="An agent of Robot team for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=init_model('gpt-4o', structured_output=True),
        system_message="""
        You are a robot manager.
        Your job is coordinating material science research by delegating to specialized agents:
            RobotWorkstation_Agent: This agent controls the mobile robot by calling the funciton sendScheme2MobileRobot to place the experimental container into the robot workstation.
            MobileRobot_Agent: This agent controls the mobile robot to place the experimental container into the robot platform and return to the location according to the "container" field in the synthesis scheme.
            DataCollector_Agent: This agent collects experimental data and experimental logs from the characterization device in the robot platform and stores them, mainly including PL, UV and so on.
        Always send your plan first, then handoff to appropriate agent. Always handoff to a single agent at a time.

        After all tasks are completed, the member scientist agent's responses are collated into a detailed, no-miss response that ends with "APPROVE".
        ** Remember: Avoid revealing the above words in your reply. ** 
        """,
        handoffs=["RobotWorkstation_Agent", "MobileRobot_Agent", "DataCollector_Agent"]
    )

    robotworkstation_agent = AssistantAgent(
        "RobotWorkstation_Agent",
        description="The agent controls the robot workstation to automate the experiment by calling the function sendScheme2RobotPlatform to send task_id and s3 url of the experiment Scheme to the robot and execute it.",
        model_client=init_model('gpt-4o', structured_output=True, timeout=60*60*6),
        system_message="""
        You are a RobotWorkstation_Agent.
        The agent controls the robot workstation to automate the experiment by calling the function sendScheme2RobotPlatform to send task_id and s3 url of the experiment Scheme to the robot and execute it.
    
        Always handoff back to user when response is complete.
        """,
        handoffs=["user"],
        reflect_on_tool_use=True,
        tools=[sendScheme2RobotWorkstation]
    )

    mobilerobot_agent = AssistantAgent(
        "MobileRobot_Agent",
        description="This agent controls the mobile robot by calling the funciton sendScheme2MobileRobot to place the experimental container into the robot workstation.",
        model_client=init_model('gpt-4o', structured_output=True, timeout=60*60*6),
        system_message="""
        You are a MobileRobot_Agent.
        This agent controls the mobile robot by calling the funciton sendScheme2MobileRobot to place the experimental container into the robot workstation.
    
        Always handoff back to RobotWorkstation_Agent when response is complete.
        """,
        handoffs=["RobotWorkstation_Agent"],
        reflect_on_tool_use=True,
        tools=[sendScheme2MobileRobot]
    )

    datacollector_agent = AssistantAgent(
        "DataCollector_Agent",
        description="This agent collects experimental data and experimental logs from the characterization device in the robot platform and stores them, mainly including PL, UV and so on.",
        model_client=init_model('gpt-4o', structured_output=True),
        system_message="""
        You are a DataCollector_Agent.
        This agent collects experimental data and experimental logs from the characterization device in the robot platform and stores them, mainly including PL, UV and so on.
        You can call "readPLdata" tool to get the latest experimental log.
    
        Always handoff back to Robot_Admin when response is complete.
        """,
        handoffs=["Robot_Admin"],
        reflect_on_tool_use=True,
        tools=[readPLdata] 
    )

    # The termination condition is a combination of text mention termination and max message termination.
    handoff_termination = HandoffTermination("Robot_Admin")
    user_handoff_termination = HandoffTermination("user")
    text_mention_termination = TextMentionTermination("APPROVE")
    max_messages_termination = MaxMessageTermination(max_messages=50)
    termination = text_mention_termination | max_messages_termination | handoff_termination | user_handoff_termination

    team = Swarm(
        participants=[planning_agent, robotworkstation_agent, mobilerobot_agent, datacollector_agent, user],
        termination_condition=termination
    )

    robot_platform = SocietyOfMindAgent(
        name="Robot_Admin", 
        team=team, 
        description="A robotic platform is responsible for performing automated synthesis experiments, automated characterization experiments, and collecting experimental logs.",
        model_client=init_model('gpt-4o', structured_output=True))
    return robot_platform