import json
import logging
import os
from typing import Any, Awaitable, Callable, Optional, Sequence
import uuid
import aiofiles
import yaml
import cv2
import base64
import asyncio
import numpy as np
import time
import subprocess
import ffmpeg
import io
from PIL import Image
from collections import deque

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from aiortc.contrib.media import MediaPlayer

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, UserInputRequestedEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage, ToolCallExecutionEvent
from constant import init_model
from scientist_team import create_scientist_team
from engineer_team import create_engineer_team
from robot_platform import create_robot_team
from analyst_team import create_analyst_team
from utils import load_agent_configs


logger = logging.getLogger(__name__)


relay = MediaRelay()


# lang = "Chinese" 
lang = "English"

async def get_team(
    user_input_func: Callable[[str, Optional[CancellationToken]], Awaitable[str]],
    session_dir: str
) -> RoundRobinGroupChat | SelectorGroupChat:
    
    # Create the team.
    scientist_team = create_scientist_team(user_input_func=user_input_func, lang=lang)
    engineer_team = create_engineer_team(user_input_func=user_input_func, lang=lang)
    robot_platform = create_robot_team(user_input_func=user_input_func, lang=lang)
    analyst_team = create_analyst_team(user_input_func=user_input_func, lang=lang)
    user = UserProxyAgent(
        name="user",
        input_func=user_input_func,  # Use the user input function.
    )
    cur_path = os.path.dirname(os.path.abspath(__file__))
    planning_agent = AssistantAgent(
        "ORCHESTRATOR",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=init_model('gpt-4o', structured_output=True),
        system_message=f"""
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
    Planning with {lang}:
    """,
        reflect_on_tool_use=False
    )

    # The termination condition is a combination of text mention termination and max message termination.
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=200)
    termination = text_mention_termination | max_messages_termination

    # The selector function is a function that takes the current message thread of the group chat
    # and returns the next speaker's name. If None is returned, the LLM-based selection method will be used.
    def selector_func(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
        if messages[-1].source != planning_agent.name:
            return planning_agent.name # Always return to the planning agent after the other agents have spoken.
        elif "HUMAN" in messages[-1].content:
            return user.name
        return None

    team = SelectorGroupChat(
        [planning_agent, user, scientist_team, engineer_team, robot_platform, analyst_team],
        model_client=init_model('gpt-4o', structured_output=True), # Use a smaller model for the selector.
        termination_condition=termination,
        selector_func=selector_func,
    )
    # Load state from file.
    state_path = os.path.join(session_dir, "team_state.json")
    if not os.path.exists(state_path):
        return team
    async with aiofiles.open(state_path, "r") as file:
        state = json.loads(await file.read())
    await team.load_state(state)
    return team



app = FastAPI()
current_task = None  # 用于跟踪当前任务
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# model_config_path = "model_config.yaml"
# state_path = "team_state.json"
# history_path = "team_history.json"

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")


async def get_session_history(session_dir: str) -> list[dict[str, Any]]:
    """Get chat history from file using UUID."""
    session_history_path = os.path.join(session_dir, "team_history.json")
    if not os.path.exists(session_history_path):
        return []
    async with aiofiles.open(session_history_path, "r") as file:
        content = await file.read()
        if content:
            return json.loads(content)
        else:
            return []


@app.websocket("/history/{session_uuid}")
async def history(websocket: WebSocket) -> list[dict[str, Any]]:
    await websocket.accept()
    data = await websocket.receive_json()
    session_uuid = data["uuid"]
    try:
        session_history = await get_session_history(session_uuid)
        await websocket.send_json(session_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.websocket("/sessions")
async def sessions(websocket: WebSocket) -> list[dict[str, str]]:
    """Get all history UUIDs and their main content."""
    await websocket.accept()
    cur_path = os.path.dirname(os.path.abspath(__file__))
    history_dir = os.path.join(cur_path, "history")
    session_data = []
    for dir_name in os.listdir(history_dir):
        session_dir = os.path.join(history_dir, dir_name)
        if os.path.isdir(session_dir):  # Check if it's a directory
            try:
                history = await get_session_history(session_dir)
                main_content = history[0]["content"] if history and "content" in history[0] else ""
                session_data.append({"uuid": dir_name, "content": main_content})

            except Exception as e:
                print(f"Error reading history for {dir_name}: {e}")  # Log the error but continue

    await websocket.send_json(session_data)


@app.websocket("/ws/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()

    # User input function used by the team.
    async def _user_input(prompt: str, cancellation_token: CancellationToken | None) -> str:
        data = await websocket.receive_json()
        return data['content']

    try:
        while True:
            # Get user message.
            data = await websocket.receive_json()

            if 'session_uuid' not in data:
                # New session
                request = TextMessage.model_validate(data)
                # request = data['content']
                session_uuid = str(uuid.uuid4())  # Initialize a unique UUID for each session
                cur_path = os.path.dirname(os.path.abspath(__file__))
                session_dir = os.path.join(cur_path, "history", session_uuid)  # Directory for session states
                os.makedirs(session_dir, exist_ok=True) # ensure the directory is created.
                history = []
            else:
                session_uuid = data['session_uuid']
                cur_path = os.path.dirname(os.path.abspath(__file__))
                session_dir = os.path.join(cur_path, "history", session_uuid)  # Directory for session states
                history = await get_session_history(session_dir)
                new_data = {k: v for k, v in data.items() if k != "session_uuid"}
                request = TextMessage.model_validate(new_data)
                request = history + request

            try:
                # Get the team and respond to the message.
                team = await get_team(_user_input, session_dir)
                
                stream = team.run_stream(task=request)
                async for message in stream:
                    if isinstance(message, TaskResult):
                        continue
                    print(f"----------------{message.source}----------------\n {message.content}")
                    if message.type == 'TextMessage' or message.type == 'HandoffMessage' \
                        or message.type == 'UserInputRequestedEvent':
                        if isinstance(message.content, str):
                            await websocket.send_json(message.model_dump())
                        else:
                            _message = message
                            _message.content = str(_message.content)
                            await websocket.send_json(_message.model_dump())
                    if not isinstance(message, UserInputRequestedEvent):
                        history.append(message.model_dump())

                    # Save chat history to file.
                    session_history_path = os.path.join(session_dir, "team_history.json")
                    async with aiofiles.open(session_history_path, "w") as file:
                        await file.write(json.dumps(history))
                
                # # Save team state to file.
                # session_state_path = os.path.join(session_dir, "team_state.json")
                # async with aiofiles.open(session_state_path, "w") as file:
                #     state = await team.save_state()
                #     await file.write(json.dumps(state))
                    
            except Exception as e:
                # Send error message to client
                error_message = {
                    "type": "error",
                    "content": f"Error: {str(e)}",
                    "source": "system"
                }
                await websocket.send_json(error_message)
                # Re-enable input after error
                await websocket.send_json({
                    "type": "UserInputRequestedEvent",
                    "content": "An error occurred. Please try again.",
                    "source": "system"
                })
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"Unexpected error: {str(e)}",
                "source": "system"
            })
        except:
            pass



RTSP_STREAMS = {
    "camera1": "rtsp://admin:@192.168.1.13:554/live",
    "camera2": "rtsp://admin:@192.168.1.10:554/live",
}

@app.websocket("/video_stream/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    rtsp_url = RTSP_STREAMS.get(camera_id)
    if not rtsp_url:
        await websocket.close(code=4001)
        return
    
    fps = 10
    frame_interval = 1 / fps
    max_buffer_size = 3  # 限制缓冲区大小，只保留最新的几帧
    frame_buffer = deque(maxlen=max_buffer_size)  # 设置最大长度
    
    # 启动FFmpeg进程
    process = (
        ffmpeg
        .input(rtsp_url, rtsp_transport='tcp')
        .output('pipe:', format='image2pipe', vcodec='mjpeg', r=fps)
        .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
    )
    
    async def capture_frames():
        """读取并分割完整的JPEG帧"""
        buffer = b''
        loop = asyncio.get_event_loop()
        while True:
            # 异步读取防止阻塞
            data = await loop.run_in_executor(None, process.stdout.read, 4096)
            if not data:
                break
            buffer += data
            # 提取完整帧
            while True:
                start = buffer.find(b'\xff\xd8')  # JPEG开始标记
                if start == -1:
                    break
                end = buffer.find(b'\xff\xd9', start + 2)  # JPEG结束标记
                if end == -1:
                    break
                # 提取帧数据并编码
                frame_data = buffer[start:end+2]
                frame = base64.b64encode(frame_data).decode('utf-8')
                
                # 如果缓冲区已满，清空旧帧再添加新帧
                if len(frame_buffer) >= max_buffer_size:
                    frame_buffer.clear()
                frame_buffer.append(frame)
                
                buffer = buffer[end+2:]  # 移除已处理数据
        # 清理FFmpeg进程
        process.kill()
    
    async def send_frames():
        """发送帧到客户端"""
        last_send_time = time.time()
        while True:
            current_time = time.time()
            elapsed = current_time - last_send_time
            
            # 确保按照帧率发送
            if elapsed >= frame_interval and frame_buffer:
                frame = frame_buffer.popleft()
                try:
                    await websocket.send_text(frame)
                    last_send_time = current_time
                except Exception:
                    break
            else:
                # 短暂休眠避免CPU过度使用
                await asyncio.sleep(max(0.001, frame_interval - elapsed))
    
    # 启动任务并处理退出
    capture_task = asyncio.create_task(capture_frames())
    send_task = asyncio.create_task(send_frames())
    
    try:
        await asyncio.gather(capture_task, send_task)
    except asyncio.CancelledError:
        capture_task.cancel()
        send_task.cancel()
        await asyncio.gather(capture_task, send_task, return_exceptions=True)
    finally:
        if process and process.poll() is None:
            process.kill()
        await websocket.close()


@app.get("/")
async def debug_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RTSP Stream</title>
    <style>
        #video {
            width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <!-- 视频流 1 的画布 -->
    <canvas id="canvas1"></canvas>
    <script>
        const canvas1 = document.getElementById('canvas1');
        const ctx1 = canvas1.getContext('2d');
        const img1 = new Image();

        img1.onload = () => {
            ctx1.drawImage(img1, 0, 0, canvas1.width, canvas1.height);
        };

        const ws1 = new WebSocket('ws://100.85.52.31:8000/video_stream/camera1');

        ws1.onopen = () => {
            console.log('WebSocket connection established for camera1'); // Log connection status.
        };

        ws1.onmessage = (event) => {
            img1.src = 'data:image/jpeg;base64,' + event.data;
        };

        ws1.onclose = () => {
            console.log('WebSocket connection closed for camera1');
        };

        // 动态调整 Canvas 尺寸
        function resizeCanvas1() {
            canvas1.width = 720; // 设置宽度为屏幕的一半
            canvas1.height = 480; // 设置高度为屏幕的一半
        }
        resizeCanvas1();
        window.addEventListener('resize', resizeCanvas1);


    </script>

    <!-- 视频流 2 的画布 -->
    <canvas id="canvas2"></canvas>
    <script>
        const canvas2 = document.getElementById('canvas2');
        const ctx2 = canvas2.getContext('2d');
        const img2 = new Image();

        img2.onload = () => {
            ctx2.drawImage(img2, 0, 0, canvas2.width, canvas2.height);
        };

        const ws2 = new WebSocket('ws://100.85.52.31:8000/video_stream/camera2');

        ws2.onopen = () => {
            console.log('WebSocket connection established for camera2');
        };


        ws2.onmessage = (event) => {
            img2.src = 'data:image/jpeg;base64,' + event.data;
        };

        ws2.onclose = () => {
            console.log('WebSocket connection closed for camera2');
        };

        // 动态调整 Canvas 尺寸
        function resizeCanvas2() {
            canvas2.width = 720; // 设置宽度为屏幕的一半
            canvas2.height = 480; // 设置高度为屏幕的一半
        }
        resizeCanvas2();
        window.addEventListener('resize', resizeCanvas2);
    </script>

</body>
</html>
""")


from pydantic import BaseModel
# 申明数据model
class LoginRequest(BaseModel):
    user_name: str
    pass_word: str
    code: str = ''   # 兼容传了验证码

@app.post("/matagent/login")
async def login(data: LoginRequest):
    # 你实际可以加验证码判断，这里略过
    if data.user_name == "test" and data.pass_word == "111111":
        # 简单生成一个token，实际生产用jwt等
        token = "fake_token_example"
        return {"token": token}
    else:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

# Example usage
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
