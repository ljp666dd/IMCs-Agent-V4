from pathlib import Path
from typing import Optional
import os
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core.models import AssistantMessage, LLMMessage, ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Define your API keys and configurations
OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""
MODEL = "gpt-4o-2024-11-20"
config_list = [{"model": MODEL, "api_key": OPENAI_API_KEY, "base_url": OPENAI_BASE_URL}]

# agent setting
SILENT = True # 关闭嵌套智能体的输出
STREAM = True # stream on console
CACHE = None # None 就是关闭 41是默认值开启

# rag setting
GRAPH_RAG = "graphrag/settings.yaml"
# hybrid rag setting of dify
HYBRID_RAG_HOST_URI = "http://10.10.17.3:4080/v1/chat-messages"
HYBRID_RAG_HOST_KEY = "app-uJgo3TQKcS1O9PMCDHko71Fp"
PAPER_RAG_HOST_KEY = "app-GxR6tNNGSDKngzsPkph6j20B"
ONLINE_RAG_HOST_KEY = "app-KUl65xV6JcFDziWJgizlHV27"
# minio setting
MINIO_SERVER = "http://100.85.52.31:9000"
MINIO_KEY_ID = "9bUtQL1Gpo9JB6o3pSGr"
MINIO_KEY = "1Qug5H73R3kP8boIHvdVcFtcb1jU9GRWnlmMpx0g"
MINIO_BUCKET = "temp"
# mcp setting
MCP_SERVER_URI = "https://mcpserver.matai.center/sse"
# Robot Platform and Mobile Robot HTTP Server URI
PLATFORM_HTTP_SERVER_URI = "http://100.122.132.69:50000/sendScheme2RobotPlatform"
MOBILE_ROBOT_HTTP_SERVER_URI = "http://100.122.132.69:50000/sendScheme2MobileRobot"
# PL FastAPI Server URI
PL_SERVER_URI = "http://100.122.132.69:50000/getPLdata"


WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".coding")
if not os.path.exists(WORK_DIR):
    os.mkdir(WORK_DIR)


def init_model(name: str, structured_output: bool = False, timeout: Optional[int] = None):
    if "gpt-4o" in name:
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-2024-11-20",
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            model_info={
                "vision": True,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.GPT_4O,
                "structured_output": structured_output,
                "multiple_system_messages": True
            },
            timeout=30 if not timeout else timeout,
            max_retries=5,
            max_tokens=4096
        )
    
    elif "deepseek-r1" == name or "deepseek-reasoner" == name or "deepseek-ai/DeepSeek-R1" == name:
        model_client = OpenAIChatCompletionClient(
            model=name,
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.R1,
                "structured_output": structured_output,
                "multiple_system_messages": True
            },
            timeout=120 if not timeout else timeout,
            max_retries=3,
            max_tokens=4096
        )
    
    elif "deepseek-v3" == name or "deepseek-chat" == name:
        model_client = OpenAIChatCompletionClient(
            model=name,
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.R1,
                "structured_output": structured_output,
                "multiple_system_messages": True
            },
            timeout=120 if not timeout else timeout,
            max_retries=3,
            max_tokens=4096
        )
    
    elif "kimi-k2-0711-preview" == name or "moonshotai/Kimi-K2-Instruct" == name:
        model_client = OpenAIChatCompletionClient(
            model=name,
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.UNKNOWN,
                "structured_output": structured_output,
                "multiple_system_messages": True
            },
            timeout=120 if not timeout else timeout,
            max_retries=3,
            max_tokens=4096
        )

    else:
        raise NotImplementedError(f"Model {name} is not implemented.")
    return model_client