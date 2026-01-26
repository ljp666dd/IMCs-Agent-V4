from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()


def upload_to_s3(json_data: str):
    import json
    import re
    import subprocess
    import sys
    import tempfile
    import datetime

    def install_boto3():
        try:
            # 检查 boto3 是否已安装
            import boto3
            print("boto3 已安装。")
        except ImportError:
            # 如果未安装，动态安装 boto3
            print("正在安装 boto3...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3"])
            print("boto3 安装完成。")

    def handle_minio_upload(file_path: str, file_name: str) -> str:
        """统一处理MinIO上传"""
        import boto3
        try:
            client = boto3.client(
                's3',
                endpoint_url="http://100.85.52.31:9000" or "https://s3-api.siat-mic.com",
                aws_access_key_id="9bUtQL1Gpo9JB6o3pSGr",
                aws_secret_access_key="1Qug5H73R3kP8boIHvdVcFtcb1jU9GRWnlmMpx0g"
            )
            client.upload_file(file_path, "temp", file_name, ExtraArgs={"ACL": "private"})
            
            # 生成预签名 URL
            url = client.generate_presigned_url(
                'get_object',
                Params={'Bucket': "temp", 'Key': file_name},
                ExpiresIn=3600
            )
            return url.replace("http://100.85.52.31:9000" or "", "https://s3-api.siat-mic.com")
        
        except Exception as e:
            # print(e)
            return f"Error: {str(e)}, Request human/user intervention."
        
    install_boto3()
    # 去掉可能存在的 ```json 和 ``` 标记
    json_data_cleaned = re.sub(r'```json|```', '', json_data).strip()
    try:
        # 尝试解析清理后的JSON数据
        data = json.loads(json_data_cleaned)
        # 取得task id
        task_id = data['TaskId']
        # print("解析后的JSON数据:", data)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            try:
                json.dump(data, temp_file, indent=4, ensure_ascii=False)
                temp_file.flush()  # 确保数据写入文件
                file_name = f"robotExprimentScheme_{task_id}.json"
                url = handle_minio_upload(temp_file.name, file_name)
                return f"JSON Scheme has been uploaded to S3 storage. The unique URL is: {url}, please pass it to the robot platform."
            except Exception as e:
                # print(f"写入临时文件或上传文件时出错: {e}")
                return f"Error: {str(e)}, Request human/user intervention."
    except json.JSONDecodeError as e:
        # print(f"JSON解析错误: {e}")
        return f"Error: {str(e)}, Request human/user intervention."


@app.post("/sendScheme2RobotWorkstation")
async def receive_url(scheme_request: dict):
    """
    接收并处理机器人方案
    Args:
        scheme_request (dict): 包含机器人方案的请求体
        
    Returns:
        dict: 包含处理结果的响应
    """
    try:
        # 这里可以添加处理机器人方案的逻辑
        print(f"Received scheme: {scheme_request}")
        
        # 返回成功响应
        return {
            "status": "success",
            "message": "Scheme received successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sendScheme2MobileRobot")
async def receive_url(scheme_request: dict):
    """
    接收并处理机器人方案
    Args:
        scheme_request (dict): 包含机器人方案的请求体
        
    Returns:
        dict: 包含处理结果的响应
    """
    try:
        # 这里可以添加处理机器人方案的逻辑
        print(f"Received scheme: {scheme_request}")
        
        # 返回成功响应
        return {
            "status": "success",
            "message": "Scheme received successfully",
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_server(host: str = "0.0.0.0", port: int = 50000):
    """
    启动FastAPI服务器
    
    Args:
        host (str): 服务器主机地址
        port (int): 服务器端口号
    """
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
