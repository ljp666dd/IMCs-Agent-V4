"""
Author: Yutang LI
Institution: SIAT-MIC
Contact: yt.li2@siat.ac.cn
"""

import os
import boto3
import logging
import yaml
from typing import Optional, Dict
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # Model
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(None, env="OPENAI_BASE_URL")
    model_name: Optional[str] = Field(None, env="MODEL_NAME")

    # Path
    coding_path: Optional[str] = Field(None, env=os.path.join(os.path.dirname(os.path.abspath(__file__)), "CODING_PATH"))
    
    # MinIO
    minio_endpoint: Optional[str] = Field(None, env="MINIO_ENDPOINT")
    internal_minio_endpoint: Optional[str] = Field(None, env="INTERNAL_MINIO_ENDPOINT")
    minio_access_key: Optional[str] = Field(None, env="MINIO_ACCESS_KEY")
    minio_secret_key: Optional[str] = Field(None, env="MINIO_SECRET_KEY")
    minio_bucket: Optional[str] = Field("temp", env="MINIO_BUCKET")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 初始化配置
settings = Settings()

def handle_minio_error(e: Exception) -> Dict[str, str]:
    """处理MinIO相关错误"""
    logger.error(f"MinIO operation failed: {str(e)}")
    return {
        "status": "error",
        "data": f"MinIO operation failed: {str(e)}"
    }

def get_minio_client(settings: Settings):
    """获取MinIO客户端"""
    return boto3.client(
        's3',
        endpoint_url=settings.internal_minio_endpoint or settings.minio_endpoint,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key
    )

def handle_minio_upload(file_path: str, file_name: str) -> str:
    """统一处理MinIO上传"""
    try:
        client = get_minio_client(settings)
        client.upload_file(file_path, settings.minio_bucket, file_name, ExtraArgs={"ACL": "private"})
        
        # 生成预签名 URL
        url = client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.minio_bucket, 'Key': file_name},
            ExpiresIn=3600
        )
        return url.replace(settings.internal_minio_endpoint or "", settings.minio_endpoint)
    
    except Exception as e:
        return handle_minio_error(e)
    
def load_agent_configs(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)