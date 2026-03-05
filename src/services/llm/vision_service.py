import os
import json
import base64
from typing import List, Dict, Any, Optional
from src.core.logger import get_logger
from src.config.config import config

logger = get_logger(__name__)

class VisionService:
    """
    Service for multi-modal analysis of document images using LLM (Gemini Vision).
    """
    def __init__(self):
        self.available = False
        gemini_key = os.environ.get("GEMINI_API_KEY", getattr(config, "GEMINI_API_KEY", ""))
        
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                # Using flash model for high-speed vision tasks
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.available = True
                logger.info("VisionService initialized with Gemini 1.5 Flash.")
            except ImportError:
                logger.warning("google-generativeai not installed for VisionService.")
            except Exception as e:
                logger.error(f"VisionService init failed: {e}")

    def analyze_page(self, image_path: str, task: str = "extract_tables") -> Dict[str, Any]:
        """
        Analyze a page image for specific scientific data.
        Task can be 'extract_tables', 'analyze_curves', or 'general_knowledge'.
        """
        if not self.available:
            return {"error": "VisionService not available"}
            
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
            
        prompt = self._get_prompt_for_task(task)
        
        try:
            from PIL import Image
            img = Image.open(image_path)
            
            # Call Gemini Vision
            response = self.model.generate_content([prompt, img])
            text = response.text
            
            # Extract JSON if present
            return self._parse_json_result(text)
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return {"error": str(e)}

    def digitize_curve(self, image_path: str) -> Dict[str, Any]:
        """
        Digitize a scientific curve by:
        1. Calibrating axes (pixel to physical mapping).
        2. Sampling curve points.
        """
        if not self.available:
            return {"error": "VisionService not available"}
            
        logger.info(f"Digitizing curve in {image_path}")
        
        # Step 1: Calibration Prompt
        calibration_prompt = (
            "你是一个高精度科学仪器数据提取专家。请分析图片中 LSV 或极化曲线图的坐标轴。\n"
            "请识别并返回以下像素点坐标（(x, y) 格式，(0,0) 是左上角）：\n"
            "1. 坐标原点 (Origin)\n"
            "2. X 轴上一个已知的刻度点及其物理值（如电位 V）\n"
            "3. Y 轴上一个已知的刻度点及其物理值（如电流密度 mA/cm2）\n"
            "请严格以以下 JSON 格式返回：\n"
            "{\n"
            "  \"origin\": [px_x, px_y],\n"
            "  \"x_tick\": {\"px\": [px_x, px_y], \"value\": val_x},\n"
            "  \"y_tick\": {\"px\": [px_x, px_y], \"value\": val_y},\n"
            "  \"x_label\": \"Potential / V\",\n"
            "  \"y_label\": \"Current Density / mA cm-2\"\n"
            "}"
        )
        
        # Step 2: Sampling Prompt
        sampling_prompt = (
            "请识别图片中主要的实验曲线轨迹。请沿曲线从左到右等间距提取 20 个点的像素坐标 [px_x, px_y]。\n"
            "只输出 JSON 数据：{\"pixel_points\": [[x1, y1], [x2, y2], ...]}"
        )
        
        try:
            from PIL import Image
            img = Image.open(image_path)
            
            # Get Calibration
            cal_resp = self.model.generate_content([calibration_prompt, img])
            cal_data = self._parse_json_result(cal_resp.text)
            
            # Get Points
            pts_resp = self.model.generate_content([sampling_prompt, img])
            pts_data = self._parse_json_result(pts_resp.text)
            
            if "error" in cal_data or "pixel_points" not in pts_data:
                return {"error": "Failed to extract features", "raw": [cal_resp.text, pts_resp.text]}
                
            origin_px = cal_data["origin"]
            x_tick = cal_data["x_tick"]
            y_tick = cal_data["y_tick"]
            
            dx_px = x_tick["px"][0] - origin_px[0]
            dx_val = x_tick["value"] - 0
            
            dy_px = y_tick["px"][1] - origin_px[1] 
            dy_val = y_tick["value"] - 0
            
            scale_x = dx_val / dx_px if dx_px != 0 else 1
            scale_y = dy_val / dy_px if dy_px != 0 else 1
            
            physical_points = []
            for px in pts_data["pixel_points"]:
                vx = (px[0] - origin_px[0]) * scale_x
                vy = (px[1] - origin_px[1]) * scale_y
                physical_points.append({"x": round(vx, 4), "y": round(vy, 4)})
            
            return {
                "success": True,
                "metadata": cal_data,
                "data_points": physical_points,
                "units": {"x": cal_data.get("x_label"), "y": cal_data.get("y_label")}
            }
            
        except Exception as e:
            logger.error(f"Curve digitization failed: {e}")
            return {"error": str(e)}

    def _get_prompt_for_task(self, task: str) -> str:
        if task == "extract_tables":
            return (
                "你是一个高度精确的文档解析机器人。请分析这张图片，找到所有包含电催化（HOR/HER）性能的表格。"
                "请提取这些表格并以 JSON 格式输出。JSON 应该是一个列表，每个项包含 'material', 'metric' (如 'overpotential', 'exchange_current'), 'value', 'units'。"
                "只输出 JSON 块，不要有任何解释。"
            )
        elif task == "analyze_curves":
            return (
                "分析这张图片中的实验曲线（通常是 LSV 或极化曲线）。请识别 X 轴（电位）和 Y 轴（电流密度）。"
                "描述曲线的主要物理特征：起始电位位置、在 0.1V 时的电流密度估计。以 JSON 格式返回结果。"
            )
        else:
            return "分析这张图片中的科学内容，并以结构化 JSON 格式提取关键发现。"

    def _parse_json_result(self, text: str) -> Dict[str, Any]:
        try:
            # Look for ```json ... ```
            import re
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # Simple fallback for standard JSON blobs without markdown
            match_raw = re.search(r'\{.*\}', text, re.DOTALL)
            if match_raw:
                return json.loads(match_raw.group(0))
                
            return json.loads(text)
        except Exception:
            return {"error": "JSON parse failed", "raw_text": text}

def get_vision_service():
    return VisionService()
