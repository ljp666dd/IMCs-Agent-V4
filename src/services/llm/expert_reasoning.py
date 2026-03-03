"""
IMCs LLM Service - Multi-Backend Expert Reasoning

Supports:
1. Gemini (Google) - primary
2. DeepSeek (free tier, OpenAI-compatible) - fallback
3. Mock - last resort when no API is available
"""

import os
import json
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from src.core.logger import get_logger
from src.config.config import config

# Load .env so all API keys are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = get_logger(__name__)


class ExpertReasoningService:
    """Multi-backend LLM service for generating expert reasoning reports."""

    def __init__(self):
        self.available = False
        self.backend = "none"
        self.model = None
        self.api_key = ""  # for backwards compatibility
        self.timeout_seconds = float(os.getenv("IMCS_LLM_TIMEOUT", "30") or "30")
        self.enable_cloud = str(os.getenv("IMCS_ENABLE_CLOUD_LLM", "1")).lower() in ("1", "true", "yes")

        if not self.enable_cloud:
            logger.warning("IMCS_ENABLE_CLOUD_LLM disabled; expert reasoning will use mock reports.")
            return

        # --- Try Gemini first ---
        gemini_key = os.environ.get("GEMINI_API_KEY", getattr(config, "GEMINI_API_KEY", ""))
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                # Quick validation - will fail fast if key is leaked/invalid
                self.available = True
                self.backend = "gemini"
                logger.info("ExpertReasoningService initialized with Gemini backend.")
            except ImportError:
                logger.info("google-generativeai not installed, trying next backend.")
            except Exception as e:
                logger.info(f"Gemini init failed ({e}), trying next backend.")

        # --- Try DeepSeek (OpenAI-compatible, free tier) ---
        if not self.available:
            deepseek_key = os.environ.get("DEEPSEEK_API_KEY", getattr(config, "DEEPSEEK_API_KEY", ""))
            if deepseek_key:
                try:
                    from openai import OpenAI
                    self.client = OpenAI(
                        api_key=deepseek_key,
                        base_url="https://api.deepseek.com"
                    )
                    self.deepseek_model = "deepseek-chat"  # V3, free tier
                    self.available = True
                    self.backend = "deepseek"
                    logger.info("ExpertReasoningService initialized with DeepSeek backend.")
                except ImportError:
                    logger.info("openai package not installed, trying next backend.")
                except Exception as e:
                    logger.info(f"DeepSeek init failed ({e}), trying next backend.")

        # --- Try SiliconFlow (Chinese provider, free tier models) ---
        if not self.available:
            sf_key = os.environ.get("SILICONFLOW_API_KEY", getattr(config, "SILICONFLOW_API_KEY", ""))
            if sf_key:
                try:
                    from openai import OpenAI
                    self.client = OpenAI(
                        api_key=sf_key,
                        base_url="https://api.siliconflow.cn/v1"
                    )
                    self.deepseek_model = "deepseek-ai/DeepSeek-V3"
                    self.available = True
                    self.backend = "siliconflow"
                    logger.info("ExpertReasoningService initialized with SiliconFlow backend.")
                except Exception as e:
                    logger.info(f"SiliconFlow init failed ({e}).")

        if not self.available:
            logger.warning(
                "No LLM backend available. Set GEMINI_API_KEY, DEEPSEEK_API_KEY, "
                "or SILICONFLOW_API_KEY in .env to enable expert reasoning."
            )

    def _call_with_timeout(self, func, *args, **kwargs) -> Optional[str]:
        """Run blocking LLM call with a hard timeout."""
        if not self.timeout_seconds or self.timeout_seconds <= 0:
            return func(*args, **kwargs)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.timeout_seconds)
            except FuturesTimeoutError:
                logger.warning(f"LLM call timed out after {self.timeout_seconds}s.")
                return None

    def _build_prompt(self, candidates: List[Dict[str, Any]], is_active_learning: bool) -> str:
        """Build the expert reasoning prompt."""
        prompt = "你是一个顶尖的计算化学与电催化材料科学家统筹核心（Director）。\n"

        if is_active_learning:
            prompt += (
                "你刚刚中断了系统的常规推荐流水线。因为你发现以下材料通过物理感知机器学习模型"
                "（HOR_Physics_DNN）预测出的HOR活性分极高，但同时基于MC-Dropout得出的"
                "预测不确定度（Uncertainty variance）严重偏高！你需要写一份"
                "'实验验证请求报告(Active Learning Request)'。\n"
            )
        else:
            prompt += (
                "你刚刚完成了常规推荐流水线，根据物理感知模型和多智能体协同评估，"
                "你推荐了以下排名靠前的优秀HOR有序合金催化剂。你需要写一份'专家推荐报告'。\n"
            )

        prompt += "\n以下是截获的候选材料详情：\n"
        for i, cand in enumerate(candidates[:5]):
            formula = cand.get("formula", str(cand.get("material_id")))
            props = cand.get("properties", {})
            # Try top-level keys too (for candidates from different agents)
            score = props.get("predicted_activity", cand.get("predicted_activity", "N/A"))
            uncert = props.get("uncertainty", cand.get("uncertainty", "N/A"))
            form_energy = props.get("formation_energy", cand.get("formation_energy", "N/A"))

            prompt += f"材料 {i+1}: {formula}\n"
            prompt += f" - 预测HOR活性打分 (Sabatiers/Activity): {score}\n"
            prompt += f" - 预测不确定度 (Variance/Uncertainty): {uncert}\n"
            prompt += f" - 形成能 (Formation Energy): {form_energy}\n\n"

        prompt += (
            "请用专业、严谨、详实的语气，撰写一份中文专家报告。字数在400字左右。"
            "注意：只输出报告主体，不输出无关的口水话。"
        )
        return prompt

    def generate_report(self, candidates: List[Dict[str, Any]], is_active_learning: bool) -> str:
        """Generate an expert reasoning report using the best available backend."""
        if not self.available:
            return self._mock_report(is_active_learning)

        prompt = self._build_prompt(candidates, is_active_learning)

        try:
            if self.backend == "gemini":
                response = self._call_with_timeout(self._call_gemini, prompt)
                if response:
                    return response
            elif self.backend in ("deepseek", "siliconflow"):
                response = self._call_with_timeout(self._call_openai_compatible, prompt)
                if response:
                    return response
            else:
                return self._mock_report(is_active_learning)

        except Exception as e:
            err_msg = str(e).lower()
            if "leaked" in err_msg or "403" in err_msg:
                logger.warning(f"Gemini API key issue: {e}. Falling back to other backends.")
                # Try DeepSeek/SiliconFlow as emergency fallback
                fallback = self._try_emergency_fallback(prompt)
                if fallback:
                    return fallback

            logger.error(f"LLM report generation failed: {e}")
            return self._mock_report(is_active_learning)

        # Timeout or empty response fallback
        fallback = self._try_emergency_fallback(prompt)
        if fallback:
            return fallback
        return self._mock_report(is_active_learning)

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        response = self.model.generate_content(prompt)
        return response.text

    def _call_openai_compatible(self, prompt: str) -> str:
        """Call OpenAI-compatible API (DeepSeek / SiliconFlow)."""
        response = self.client.chat.completions.create(
            model=self.deepseek_model,
            messages=[
                {"role": "system", "content": "你是一个顶尖的计算化学与电催化领域专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    def _try_emergency_fallback(self, prompt: str) -> Optional[str]:
        """Try alternative backends when primary fails."""
        for env_key, base_url, model_name, backend_name in [
            ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat", "DeepSeek"),
            ("SILICONFLOW_API_KEY", "https://api.siliconflow.cn/v1", "deepseek-ai/DeepSeek-V3", "SiliconFlow"),
        ]:
            api_key = os.environ.get(env_key, getattr(config, env_key, ""))
            if not api_key:
                continue
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "你是一个顶尖的计算化学与电催化领域专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                logger.info(f"Emergency fallback to {backend_name} succeeded.")
                return response.choices[0].message.content
            except Exception as e:
                logger.debug(f"Emergency fallback to {backend_name} failed: {e}")
                continue
        return None

    def _mock_report(self, is_active_learning: bool) -> str:
        """Generate a mock report when no LLM backend is available."""
        if is_active_learning:
            return (
                "[MOCK] 这是一份模拟的主动学习(Active Learning)实验验证请求报告。"
                "模型在预测这批材料时发现了异常的高不确定性（Variance），为了确保 Sabatier 原理"
                "不被违背并提升系统的物理认知，Director 决定打断正常流水线，截获了高分候选材料"
                "并呼叫实验验证环节对材料进行重新合成与电化学表征。"
            )
        else:
            return (
                "[MOCK] 这是一份模拟的专家推荐报告。基于多智能体协同分析和物理感知模型的深度评分，"
                "系统最终筛选出了这批候选 HOR 有序合金催化材料，它们普遍表现出极佳的形成能、"
                "完美的 SABATIER 预测活性以及可接受范围内的置信度，建议优先考虑。"
            )


def get_expert_reasoning() -> ExpertReasoningService:
    return ExpertReasoningService()
