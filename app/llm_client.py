"""
Google Gemini LLM client wrapper with active context management,
resilient retry/fallback logic, and structured error handling.
"""

import time
import logging
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.persona import get_system_prompt
from app.context_manager import Conversation

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """Structured exception for LLM operations with HTTP status codes and friendly messages."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or message


class GeminiLLMClient:
    """Client for Google Gemini models through LangChain."""

    # Fallback models ordered by reliability if the primary model encounters 503 or 404
    FALLBACK_MODELS = [
        "gemini-flash-latest",
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.8-flash",
    ]

    @staticmethod
    def _message_text(message: AIMessage) -> str:
        """Extract plain text from LangChain string or structured content."""
        content = message.content
        if isinstance(content, str):
            return content
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    def __init__(self):
        self._client: Optional[ChatGoogleGenerativeAI] = None
        self._cached_api_key: Optional[str] = None
        self._cached_model: Optional[str] = None

    def get_client(self, model: Optional[str] = None) -> ChatGoogleGenerativeAI:
        """Return a cached LangChain Gemini chat model for the selected model."""
        if not settings.is_api_key_configured():
            raise LLMClientError(
                message="Google Gemini API key is missing or not configured. Please add your GEMINI_API_KEY in the .env file.",
                status_code=401,
            )
        current_key = settings.GEMINI_API_KEY
        current_model = model or settings.GEMINI_MODEL
        if (
            self._client is None
            or self._cached_api_key != current_key
            or self._cached_model != current_model
        ):
            self._client = ChatGoogleGenerativeAI(
                model=current_model,
                google_api_key=current_key,
                temperature=0.7,
                max_output_tokens=1500,
            )
            self._cached_api_key = current_key
            self._cached_model = current_model
        return self._client

    def _get_candidate_models(self) -> List[str]:
        """Returns the primary model followed by available fallbacks."""
        primary = settings.GEMINI_MODEL
        candidates = [primary]
        for m in self.FALLBACK_MODELS:
            if m != primary and m not in candidates:
                candidates.append(m)
        return candidates

    def _call_generate_with_resilience(
        self,
        contents: List[BaseMessage],
        temperature: float = 0.7,
        max_output_tokens: int = 1500,
        max_retries: int = 3
    ) -> Tuple[Any, str]:
        """
        Executes generate_content across candidate models with retry for transient 503/429 errors.
        Returns (response_object, model_used).
        """
        candidate_models = self._get_candidate_models()
        last_exception: Optional[Exception] = None

        for model in candidate_models:
            for attempt in range(max_retries):
                try:
                    client = self.get_client(model)
                    client.temperature = temperature
                    client.max_output_tokens = max_output_tokens
                    response = client.invoke(contents)
                    return response, model
                except Exception as e:
                    last_exception = e
                    err_str = str(e).lower()
                    status_code = getattr(e, "code", None) or getattr(e, "status_code", None)
                    if status_code in (401, 403) or "api_key_invalid" in err_str:
                        raise LLMClientError(
                            message="Invalid Gemini API Key. Please verify your GEMINI_API_KEY in the .env file.",
                            status_code=401,
                            details=str(e),
                        )
                    if status_code == 404 or "not found" in err_str:
                        logger.warning("Model '%s' returned 404. Attempting fallback model...", model)
                        break
                    if status_code in (429, 503) or "unavailable" in err_str:
                        logger.warning("Transient error on '%s' (attempt %d/%d). Retrying...", model, attempt + 1, max_retries)
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    if "timeout" in err_str or "timed out" in err_str:
                        logger.warning("Timeout calling '%s' (attempt %d/%d). Retrying...", model, attempt + 1, max_retries)
                        time.sleep(2.0)
                        continue
                    break

        # If all candidates failed, translate the last error into a clear LLMClientError
        if last_exception:
            self._handle_api_exception(last_exception, candidate_models[0])
        raise LLMClientError(message="Failed to generate response from Gemini API.", status_code=500)

    def _handle_api_exception(self, e: Exception, model_tried: str):
        """Translate LangChain/provider exceptions into friendly API errors."""
        logger.error("Gemini API call failed for model '%s': %s", model_tried, e)
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        error_text = str(e)
        if code in (401, 403) or "api_key_invalid" in error_text.lower():
            raise LLMClientError(
                message="Invalid Gemini API Key. Please verify your GEMINI_API_KEY in the .env file.",
                status_code=401,
                details=error_text,
            )
        if code == 429 or "resource_exhausted" in error_text.lower():
            raise LLMClientError(
                message="Rate limit or quota exceeded for Gemini API. Please wait a few moments before trying again.",
                status_code=429,
                details=error_text,
            )
        if code == 404 or "not found" in error_text.lower():
            raise LLMClientError(
                message=f"Gemini model '{model_tried}' is not found or is deprecated. Please set GEMINI_MODEL to 'gemini-flash-latest' in your .env file.",
                status_code=404,
                details=error_text,
            )
        if code and code >= 500:
            raise LLMClientError(
                message="Google Gemini servers are temporarily experiencing high demand. Please try again shortly.",
                status_code=503,
                details=error_text,
            )
        error_str = str(e).lower()
        if "timeout" in error_str or "timed out" in error_str:
            raise LLMClientError(
                message="Request to Gemini API timed out. Please check your network connection and try again.",
                status_code=504,
                details=str(e)
            )
        elif "connect" in error_str or "unreachable" in error_str:
            raise LLMClientError(
                message="Unable to reach Gemini API. Please check your internet connectivity.",
                status_code=502,
                details=str(e)
            )
        else:
            raise LLMClientError(
                message=f"An error occurred while communicating with Gemini: {str(e)}",
                status_code=500,
                details=str(e)
            )

    def validate_api_key(self) -> Tuple[bool, str]:
        """Tests whether the configured API key can authenticate and query Gemini."""
        if not settings.is_api_key_configured():
            return False, "Gemini API key is not configured in .env."
        try:
            response, model_used = self._call_generate_with_resilience(
                contents=[HumanMessage(content="ping")],
                max_output_tokens=5,
                max_retries=1
            )
            return True, f"Connected to {model_used} successfully."
        except Exception as e:
            return False, f"Gemini connection failed: {e}"

    def summarize_conversation_chunk(self, previous_summary: str, messages: List[BaseMessage]) -> str:
        """Summarizes an older segment of conversation to condense memory."""
        formatted_chunk = "\n".join([f"{m.type.upper()}: {m.content}" for m in messages])
        prompt = f"""You are a conversation summarizer maintaining memory for an AI assistant.
Update the running summary of this conversation with the new conversation segment below.

Existing Summary:
{previous_summary if previous_summary else '(None)'}

New Conversation Segment to absorb:
{formatted_chunk}

Instructions:
- Keep the summary concise (under 150 words).
- Retain key facts, user preferences, names, and technical project details.
- Output ONLY the updated summary text."""

        try:
            response, _ = self._call_generate_with_resilience(
                contents=[HumanMessage(content=prompt)],
                temperature=0.2,
                max_output_tokens=300,
                max_retries=2
            )
            text = self._message_text(response).strip()
            return text if text else previous_summary
        except Exception as e:
            logger.warning("Summarization fallback due to error: %s", e)
            fallback = f"{previous_summary}\n[Extracts: " + "; ".join([m.content[:50] for m in messages[:3]]) + "]"
            return fallback.strip()

    def generate_response(
        self,
        conversation: Conversation,
        user_message: str,
        custom_system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Executes a multi-turn conversation turn:
        1. Records user message
        2. Applies context management strategy (summarization trigger or sliding window)
        3. Prepares messages and system instruction
        4. Calls Gemini model with resilience
        5. Records response and returns telemetry
        """
        # 1. Add user message
        conversation.add_user_message(user_message)

        # 2. Check if summarization strategy requires summarizing older turns
        if conversation.strategy == "summarization" and conversation.needs_summarization():
            to_summarize = conversation.get_messages_to_summarize()
            if to_summarize:
                new_summary = self.summarize_conversation_chunk(
                    conversation.running_summary,
                    to_summarize
                )
                conversation.commit_summary(new_summary, len(to_summarize))

        # 3. Prepare formatted context
        raw_contents = conversation.get_context_messages()

        # Build system instruction (base persona + any running memory)
        base_system_prompt = get_system_prompt(custom_system_prompt)
        full_system_instruction = base_system_prompt
        if conversation.running_summary:
            full_system_instruction = (
                f"{base_system_prompt}\n\n"
                f"### Running Conversation Memory\n"
                f"{conversation.running_summary}\n"
            )

        formatted_contents: List[BaseMessage] = [SystemMessage(content=full_system_instruction)]
        formatted_contents.extend(raw_contents)

        response, model_used = self._call_generate_with_resilience(
            contents=formatted_contents,
            temperature=0.7,
            max_output_tokens=1500,
            max_retries=3
        )

        reply_text = self._message_text(response)
        if not reply_text:
            reply_text = "I received your message, but the response was filtered or empty."

        # 5. Record model response in conversation memory
        conversation.add_ai_message(reply_text)

        # 6. Telemetry stats
        stats = conversation.get_stats()
        stats["system_prompt_length"] = len(full_system_instruction)
        stats["model_used"] = model_used

        return reply_text, stats


# Global LLM client singleton
llm_client = GeminiLLMClient()
