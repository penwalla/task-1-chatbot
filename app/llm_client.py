"""
Google Gemini LLM client wrapper with error handling and context window support.
Uses the modern google-genai SDK with resilient fallbacks.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from google import genai
from google.genai import types
from google.genai import errors

from app.config import settings
from app.persona import get_system_prompt
from app.context_manager import Conversation, Message

logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """Custom exception containing user-friendly error messages."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or message


class GeminiLLMClient:
    """Client for Google Gemini API."""

    def __init__(self):
        self._client: Optional[genai.Client] = None

    def get_client(self) -> genai.Client:
        """Initialize or return the cached genai Client."""
        if not settings.is_api_key_configured():
            raise LLMClientError(
                message="Google Gemini API key is missing or not configured. Please add your GEMINI_API_KEY in the .env file. (Get a free key from https://aistudio.google.com/app/apikey)",
                status_code=401,
                details="API key is empty or set to placeholder value."
            )
        
        if self._client is None or self._client.api_key != settings.GEMINI_API_KEY:
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    def validate_api_key(self) -> Tuple[bool, str]:
        """Validate if the current API key is set and connects successfully."""
        if not settings.is_api_key_configured():
            return False, "Gemini API key is not configured. Please set GEMINI_API_KEY in .env file."
        
        try:
            client = self.get_client()
            # Send a minimal test message
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents="ping",
                config=types.GenerateContentConfig(max_output_tokens=5)
            )
            return True, f"Connected to {settings.GEMINI_MODEL} successfully."
        except errors.APIError as e:
            if "API_KEY_INVALID" in str(e) or e.code == 400 or e.code == 403:
                return False, "Invalid API key. Please check your GEMINI_API_KEY in .env."
            elif e.code == 429:
                return False, "Gemini API quota/rate limit exceeded. Check your Google AI Studio plan."
            return False, f"Gemini API error ({e.code}): {e.message}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def summarize_conversation_chunk(self, previous_summary: str, messages: List[Message]) -> str:
        """
        Summarize older conversation turns into a progressive running memory.
        """
        client = self.get_client()
        
        formatted_chunk = "\n".join([f"{m.role.upper()}: {m.content}" for m in messages])
        
        prompt = f"""You are a conversation summarizer maintaining memory for an AI assistant.
Update the running summary of this conversation with the new conversation segment below.

Existing Summary:
{previous_summary if previous_summary else '(None - this is the start of the conversation)'}

New Conversation Segment to absorb:
{formatted_chunk}

Instructions:
- Keep the summary concise (under 150 words).
- Retain key facts, user preferences, names, project details, code languages, and specific decisions.
- Output ONLY the updated summary text with no commentary."""

        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=300
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Summarization call failed, falling back to heuristic: {e}")
            # Resilient fallback: produce a concise extractive note
            fallback = f"{previous_summary}\n[Earlier topics: " + "; ".join([m.content[:50] for m in messages[:3]]) + "]"
            return fallback.strip()

    def generate_response(
        self,
        conversation: Conversation,
        user_message: str,
        custom_system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Sends the user message and managed conversation history to Gemini,
        returning the response and context management metadata.
        """
        client = self.get_client()

        # Step 1: Add current user message to conversation history
        conversation.add_message(role="user", content=user_message)

        # Step 2: Handle Summarization Strategy if applicable
        if conversation.strategy == "summarization" and conversation.needs_summarization():
            to_summarize = conversation.get_messages_to_summarize()
            if to_summarize:
                new_summary = self.summarize_conversation_chunk(
                    previous_summary=conversation.running_summary,
                    messages=to_summarize
                )
                conversation.commit_summary(new_summary, len(to_summarize))

        # Step 3: Prepare the context according to strategy
        if conversation.strategy == "summarization":
            raw_contents = conversation.prepare_summarized_context()
        else: # sliding_window
            raw_contents = conversation.prepare_sliding_window_context()

        # Step 4: Build System Instruction (Persona + Running Summary if present)
        base_system_prompt = get_system_prompt(custom_system_prompt)
        if conversation.running_summary:
            full_system_instruction = (
                f"{base_system_prompt}\n\n"
                f"### Running Conversation Memory\n"
                f"Below is a summarized memory of earlier parts of this conversation:\n"
                f"{conversation.running_summary}\n"
                f"Use this memory to maintain continuity when referenced by the user."
            )
        else:
            full_system_instruction = base_system_prompt

        # Step 5: Convert contents to types.Content format for google-genai
        formatted_contents: List[types.Content] = []
        for item in raw_contents:
            role = item["role"]
            # Convert 'model' role to Gemini expected role 'model', 'user' to 'user'
            text = "".join(item.get("parts", []))
            formatted_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text)]
                )
            )

        # Step 6: Execute API call with comprehensive error catching
        try:
            config = types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                temperature=0.7,
                max_output_tokens=1500,
            )

            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=formatted_contents,
                config=config
            )

            reply_text = response.text or ""
            if not reply_text:
                # Handle edge case where safety filter or empty candidate occurred
                reply_text = "I received your message, but the response was filtered or empty. Please try phrasing your prompt differently."

            # Step 7: Record the model response in conversation history
            conversation.add_message(role="model", content=reply_text)

            # Context window stats
            stats = conversation.get_stats()
            stats["system_prompt_length"] = len(full_system_instruction)
            stats["model_used"] = settings.GEMINI_MODEL

            return reply_text, stats

        except errors.APIError as e:
            logger.error(f"Gemini API error: {e}")
            if "API_KEY_INVALID" in str(e) or e.code in (400, 403):
                raise LLMClientError(
                    message="Invalid Gemini API Key. Please verify your GEMINI_API_KEY in the .env file.",
                    status_code=401,
                    details=str(e)
                )
            elif e.code == 429:
                raise LLMClientError(
                    message="Rate limit or quota exceeded for Gemini API. Please wait a few moments before trying again.",
                    status_code=429,
                    details=str(e)
                )
            elif e.code >= 500:
                raise LLMClientError(
                    message="Google Gemini servers are temporarily unavailable. Please try again shortly.",
                    status_code=503,
                    details=str(e)
                )
            else:
                raise LLMClientError(
                    message=f"Gemini API returned error ({e.code}): {e.message}",
                    status_code=e.code or 500,
                    details=str(e)
                )
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Unexpected error calling Gemini: {e}")
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
                    message=f"An error occurred while generating response: {str(e)}",
                    status_code=500,
                    details=str(e)
                )


# Global LLM client singleton
llm_client = GeminiLLMClient()
