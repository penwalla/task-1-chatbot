"""
Context window management using LangChain.

Uses:
- langchain_core.messages: HumanMessage, AIMessage, SystemMessage
- langchain_core.chat_history: InMemoryChatMessageHistory
- langchain_core.messages.utils: trim_messages for sliding window
"""

import time
from typing import Dict, Any, Optional, List

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from app.config import settings


class _CompatibleHumanMessage(HumanMessage):
    @property
    def role(self) -> str:
        return "user"

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


class _CompatibleAIMessage(AIMessage):
    @property
    def role(self) -> str:
        return "model"

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


class Conversation:
    """Wraps a LangChain InMemoryChatMessageHistory with context window tracking."""

    def __init__(
        self,
        conversation_id: str,
        strategy: Optional[str] = None,
        max_messages: Optional[int] = None,
    ):
        self.conversation_id = conversation_id
        self.strategy = (strategy or settings.CONTEXT_STRATEGY).lower()
        self.max_messages = max_messages or settings.MAX_CONTEXT_MESSAGES

        # LangChain in-memory chat history store
        self.history = InMemoryChatMessageHistory()

        # Running summary used by the summarization strategy
        self.running_summary: str = ""
        self.summarized_up_to_index: int = 0

        self.created_at = time.time()
        self.updated_at = time.time()

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> HumanMessage:
        self.history.add_message(_CompatibleHumanMessage(content=content))
        self.updated_at = time.time()
        return self.history.messages[-1]

    def add_ai_message(self, content: str) -> AIMessage:
        self.history.add_message(_CompatibleAIMessage(content=content))
        self.updated_at = time.time()
        return self.history.messages[-1]

    def add_message(self, role: str, content: str) -> BaseMessage:
        """Add a message using the role names used by the existing API."""
        if role == "user":
            return self.add_user_message(content)
        if role == "model":
            return self.add_ai_message(content)
        raise ValueError("role must be 'user' or 'model'")

    def clear(self):
        """Reset conversation history and summary."""
        self.history.clear()
        self.running_summary = ""
        self.summarized_up_to_index = 0
        self.updated_at = time.time()

    @property
    def messages(self) -> List[BaseMessage]:
        """Return the full message list."""
        return self.history.messages

    # ------------------------------------------------------------------
    # Context window preparation
    # ------------------------------------------------------------------

    def get_context_messages(self) -> List[BaseMessage]:
        """
        Return the messages to send to the LLM after applying the
        configured context window strategy.
        """
        all_messages = self.history.messages

        if self.strategy == "sliding_window":
            # Keep the most recent max_messages, starting with a HumanMessage
            recent = all_messages[-self.max_messages:]
            while recent and not isinstance(recent[0], HumanMessage):
                recent = recent[1:]
            return recent

        # summarization strategy: return only unsummarised messages
        unsummarised = all_messages[self.summarized_up_to_index:]
        while unsummarised and not isinstance(unsummarised[0], HumanMessage):
            unsummarised = unsummarised[1:]
        return unsummarised

    def prepare_sliding_window_context(self) -> List[Dict[str, Any]]:
        """Return the sliding window in the legacy API dictionary format."""
        return [
            {
                "role": "user" if isinstance(message, HumanMessage) else "model",
                "parts": [message.content],
            }
            for message in self.get_context_messages()
        ]

    # ------------------------------------------------------------------
    # Summarization helpers
    # ------------------------------------------------------------------

    def needs_summarization(self) -> bool:
        unsummarized_count = len(self.messages) - self.summarized_up_to_index
        return unsummarized_count > self.max_messages

    def get_messages_to_summarize(self) -> List[BaseMessage]:
        total = len(self.messages)
        target_cutoff = total - self.max_messages
        if target_cutoff > self.summarized_up_to_index:
            return self.messages[self.summarized_up_to_index:target_cutoff]
        return []

    def commit_summary(self, new_summary: str, summarized_count: int):
        self.running_summary = new_summary.strip()
        self.summarized_up_to_index += summarized_count
        self.updated_at = time.time()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        total = len(self.messages)
        user_msgs = sum(1 for m in self.messages if isinstance(m, HumanMessage))
        ai_msgs = sum(1 for m in self.messages if isinstance(m, AIMessage))

        if self.strategy == "sliding_window":
            in_context = min(total, self.max_messages)
            pruned = max(0, total - self.max_messages)
        else:
            in_context = total - self.summarized_up_to_index
            pruned = self.summarized_up_to_index

        return {
            "conversation_id": self.conversation_id,
            "strategy": self.strategy,
            "max_messages": self.max_messages,
            "total_messages": total,
            "user_messages": user_msgs,
            "model_messages": ai_msgs,
            "messages_in_context": in_context,
            "messages_pruned_or_summarized": pruned,
            "has_running_summary": bool(self.running_summary),
            "running_summary_preview": (
                (self.running_summary[:120] + "...")
                if len(self.running_summary) > 120
                else self.running_summary
            ),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialise messages for the REST API response."""
        return [
            {
                "role": "user" if isinstance(m, HumanMessage) else "model",
                "content": m.content,
            }
            for m in self.messages
        ]


class ContextManager:
    """Manages active Conversation sessions in memory."""

    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}

    def get_or_create(
        self,
        conversation_id: str,
        strategy: Optional[str] = None,
        max_messages: Optional[int] = None,
    ) -> Conversation:
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = Conversation(
                conversation_id=conversation_id,
                strategy=strategy,
                max_messages=max_messages,
            )
        return self._conversations[conversation_id]

    def get(self, conversation_id: str) -> Optional[Conversation]:
        return self._conversations.get(conversation_id)

    def delete(self, conversation_id: str) -> bool:
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    def list_all(self) -> List[Dict[str, Any]]:
        return [conv.get_stats() for conv in self._conversations.values()]


# Global singleton
context_manager = ContextManager()
