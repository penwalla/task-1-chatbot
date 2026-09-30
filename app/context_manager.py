"""
Context window management module.
Implements:
1. Sliding Window Truncation (keeps last N messages)
2. Progressive Running Summarization (compresses older messages and prepends to context)
"""

import time
from typing import List, Dict, Any, Optional
from app.config import settings

class Message:
    def __init__(self, role: str, content: str, timestamp: Optional[float] = None):
        self.role = role  # 'user' or 'model'
        self.content = content
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    def to_gemini_format(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "parts": [self.content],
        }


class Conversation:
    def __init__(self, conversation_id: str, strategy: Optional[str] = None, max_messages: Optional[int] = None):
        self.conversation_id = conversation_id
        self.strategy = (strategy or settings.CONTEXT_STRATEGY).lower()
        self.max_messages = max_messages or settings.MAX_CONTEXT_MESSAGES
        self.messages: List[Message] = []
        self.running_summary: str = ""
        self.summarized_up_to_index: int = 0
        self.created_at = time.time()
        self.updated_at = time.time()

    def add_message(self, role: str, content: str) -> Message:
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg

    def clear(self):
        """Reset conversation history and summary."""
        self.messages = []
        self.running_summary = ""
        self.summarized_up_to_index = 0
        self.updated_at = time.time()

    def get_stats(self) -> Dict[str, Any]:
        total_messages = len(self.messages)
        user_messages = sum(1 for m in self.messages if m.role == "user")
        model_messages = sum(1 for m in self.messages if m.role == "model")
        
        # Calculate messages in active window
        if self.strategy == "sliding_window":
            in_context = min(total_messages, self.max_messages)
            pruned = max(0, total_messages - self.max_messages)
        else: # summarization
            # Messages that are still in active unsummarized window
            unsummarized = total_messages - self.summarized_up_to_index
            in_context = unsummarized
            pruned = self.summarized_up_to_index

        return {
            "conversation_id": self.conversation_id,
            "strategy": self.strategy,
            "max_messages": self.max_messages,
            "total_messages": total_messages,
            "user_messages": user_messages,
            "model_messages": model_messages,
            "messages_in_context": in_context,
            "messages_pruned_or_summarized": pruned,
            "has_running_summary": bool(self.running_summary),
            "running_summary_preview": (self.running_summary[:120] + "...") if len(self.running_summary) > 120 else self.running_summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def prepare_sliding_window_context(self) -> List[Dict[str, Any]]:
        """
        Sliding Window Truncation Strategy:
        Keeps only the most recent `max_messages` messages.
        Ensures the first message in the window is from 'user' to satisfy API constraints.
        """
        if not self.messages:
            return []

        recent_msgs = self.messages[-self.max_messages:]
        
        # Gemini expects conversation history to start with a 'user' turn
        while recent_msgs and recent_msgs[0].role != "user":
            recent_msgs = recent_msgs[1:]

        return [msg.to_gemini_format() for msg in recent_msgs]

    def needs_summarization(self) -> bool:
        """Check if unsummarized messages exceed max_messages limit."""
        unsummarized_count = len(self.messages) - self.summarized_up_to_index
        return unsummarized_count > self.max_messages

    def get_messages_to_summarize(self) -> List[Message]:
        """
        Return the batch of older messages that should be compressed into running summary,
        leaving the latest `max_messages` intact in the active window.
        """
        total = len(self.messages)
        target_cutoff = total - self.max_messages
        if target_cutoff > self.summarized_up_to_index:
            return self.messages[self.summarized_up_to_index:target_cutoff]
        return []

    def commit_summary(self, new_summary: str, summarized_count: int):
        """Update the running summary and advance the index."""
        self.running_summary = new_summary.strip()
        self.summarized_up_to_index += summarized_count
        self.updated_at = time.time()

    def prepare_summarized_context(self) -> List[Dict[str, Any]]:
        """
        Summarization Strategy:
        Prepends the running summary (if present) to the recent messages window.
        """
        recent_msgs = self.messages[self.summarized_up_to_index:]
        
        # Ensure starts with user
        while recent_msgs and recent_msgs[0].role != "user":
            recent_msgs = recent_msgs[1:]

        formatted_contents = [msg.to_gemini_format() for msg in recent_msgs]
        return formatted_contents


class ContextManager:
    """Manages active conversations in memory."""
    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}

    def get_or_create(
        self,
        conversation_id: str,
        strategy: Optional[str] = None,
        max_messages: Optional[int] = None
    ) -> Conversation:
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = Conversation(
                conversation_id=conversation_id,
                strategy=strategy,
                max_messages=max_messages
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


# Global context manager singleton
context_manager = ContextManager()
