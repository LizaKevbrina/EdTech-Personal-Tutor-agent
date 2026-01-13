"""
Conversation memory manager for maintaining dialogue context.
"""
from collections import deque
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """
    Manages conversation history with buffer window.
    
    Maintains a sliding window of recent messages and provides
    methods to retrieve context for LLM calls.
    """

    def __init__(
        self,
        buffer_size: int | None = None,
        session_id: str | None = None,
    ) -> None:
        """
        Initialize conversation memory.
        
        Args:
            buffer_size: Number of messages to keep
            session_id: Unique session identifier
        """
        self.buffer_size = buffer_size or settings.memory_buffer_size
        self.session_id = session_id or "default"
        self.messages: deque[BaseMessage] = deque(maxlen=self.buffer_size)

    def add_user_message(self, content: str) -> None:
        """
        Add user message to history.
        
        Args:
            content: Message content
        """
        message = HumanMessage(content=content)
        self.messages.append(message)

        logger.debug(
            "user_message_added",
            session_id=self.session_id,
            content_length=len(content),
            buffer_size=len(self.messages),
        )

    def add_ai_message(self, content: str) -> None:
        """
        Add AI message to history.
        
        Args:
            content: Message content
        """
        message = AIMessage(content=content)
        self.messages.append(message)

        logger.debug(
            "ai_message_added",
            session_id=self.session_id,
            content_length=len(content),
            buffer_size=len(self.messages),
        )

    def add_system_message(self, content: str) -> None:
        """
        Add system message to history.
        
        Args:
            content: Message content
        """
        message = SystemMessage(content=content)
        self.messages.append(message)

        logger.debug(
            "system_message_added",
            session_id=self.session_id,
            content_length=len(content),
        )

    def get_messages(self, last_n: int | None = None) -> list[BaseMessage]:
        """
        Get conversation messages.
        
        Args:
            last_n: Number of recent messages to get (None for all)
            
        Returns:
            List of messages
        """
        if last_n is None:
            return list(self.messages)

        return list(self.messages)[-last_n:]

    def get_context_string(self, last_n: int | None = None) -> str:
        """
        Get conversation context as formatted string.
        
        Args:
            last_n: Number of recent messages to include
            
        Returns:
            Formatted context string
        """
        messages = self.get_messages(last_n)

        if not messages:
            return "No conversation history yet."

        context_parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context_parts.append(f"Student: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Tutor: {msg.content}")
            elif isinstance(msg, SystemMessage):
                context_parts.append(f"System: {msg.content}")

        return "\n".join(context_parts)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

        logger.info("conversation_history_cleared", session_id=self.session_id)

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics of conversation.
        
        Returns:
            Summary dictionary
        """
        user_messages = sum(1 for m in self.messages if isinstance(m, HumanMessage))
        ai_messages = sum(1 for m in self.messages if isinstance(m, AIMessage))

        return {
            "session_id": self.session_id,
            "total_messages": len(self.messages),
            "user_messages": user_messages,
            "ai_messages": ai_messages,
            "buffer_size": self.buffer_size,
        }


class ConversationMemoryManager:
    """
    Manages multiple conversation sessions.
    """

    def __init__(self) -> None:
        """Initialize conversation memory manager."""
        self._sessions: dict[str, ConversationMemory] = {}

    def get_or_create_session(self, session_id: str) -> ConversationMemory:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationMemory instance
        """
        if session_id not in self._sessions:
            logger.info("creating_new_conversation_session", session_id=session_id)
            self._sessions[session_id] = ConversationMemory(session_id=session_id)

        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        """
        Delete a conversation session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("conversation_session_deleted", session_id=session_id)

    def get_active_sessions_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)


# Global instance
conversation_manager = ConversationMemoryManager()
