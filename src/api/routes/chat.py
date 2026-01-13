"""
Chat endpoints for conversational interactions.
"""
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.agent.chains.tutor_chain import tutor_chain
from src.agent.memory.conversation_memory import conversation_manager
from src.core.guards.input_validator import input_validator
from src.core.logging import get_logger
from src.core.metrics import active_students_total
from src.core.rate_limiter import rate_limiter

logger = get_logger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Student message",
    )
    student_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique student identifier",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity",
    )


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str = Field(..., description="AI tutor response")
    session_id: str = Field(..., description="Session ID")
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Source documents used",
    )
    tool_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Results from tool executions",
    )
    student_progress: dict[str, Any] = Field(
        default_factory=dict,
        description="Current student progress",
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat with the AI tutor.
    
    Args:
        request: Chat request
        
    Returns:
        Chat response with AI answer
        
    Raises:
        HTTPException: If chat processing fails
    """
    try:
        logger.info(
            "chat_request_received",
            student_id=request.student_id,
            message_length=len(request.message),
        )

        # Step 1: Rate limiting check
        await rate_limiter.check_request_limit(request.student_id)

        # Step 2: Input validation
        validated_message = input_validator.validate(
            request.message,
            field_name="message",
        )

        # Step 3: Get or create session
        session_id = request.session_id or f"{request.student_id}_default"
        conversation_memory = conversation_manager.get_or_create_session(session_id)

        # Step 4: Process message
        result = await tutor_chain.process_message(
            student_id=request.student_id,
            message=validated_message,
            session_id=session_id,
            conversation_memory=conversation_memory,
        )

        # Step 5: Check token budget (approximate)
        estimated_tokens = (
            len(validated_message) // 4 + len(result["response"]) // 4
        )
        await rate_limiter.check_token_limit(request.student_id, estimated_tokens)

        # Update active students metric
        active_students_total.set(conversation_manager.get_active_sessions_count())

        logger.info(
            "chat_request_completed",
            student_id=request.student_id,
            session_id=session_id,
        )

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            sources=result["sources"],
            tool_results=result["tool_results"],
            student_progress=result["student_progress"],
        )

    except Exception as e:
        logger.error(
            "chat_request_failed",
            student_id=request.student_id,
            error=str(e),
        )
        raise


@router.delete("/chat/session/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """
    Delete a conversation session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success message
    """
    try:
        conversation_manager.delete_session(session_id)

        logger.info("session_deleted", session_id=session_id)

        return {
            "message": f"Session {session_id} deleted successfully",
        }

    except Exception as e:
        logger.error(
            "session_deletion_failed",
            session_id=session_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {e}",
        )


@router.get("/chat/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    last_n: int | None = None,
) -> dict[str, Any]:
    """
    Get conversation history for a session.
    
    Args:
        session_id: Session identifier
        last_n: Number of recent messages (optional)
        
    Returns:
        Conversation history
    """
    try:
        conversation_memory = conversation_manager.get_or_create_session(session_id)

        messages = conversation_memory.get_messages(last_n=last_n)

        # Format messages
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.type,
                "content": msg.content,
            })

        summary = conversation_memory.get_summary()

        return {
            "session_id": session_id,
            "messages": formatted_messages,
            "summary": summary,
        }

    except Exception as e:
        logger.error(
            "get_history_failed",
            session_id=session_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {e}",
        )
