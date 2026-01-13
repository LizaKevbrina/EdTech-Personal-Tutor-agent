"""
Student progress endpoints.
"""
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.agent.tools.progress_tracker import progress_tracker
from src.core.logging import get_logger
from src.core.rate_limiter import rate_limiter

logger = get_logger(__name__)

router = APIRouter()


class MarkTopicRequest(BaseModel):
    """Mark topic request."""

    student_id: str = Field(..., description="Student identifier")
    topic: str = Field(..., min_length=1, max_length=200, description="Topic name")


class QuizResultRequest(BaseModel):
    """Quiz result request."""

    student_id: str = Field(..., description="Student identifier")
    topic: str = Field(..., description="Quiz topic")
    passed: bool = Field(..., description="Whether quiz was passed")
    score: float = Field(..., ge=0.0, le=1.0, description="Quiz score (0.0-1.0)")


@router.get("/progress/{student_id}")
async def get_progress(student_id: str) -> dict[str, Any]:
    """
    Get student progress summary.
    
    Args:
        student_id: Student identifier
        
    Returns:
        Progress summary
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        logger.info("progress_requested", student_id=student_id)

        # Rate limiting
        await rate_limiter.check_request_limit(student_id)

        # Get progress
        progress = await progress_tracker.get_progress_summary(student_id)

        logger.info("progress_retrieved", student_id=student_id)

        return progress

    except Exception as e:
        logger.error(
            "progress_retrieval_failed",
            student_id=student_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve progress: {e}",
        )


@router.post("/progress/topic/complete")
async def complete_topic(request: MarkTopicRequest) -> dict[str, Any]:
    """
    Mark a topic as completed.
    
    Args:
        request: Mark topic request
        
    Returns:
        Updated progress
        
    Raises:
        HTTPException: If update fails
    """
    try:
        logger.info(
            "topic_completion_requested",
            student_id=request.student_id,
            topic=request.topic,
        )

        result = await progress_tracker.complete_topic(
            student_id=request.student_id,
            topic=request.topic,
        )

        logger.info(
            "topic_completed",
            student_id=request.student_id,
            topic=request.topic,
        )

        return result

    except Exception as e:
        logger.error(
            "topic_completion_failed",
            student_id=request.student_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete topic: {e}",
        )


@router.post("/progress/topic/difficult")
async def mark_difficult(request: MarkTopicRequest) -> dict[str, Any]:
    """
    Mark a topic as difficult.
    
    Args:
        request: Mark topic request
        
    Returns:
        Updated progress
        
    Raises:
        HTTPException: If update fails
    """
    try:
        logger.info(
            "mark_difficult_requested",
            student_id=request.student_id,
            topic=request.topic,
        )

        result = await progress_tracker.mark_difficult(
            student_id=request.student_id,
            topic=request.topic,
        )

        logger.info(
            "topic_marked_difficult",
            student_id=request.student_id,
            topic=request.topic,
        )

        return result

    except Exception as e:
        logger.error(
            "mark_difficult_failed",
            student_id=request.student_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mark topic difficult: {e}",
        )


@router.post("/progress/quiz/result")
async def record_quiz_result(request: QuizResultRequest) -> dict[str, Any]:
    """
    Record quiz completion result.
    
    Args:
        request: Quiz result request
        
    Returns:
        Updated progress
        
    Raises:
        HTTPException: If recording fails
    """
    try:
        logger.info(
            "quiz_result_recording_requested",
            student_id=request.student_id,
            topic=request.topic,
            passed=request.passed,
        )

        result = await progress_tracker.record_quiz_result(
            student_id=request.student_id,
            topic=request.topic,
            passed=request.passed,
            score=request.score,
        )

        logger.info(
            "quiz_result_recorded",
            student_id=request.student_id,
            topic=request.topic,
            passed=request.passed,
        )

        return result

    except Exception as e:
        logger.error(
            "quiz_result_recording_failed",
            student_id=request.student_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record quiz result: {e}",
        )


@router.get("/progress/{student_id}/suggest-next")
async def suggest_next_topic(student_id: str) -> dict[str, str]:
    """
    Get suggested next topic for student.
    
    Args:
        student_id: Student identifier
        
    Returns:
        Suggested topic
        
    Raises:
        HTTPException: If suggestion fails
    """
    try:
        logger.info("next_topic_suggestion_requested", student_id=student_id)

        suggestion = await progress_tracker.suggest_next_topic(student_id)

        logger.info("next_topic_suggested", student_id=student_id)

        return {
            "student_id": student_id,
            "suggestion": suggestion,
        }

    except Exception as e:
        logger.error(
            "topic_suggestion_failed",
            student_id=student_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to suggest topic: {e}",
        )


@router.get("/progress/{student_id}/usage")
async def get_usage_stats(student_id: str) -> dict[str, Any]:
    """
    Get rate limiting usage statistics for student.
    
    Args:
        student_id: Student identifier
        
    Returns:
        Usage statistics
    """
    try:
        stats = await rate_limiter.get_usage_stats(student_id)
        return stats

    except Exception as e:
        logger.error(
            "usage_stats_failed",
            student_id=student_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage stats: {e}",
        )
