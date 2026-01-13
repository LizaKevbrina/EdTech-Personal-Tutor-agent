"""
Quiz endpoints for generating and validating quizzes.
"""
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.agent.tools.quiz_generator import Quiz, QuizQuestion, quiz_generator
from src.core.guards.input_validator import input_validator
from src.core.logging import get_logger
from src.core.rate_limiter import rate_limiter

logger = get_logger(__name__)

router = APIRouter()


class QuizGenerateRequest(BaseModel):
    """Quiz generation request."""

    topic: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Quiz topic",
    )
    count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of questions",
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium",
        description="Difficulty level",
    )
    student_id: str = Field(
        ...,
        description="Student identifier for rate limiting",
    )
    use_rag: bool = Field(
        default=True,
        description="Whether to use RAG for context",
    )


class QuizValidateRequest(BaseModel):
    """Quiz answer validation request."""

    question_text: str = Field(..., description="Question text")
    options: list[str] = Field(..., description="Answer options")
    correct_answer_index: int = Field(..., ge=0, description="Correct answer index")
    explanation: str = Field(..., description="Explanation")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="medium")
    student_answer_index: int = Field(
        ...,
        ge=0,
        le=3,
        description="Student's selected answer index",
    )


@router.post("/quiz/generate", response_model=Quiz)
async def generate_quiz(request: QuizGenerateRequest) -> Quiz:
    """
    Generate a quiz on a given topic.
    
    Args:
        request: Quiz generation request
        
    Returns:
        Generated quiz
        
    Raises:
        HTTPException: If generation fails
    """
    try:
        logger.info(
            "quiz_generation_requested",
            student_id=request.student_id,
            topic=request.topic,
            count=request.count,
        )

        # Rate limiting
        await rate_limiter.check_request_limit(request.student_id)

        # Input validation
        validated_topic = input_validator.validate(
            request.topic,
            field_name="topic",
        )

        # Generate quiz
        quiz = await quiz_generator.generate_quiz(
            topic=validated_topic,
            count=request.count,
            difficulty=request.difficulty,
            use_rag=request.use_rag,
        )

        logger.info(
            "quiz_generated",
            student_id=request.student_id,
            topic=validated_topic,
            questions_count=quiz.total_questions,
        )

        return quiz

    except Exception as e:
        logger.error(
            "quiz_generation_failed",
            student_id=request.student_id,
            topic=request.topic,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate quiz: {e}",
        )


@router.post("/quiz/validate")
async def validate_answer(request: QuizValidateRequest) -> dict[str, Any]:
    """
    Validate a student's quiz answer.
    
    Args:
        request: Validation request
        
    Returns:
        Validation result with feedback
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        # Create QuizQuestion from request
        question = QuizQuestion(
            question=request.question_text,
            options=request.options,
            correct_answer_index=request.correct_answer_index,
            explanation=request.explanation,
            difficulty=request.difficulty,
        )

        # Validate answer
        result = await quiz_generator.validate_answer(
            question=question,
            student_answer_index=request.student_answer_index,
        )

        logger.info(
            "answer_validated",
            correct=result["correct"],
            difficulty=question.difficulty,
        )

        return result

    except Exception as e:
        logger.error(
            "answer_validation_failed",
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate answer: {e}",
        )
