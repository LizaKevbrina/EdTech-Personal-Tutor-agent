"""
Progress tracker tool for updating student learning progress.
"""
from typing import Any

from src.agent.memory.student_memory import student_memory
from src.core.exceptions import ProgressUpdateError
from src.core.logging import get_logger
from src.core.metrics import (
    difficult_topics_total,
    quiz_completions_total,
    student_questions_total,
)

logger = get_logger(__name__)


class ProgressTracker:
    """
    Tracks and updates student learning progress.
    
    Integrates with student memory and updates metrics.
    """

    async def record_question(
        self,
        student_id: str,
        topic: str,
        difficulty: str = "medium",
    ) -> dict[str, Any]:
        """
        Record that a student asked a question.
        
        Args:
            student_id: Student identifier
            topic: Question topic
            difficulty: Question difficulty
            
        Returns:
            Updated progress summary
        """
        try:
            logger.info(
                "recording_question",
                student_id=student_id,
                topic=topic,
            )

            # Load student progress
            progress = await student_memory.load_student(student_id)

            # Update question count
            progress.total_questions += 1

            # Save updated progress
            await student_memory.save_student(progress)

            # Update metrics
            student_questions_total.labels(
                topic=topic,
                difficulty=difficulty,
            ).inc()

            return {
                "student_id": student_id,
                "total_questions": progress.total_questions,
                "action": "question_recorded",
            }

        except Exception as e:
            logger.error(
                "record_question_failed",
                student_id=student_id,
                error=str(e),
            )
            raise ProgressUpdateError(
                f"Failed to record question: {e}",
                details={"student_id": student_id},
            ) from e

    async def complete_topic(
        self,
        student_id: str,
        topic: str,
    ) -> dict[str, Any]:
        """
        Mark a topic as completed.
        
        Args:
            student_id: Student identifier
            topic: Topic name
            
        Returns:
            Updated progress summary
        """
        try:
            logger.info(
                "completing_topic",
                student_id=student_id,
                topic=topic,
            )

            progress = await student_memory.mark_topic_completed(student_id, topic)

            return {
                "student_id": student_id,
                "topic": topic,
                "completed_topics": progress.completed_topics,
                "action": "topic_completed",
            }

        except Exception as e:
            logger.error(
                "complete_topic_failed",
                student_id=student_id,
                error=str(e),
            )
            raise ProgressUpdateError(
                f"Failed to complete topic: {e}",
                details={"student_id": student_id, "topic": topic},
            ) from e

    async def mark_difficult(
        self,
        student_id: str,
        topic: str,
    ) -> dict[str, Any]:
        """
        Mark a topic as difficult for student.
        
        Args:
            student_id: Student identifier
            topic: Topic name
            
        Returns:
            Updated progress summary
        """
        try:
            logger.info(
                "marking_topic_difficult",
                student_id=student_id,
                topic=topic,
            )

            progress = await student_memory.mark_topic_difficult(student_id, topic)

            # Update metrics
            difficult_topics_total.labels(topic=topic).inc()

            return {
                "student_id": student_id,
                "topic": topic,
                "difficulty_count": progress.difficult_topics.get(topic, 0),
                "action": "topic_marked_difficult",
            }

        except Exception as e:
            logger.error(
                "mark_difficult_failed",
                student_id=student_id,
                error=str(e),
            )
            raise ProgressUpdateError(
                f"Failed to mark topic difficult: {e}",
                details={"student_id": student_id, "topic": topic},
            ) from e

    async def record_quiz_result(
        self,
        student_id: str,
        topic: str,
        passed: bool,
        score: float,
    ) -> dict[str, Any]:
        """
        Record quiz completion result.
        
        Args:
            student_id: Student identifier
            topic: Quiz topic
            passed: Whether quiz was passed
            score: Quiz score (0.0 - 1.0)
            
        Returns:
            Updated progress summary
        """
        try:
            logger.info(
                "recording_quiz_result",
                student_id=student_id,
                topic=topic,
                passed=passed,
                score=score,
            )

            progress = await student_memory.update_quiz_stats(student_id, passed)

            # Update metrics
            result = "passed" if passed else "failed"
            quiz_completions_total.labels(topic=topic, result=result).inc()

            return {
                "student_id": student_id,
                "topic": topic,
                "passed": passed,
                "score": score,
                "total_quizzes": progress.total_quizzes,
                "pass_rate": progress.quiz_pass_rate,
                "action": "quiz_recorded",
            }

        except Exception as e:
            logger.error(
                "record_quiz_failed",
                student_id=student_id,
                error=str(e),
            )
            raise ProgressUpdateError(
                f"Failed to record quiz result: {e}",
                details={"student_id": student_id, "topic": topic},
            ) from e

    async def get_progress_summary(
        self,
        student_id: str,
    ) -> dict[str, Any]:
        """
        Get comprehensive progress summary.
        
        Args:
            student_id: Student identifier
            
        Returns:
            Progress summary
        """
        try:
            progress = await student_memory.load_student(student_id)

            summary = {
                "student_id": student_id,
                "completed_topics": progress.completed_topics,
                "difficult_topics": progress.difficult_topics,
                "learning_pace": progress.learning_pace,
                "preferred_format": progress.preferred_format,
                "total_questions": progress.total_questions,
                "total_quizzes": progress.total_quizzes,
                "quiz_pass_rate": progress.quiz_pass_rate,
                "last_active": progress.last_active.isoformat(),
            }

            return summary

        except Exception as e:
            logger.error(
                "get_progress_summary_failed",
                student_id=student_id,
                error=str(e),
            )
            raise ProgressUpdateError(
                f"Failed to get progress summary: {e}",
                details={"student_id": student_id},
            ) from e

    async def suggest_next_topic(
        self,
        student_id: str,
    ) -> str:
        """
        Suggest next topic based on progress.
        
        Args:
            student_id: Student identifier
            
        Returns:
            Suggested topic
        """
        try:
            progress = await student_memory.load_student(student_id)

            # Simple logic: suggest reviewing difficult topics
            if progress.difficult_topics:
                # Return most difficult topic
                most_difficult = max(
                    progress.difficult_topics.items(),
                    key=lambda x: x[1],
                )
                return f"Review: {most_difficult[0]} (struggled {most_difficult[1]} times)"

            # Or suggest continuing with course progression
            return "Continue with next course topic"

        except Exception as e:
            logger.warning(
                "suggest_topic_failed",
                student_id=student_id,
                error=str(e),
            )
            return "Continue learning at your own pace"


# Global instance
progress_tracker = ProgressTracker()
