"""
Quiz generator tool that creates educational quizzes based on topics.
"""
import json
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.llm.llm_provider import llm_provider
from src.agent.retrieval.rag_pipeline import create_rag_pipeline
from src.core.config import settings
from src.core.exceptions import QuizGenerationError
from src.core.logging import get_logger
from src.core.metrics import tool_execution_success_rate

logger = get_logger(__name__)


class QuizQuestion(BaseModel):
    """Single quiz question model."""

    question: str = Field(..., description="The question text")
    options: list[str] = Field(..., description="List of 4 answer options")
    correct_answer_index: int = Field(
        ..., ge=0, le=3, description="Index of correct answer (0-3)"
    )
    explanation: str = Field(..., description="Explanation of the correct answer")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium", description="Question difficulty level"
    )


class Quiz(BaseModel):
    """Quiz model containing multiple questions."""

    topic: str = Field(..., description="Quiz topic")
    questions: list[QuizQuestion] = Field(..., description="List of questions")
    total_questions: int = Field(..., description="Total number of questions")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium", description="Overall quiz difficulty"
    )


QUIZ_GENERATION_SYSTEM_PROMPT = """You are an expert educational quiz generator for an EdTech platform.

Your task is to generate {count} high-quality multiple-choice questions about: {topic}

Requirements:
1. Each question must have exactly 4 options
2. Only ONE option should be correct
3. Questions should be clear, unambiguous, and educational
4. Difficulty level: {difficulty}
5. Include explanations for correct answers
6. Make distractors (wrong answers) plausible but clearly incorrect
7. Cover different aspects of the topic
8. Avoid trick questions or overly obscure details

Return ONLY valid JSON matching this exact structure:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer_index": 0,
      "explanation": "Explanation of why this is correct",
      "difficulty": "medium"
    }}
  ]
}}

DO NOT include any markdown formatting, backticks, or additional text. Return ONLY the raw JSON object."""


class QuizGenerator:
    """
    Generates educational quizzes using RAG and LLM.
    
    Features:
    - RAG-based question generation from course content
    - Multiple difficulty levels
    - JSON structured output with validation
    - Error handling and fallbacks
    """

    def __init__(self) -> None:
        """Initialize quiz generator."""
        self.rag_pipeline = create_rag_pipeline(settings.qdrant_collection_courses)

    async def generate_quiz(
        self,
        topic: str,
        count: int = 5,
        difficulty: Literal["easy", "medium", "hard"] = "medium",
        use_rag: bool = True,
    ) -> Quiz:
        """
        Generate a quiz on a given topic.
        
        Args:
            topic: Topic for the quiz
            count: Number of questions to generate
            difficulty: Difficulty level
            use_rag: Whether to use RAG for context
            
        Returns:
            Quiz object
            
        Raises:
            QuizGenerationError: If generation fails
        """
        try:
            logger.info(
                "quiz_generation_started",
                topic=topic,
                count=count,
                difficulty=difficulty,
            )

            # Step 1: Get relevant context via RAG (if enabled)
            context = ""
            if use_rag:
                context = await self._get_topic_context(topic)

            # Step 2: Generate quiz questions
            questions = await self._generate_questions(
                topic=topic,
                count=count,
                difficulty=difficulty,
                context=context,
            )

            # Step 3: Create Quiz object
            quiz = Quiz(
                topic=topic,
                questions=questions,
                total_questions=len(questions),
                difficulty=difficulty,
            )

            # Update metrics
            tool_execution_success_rate.labels(tool_name="quiz_generator").set(1.0)

            logger.info(
                "quiz_generation_completed",
                topic=topic,
                questions_count=len(questions),
            )

            return quiz

        except Exception as e:
            logger.error(
                "quiz_generation_failed",
                topic=topic,
                error=str(e),
            )

            # Update metrics
            tool_execution_success_rate.labels(tool_name="quiz_generator").set(0.0)

            raise QuizGenerationError(
                f"Failed to generate quiz for topic '{topic}': {e}",
                details={"topic": topic, "count": count},
            ) from e

    async def _get_topic_context(self, topic: str) -> str:
        """
        Get relevant context for topic via RAG.
        
        Args:
            topic: Topic to search
            
        Returns:
            Concatenated context from retrieved documents
        """
        try:
            # Retrieve relevant documents
            documents = await self.rag_pipeline.retrieve(
                query=f"Explain {topic} concepts and examples",
                enable_multi_query=False,
            )

            if not documents:
                logger.warning("no_context_found_for_topic", topic=topic)
                return ""

            # Concatenate document contents
            context = "\n\n".join(doc.page_content for doc in documents[:3])

            logger.debug(
                "topic_context_retrieved",
                topic=topic,
                docs_count=len(documents),
                context_length=len(context),
            )

            return context

        except Exception as e:
            logger.warning(
                "context_retrieval_failed",
                topic=topic,
                error=str(e),
            )
            # Continue without context
            return ""

    async def _generate_questions(
        self,
        topic: str,
        count: int,
        difficulty: str,
        context: str,
    ) -> list[QuizQuestion]:
        """
        Generate quiz questions using LLM.
        
        Args:
            topic: Quiz topic
            count: Number of questions
            difficulty: Difficulty level
            context: Optional context from RAG
            
        Returns:
            List of QuizQuestion objects
        """
        system_prompt = QUIZ_GENERATION_SYSTEM_PROMPT.format(
            topic=topic,
            count=count,
            difficulty=difficulty,
        )

        user_message = f"Topic: {topic}\nDifficulty: {difficulty}\n"
        if context:
            user_message += f"\nContext:\n{context}\n"
        user_message += f"\nGenerate {count} questions."

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        # Generate with retry on fallback model if needed
        result = await llm_provider.generate(messages, use_fallback=False)
        response_text = result.generations[0][0].text.strip()

        # Parse JSON response
        questions = self._parse_quiz_response(response_text, count)

        return questions

    def _parse_quiz_response(self, response: str, expected_count: int) -> list[QuizQuestion]:
        """
        Parse LLM response into QuizQuestion objects.
        
        Args:
            response: LLM response text
            expected_count: Expected number of questions
            
        Returns:
            List of validated QuizQuestion objects
            
        Raises:
            QuizGenerationError: If parsing fails
        """
        try:
            # Remove markdown code blocks if present
            response = response.replace("```json", "").replace("```", "").strip()

            # Parse JSON
            data = json.loads(response)

            # Validate structure
            if "questions" not in data:
                raise QuizGenerationError("Response missing 'questions' field")

            questions_data = data["questions"]

            if len(questions_data) < expected_count:
                logger.warning(
                    "fewer_questions_generated",
                    expected=expected_count,
                    actual=len(questions_data),
                )

            # Parse and validate each question
            questions = []
            for q_data in questions_data:
                try:
                    question = QuizQuestion(**q_data)
                    questions.append(question)
                except Exception as e:
                    logger.warning(
                        "invalid_question_skipped",
                        error=str(e),
                        question_data=q_data,
                    )
                    continue

            if not questions:
                raise QuizGenerationError("No valid questions parsed")

            return questions

        except json.JSONDecodeError as e:
            logger.error("json_parsing_failed", response=response[:200], error=str(e))
            raise QuizGenerationError(f"Failed to parse JSON response: {e}") from e

    async def validate_answer(
        self,
        question: QuizQuestion,
        student_answer_index: int,
    ) -> dict[str, Any]:
        """
        Validate a student's answer.
        
        Args:
            question: The quiz question
            student_answer_index: Index of student's selected answer
            
        Returns:
            Validation result with feedback
        """
        is_correct = student_answer_index == question.correct_answer_index

        result = {
            "correct": is_correct,
            "student_answer": question.options[student_answer_index],
            "correct_answer": question.options[question.correct_answer_index],
            "explanation": question.explanation,
        }

        logger.info(
            "answer_validated",
            correct=is_correct,
            difficulty=question.difficulty,
        )

        return result


# Global instance
quiz_generator = QuizGenerator()
