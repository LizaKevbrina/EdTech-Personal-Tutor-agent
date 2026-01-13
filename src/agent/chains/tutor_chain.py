"""
Main tutor agent chain that orchestrates all components.
"""
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.llm.llm_provider import llm_provider
from src.agent.memory.conversation_memory import ConversationMemory
from src.agent.memory.student_memory import student_memory
from src.agent.retrieval.rag_pipeline import create_rag_pipeline
from src.agent.tools import code_executor, progress_tracker, quiz_generator
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


TUTOR_SYSTEM_PROMPT = """You are an expert AI tutor for an EdTech platform. Your role is to help students learn effectively.

Your capabilities:
1. Answer questions about course topics using your knowledge and retrieved materials
2. Generate quizzes to test understanding
3. Execute Python code examples to demonstrate concepts
4. Track student progress and adapt to their learning pace

Guidelines:
- Be patient, encouraging, and supportive
- Explain concepts clearly with examples
- Ask follow-up questions to check understanding
- Adapt explanations to the student's level
- Use the student's learning history to personalize responses
- If unsure, retrieve information from the course materials

Available tools:
- quiz_generator: Create quizzes on any topic
- code_executor: Run Python code examples
- progress_tracker: Update student progress

Student Context:
{student_context}

Conversation History:
{conversation_history}

Retrieved Course Materials:
{rag_context}
"""


class TutorChain:
    """
    Main tutor agent chain orchestrating all components.
    
    Flow:
    1. Load student progress and conversation history
    2. Retrieve relevant course materials via RAG
    3. Generate response with LLM
    4. Optionally call tools (quiz, code execution)
    5. Update progress and conversation history
    """

    def __init__(self) -> None:
        """Initialize tutor chain."""
        self.rag_pipeline = create_rag_pipeline(settings.qdrant_collection_courses)

    async def process_message(
        self,
        student_id: str,
        message: str,
        session_id: str,
        conversation_memory: ConversationMemory,
    ) -> dict[str, Any]:
        """
        Process a student message and generate response.
        
        Args:
            student_id: Student identifier
            message: Student message
            session_id: Conversation session ID
            conversation_memory: Conversation memory instance
            
        Returns:
            Response dictionary with answer and metadata
        """
        try:
            logger.info(
                "processing_student_message",
                student_id=student_id,
                session_id=session_id,
                message_length=len(message),
            )

            # Step 1: Load student progress
            student_progress = await student_memory.load_student(student_id)
            student_context = student_memory.get_summary(student_progress)

            # Step 2: Retrieve relevant course materials
            rag_docs = await self.rag_pipeline.retrieve(
                query=message,
                enable_multi_query=True,
            )
            rag_context = self._format_rag_context(rag_docs)

            # Step 3: Get conversation history
            conversation_history = conversation_memory.get_context_string(last_n=10)

            # Step 4: Build prompt
            system_prompt = TUTOR_SYSTEM_PROMPT.format(
                student_context=student_context,
                conversation_history=conversation_history,
                rag_context=rag_context,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message),
            ]

            # Step 5: Generate response
            result = await llm_provider.generate(messages)
            ai_response = result.generations[0][0].text

            # Step 6: Update conversation memory
            conversation_memory.add_user_message(message)
            conversation_memory.add_ai_message(ai_response)

            # Step 7: Update progress (record question)
            await progress_tracker.record_question(
                student_id=student_id,
                topic=self._extract_topic(message),
            )

            # Step 8: Check if tool calling is needed
            tool_results = await self._handle_tool_calls(
                student_id=student_id,
                message=message,
                ai_response=ai_response,
            )

            logger.info(
                "student_message_processed",
                student_id=student_id,
                response_length=len(ai_response),
                tools_used=list(tool_results.keys()),
            )

            return {
                "response": ai_response,
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata,
                    }
                    for doc in rag_docs[:3]
                ],
                "tool_results": tool_results,
                "student_progress": {
                    "completed_topics": student_progress.completed_topics,
                    "total_questions": student_progress.total_questions,
                    "quiz_pass_rate": student_progress.quiz_pass_rate,
                },
            }

        except Exception as e:
            logger.error(
                "message_processing_failed",
                student_id=student_id,
                error=str(e),
            )
            raise

    def _format_rag_context(self, documents: list[Any]) -> str:
        """Format RAG documents into context string."""
        if not documents:
            return "No relevant course materials found."

        context_parts = []
        for i, doc in enumerate(documents[:5], 1):
            context_parts.append(
                f"[Source {i}]\n{doc.page_content}\n"
            )

        return "\n".join(context_parts)

    def _extract_topic(self, message: str) -> str:
        """Extract topic from message (simple heuristic)."""
        # In production, use NER or classification
        keywords = ["python", "recursion", "loops", "functions", "classes", "algorithm"]
        message_lower = message.lower()

        for keyword in keywords:
            if keyword in message_lower:
                return keyword
              return "general"

async def _handle_tool_calls(
    self,
    student_id: str,
    message: str,
    ai_response: str,
) -> dict[str, Any]:
    """
    Detect and handle tool calls based on message/response.
    
    Args:
        student_id: Student ID
        message: Original message
        ai_response: AI response
        
    Returns:
        Tool results dictionary
    """
    tool_results = {}
    message_lower = message.lower()

    # Detect quiz request
    if any(word in message_lower for word in ["quiz", "test", "practice"]):
        try:
            topic = self._extract_topic(message)
            quiz = await quiz_generator.generate_quiz(
                topic=topic,
                count=5,
                difficulty="medium",
            )
            tool_results["quiz"] = quiz.model_dump()
            logger.info("quiz_generated", student_id=student_id, topic=topic)
        except Exception as e:
            logger.warning("quiz_generation_skipped", error=str(e))

    # Detect code execution request
    if "```python" in message or "```" in message:
        try:
            # Extract code block
            code = self._extract_code_block(message)
            if code:
                exec_result = await code_executor.execute(code)
                tool_results["code_execution"] = exec_result
                logger.info("code_executed", student_id=student_id)
        except Exception as e:
            logger.warning("code_execution_skipped", error=str(e))

    return tool_results

def _extract_code_block(self, text: str) -> str | None:
    """Extract Python code block from markdown."""
    if "```python" in text:
        start = text.find("```python") + 9
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()

    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()

    return None

Global instance
tutor_chain = TutorChain()
