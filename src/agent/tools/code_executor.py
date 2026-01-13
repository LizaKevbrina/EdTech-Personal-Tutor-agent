"""
Safe code executor tool with sandboxing and timeout.
"""
import asyncio
import sys
from io import StringIO
from typing import Any

from src.core.config import settings
from src.core.exceptions import CodeExecutionError
from src.core.logging import get_logger
from src.core.metrics import tool_execution_success_rate

logger = get_logger(__name__)


class SafeCodeExecutor:
    """
    Executes Python code in a safe sandboxed environment.
    
    Security features:
    - Timeout limits
    - Restricted imports (no os, subprocess, etc.)
    - Captured stdout/stderr
    - Memory limits (conceptual - full implementation needs Docker)
    
    Note: For production, use Docker containers or cloud sandboxes.
    This is a basic implementation for demonstration.
    """

    # Allowed modules that are safe to import
    ALLOWED_MODULES = {
        "math",
        "random",
        "datetime",
        "json",
        "re",
        "string",
        "collections",
        "itertools",
        "functools",
    }

    # Forbidden patterns in code
    FORBIDDEN_PATTERNS = [
        "import os",
        "import subprocess",
        "import sys",
        "__import__",
        "eval(",
        "exec(",
        "compile(",
        "open(",
        "file(",
        "input(",
        "raw_input(",
    ]

    def __init__(self, timeout: int | None = None) -> None:
        """
        Initialize code executor.
        
        Args:
            timeout: Execution timeout in seconds
        """
        self.timeout = timeout or settings.code_executor_timeout

    async def execute(self, code: str) -> dict[str, Any]:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution result with stdout, stderr, and status
            
        Raises:
            CodeExecutionError: If execution fails or is unsafe
        """
        try:
            logger.info(
                "code_execution_started",
                code_length=len(code),
                timeout=self.timeout,
            )

            # Step 1: Validate code safety
            self._validate_code_safety(code)

            # Step 2: Execute with timeout
            result = await self._execute_with_timeout(code)

            # Update metrics
            tool_execution_success_rate.labels(tool_name="code_executor").set(1.0)

            logger.info(
                "code_execution_completed",
                success=result["success"],
                output_length=len(result["stdout"]),
            )

            return result

        except Exception as e:
            logger.error("code_execution_failed", error=str(e))

            # Update metrics
            tool_execution_success_rate.labels(tool_name="code_executor").set(0.0)

            raise CodeExecutionError(
                f"Code execution failed: {e}",
                details={"code_length": len(code)},
            ) from e

    def _validate_code_safety(self, code: str) -> None:
        """
        Validate that code is safe to execute.
        
        Args:
            code: Code to validate
            
        Raises:
            CodeExecutionError: If code contains forbidden patterns
        """
        code_lower = code.lower()

        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern.lower() in code_lower:
                raise CodeExecutionError(
                    f"Forbidden pattern detected: {pattern}",
                    details={"pattern": pattern},
                )

        logger.debug("code_safety_validated")

    async def _execute_with_timeout(self, code: str) -> dict[str, Any]:
        """
        Execute code with timeout.
        
        Args:
            code: Code to execute
            
        Returns:
            Execution result
            
        Raises:
            CodeExecutionError: If execution fails or times out
        """
        try:
            # Run in thread pool to avoid blocking
            result = await asyncio.wait_for(
                asyncio.to_thread(self._execute_sync, code),
                timeout=self.timeout,
            )

            return result

        except asyncio.TimeoutError:
            raise CodeExecutionError(
                f"Code execution timed out after {self.timeout}s",
                details={"timeout": self.timeout},
            )

    def _execute_sync(self, code: str) -> dict[str, Any]:
        """
        Synchronous code execution (runs in thread).
        
        Args:
            code: Code to execute
            
        Returns:
            Execution result
        """
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        success = False
        error_message = ""

        try:
            # Create restricted globals
            restricted_globals = {
                "__builtins__": {
                    "print": print,
                    "range": range,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                },
            }

            # Execute code
            exec(code, restricted_globals)
            success = True

        except Exception as e:
            error_message = str(e)
            logger.warning("code_execution_error", error=error_message)

        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return {
            "success": success,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "error": error_message,
        }

    def format_result(self, result: dict[str, Any]) -> str:
        """
        Format execution result for display.
        
        Args:
            result: Execution result
            
        Returns:
            Formatted string
        """
        if result["success"]:
            output = result["stdout"]
            if output:
                return f"✓ Execution successful:\n{output}"
            return "✓ Execution successful (no output)"

        error = result["error"] or result["stderr"]
        return f"✗ Execution failed:\n{error}"


# Global instance
code_executor = SafeCodeExecutor()
