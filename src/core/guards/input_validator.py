"""
Input validation and sanitization with PII detection and prompt injection protection.
"""
import re
from typing import Any

from src.core.exceptions import PIIDetectedError, PromptInjectionError, ValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


class InputValidator:
    """
    Validates and sanitizes user input for security.
    
    Features:
    - PII detection (email, phone, SSN, credit cards)
    - Prompt injection detection
    - Length limits
    - Content moderation
    """

    # PII patterns
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b')
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r'ignore\s+(?:previous|above|all)\s+(?:instructions?|prompts?)',
        r'disregard\s+(?:previous|above|all)',
        r'you\s+are\s+now',
        r'new\s+instructions?',
        r'system\s*:\s*',
        r'assistant\s*:\s*',
        r'<\|im_start\|>',
        r'<\|im_end\|>',
        r'\[INST\]',
        r'\[/INST\]',
    ]
    
    # Suspicious content patterns
    SUSPICIOUS_PATTERNS = [
        r'</?\s*script',
        r'javascript\s*:',
        r'on(?:load|error|click)\s*=',
        r'<\s*iframe',
    ]

    def __init__(
        self,
        max_length: int = 10000,
        enable_pii_detection: bool = True,
        enable_injection_detection: bool = True,
    ) -> None:
        """
        Initialize input validator.
        
        Args:
            max_length: Maximum input length
            enable_pii_detection: Whether to check for PII
            enable_injection_detection: Whether to check for prompt injection
        """
        self.max_length = max_length
        self.enable_pii_detection = enable_pii_detection
        self.enable_injection_detection = enable_injection_detection

    def validate(self, text: str, field_name: str = "input") -> str:
        """
        Validate and sanitize input text.
        
        Args:
            text: Input text to validate
            field_name: Name of the field (for error messages)
            
        Returns:
            Sanitized text
            
        Raises:
            ValidationError: If validation fails
            PIIDetectedError: If PII is detected
            PromptInjectionError: If injection attempt detected
        """
        logger.debug(
            "validating_input",
            field_name=field_name,
            length=len(text),
        )

        # Check if empty
        if not text or not text.strip():
            raise ValidationError(
                f"{field_name} cannot be empty",
                details={"field": field_name},
            )

        # Check length
        if len(text) > self.max_length:
            raise ValidationError(
                f"{field_name} exceeds maximum length of {self.max_length} characters",
                details={
                    "field": field_name,
                    "length": len(text),
                    "max_length": self.max_length,
                },
            )

        # Check for PII
        if self.enable_pii_detection:
            self._check_pii(text, field_name)

        # Check for prompt injection
        if self.enable_injection_detection:
            self._check_injection(text, field_name)

        # Check for suspicious content
        self._check_suspicious_content(text, field_name)

        # Sanitize
        sanitized = self._sanitize(text)

        logger.debug(
            "input_validated",
            field_name=field_name,
            original_length=len(text),
            sanitized_length=len(sanitized),
        )

        return sanitized

    def _check_pii(self, text: str, field_name: str) -> None:
        """
        Check for PII in text.
        
        Args:
            text: Text to check
            field_name: Field name
            
        Raises:
            PIIDetectedError: If PII found
        """
        pii_found = []

        # Check email
        if self.EMAIL_PATTERN.search(text):
            pii_found.append("email")

        # Check phone
        if self.PHONE_PATTERN.search(text):
            pii_found.append("phone")

        # Check SSN
        if self.SSN_PATTERN.search(text):
            pii_found.append("ssn")

        # Check credit card
        if self.CREDIT_CARD_PATTERN.search(text):
            pii_found.append("credit_card")

        if pii_found:
            logger.warning(
                "pii_detected",
                field_name=field_name,
                pii_types=pii_found,
            )
            raise PIIDetectedError(
                f"PII detected in {field_name}: {', '.join(pii_found)}",
                details={
                    "field": field_name,
                    "pii_types": pii_found,
                },
            )

    def _check_injection(self, text: str, field_name: str) -> None:
        """
        Check for prompt injection attempts.
        
        Args:
            text: Text to check
            field_name: Field name
            
        Raises:
            PromptInjectionError: If injection detected
        """
        text_lower = text.lower()

        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(
                    "prompt_injection_detected",
                    field_name=field_name,
                    pattern=pattern,
                )
                raise PromptInjectionError(
                    f"Potential prompt injection detected in {field_name}",
                    details={
                        "field": field_name,
                        "pattern": pattern,
                    },
                )

    def _check_suspicious_content(self, text: str, field_name: str) -> None:
        """
        Check for suspicious content patterns.
        
        Args:
            text: Text to check
            field_name: Field name
            
        Raises:
            ValidationError: If suspicious content found
        """
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(
                    "suspicious_content_detected",
                    field_name=field_name,
                    pattern=pattern,
                )
                raise ValidationError(
                    f"Suspicious content detected in {field_name}",
                    details={
                        "field": field_name,
                        "reason": "suspicious_pattern",
                    },
                )

    def _sanitize(self, text: str) -> str:
        """
        Sanitize input text.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Strip leading/trailing whitespace
        sanitized = text.strip()

        # Normalize whitespace (multiple spaces to single)
        sanitized = re.sub(r'\s+', ' ', sanitized)

        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')

        return sanitized

    def anonymize_pii(self, text: str) -> str:
        """
        Anonymize PII in text for logging.
        
        Args:
            text: Text with potential PII
            
        Returns:
            Text with PII masked
        """
        anonymized = text

        # Mask emails
        anonymized = self.EMAIL_PATTERN.sub('[EMAIL]', anonymized)

        # Mask phones
        anonymized = self.PHONE_PATTERN.sub('[PHONE]', anonymized)

        # Mask SSN
        anonymized = self.SSN_PATTERN.sub('[SSN]', anonymized)

        # Mask credit cards
        anonymized = self.CREDIT_CARD_PATTERN.sub('[CREDIT_CARD]', anonymized)

        return anonymized


# Global instance
input_validator = InputValidator()
