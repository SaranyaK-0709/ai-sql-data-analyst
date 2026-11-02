"""
LLM Service Module
Provides a resilient, modular integration with Groq API using the official Groq Python SDK.
Includes error handling, model fallback options, and mock support for automated testing.
"""

from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
SUPPORTED_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]


class LLMServiceError(Exception):
    """Custom exception raised for LLM API invocation failures."""
    pass


def get_available_groq_models(api_key: Optional[str] = None) -> List[str]:
    """Dynamically fetches active chat models available on the provided Groq key."""
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key or key.strip() in ("", "your_groq_api_key_here"):
        return SUPPORTED_GROQ_MODELS

    try:
        from groq import Groq
        client = Groq(api_key=key.strip())
        model_list = client.models.list()
        # Filter out non-chat models (like whisper, prompt-guards)
        chat_models = [
            m.id for m in model_list.data
            if not any(x in m.id for x in ["whisper", "guard", "orpheus"])
        ]
        return chat_models if chat_models else SUPPORTED_GROQ_MODELS
    except Exception:
        return SUPPORTED_GROQ_MODELS


class GroqLLMService:
    """
    Manages communication with Groq LLM API.
    Can be configured via environment variables or explicitly passed credentials.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        mock_mode: bool = False
    ):
        self.api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.mock_mode = mock_mode
        self._client = None

    def _get_client(self):
        """Initializes and caches the Groq client instance."""
        if self.mock_mode:
            return None

        if not self.api_key or self.api_key.strip() in ("", "your_groq_api_key_here"):
            raise LLMServiceError(
                "Missing Groq API Key. Please provide your GROQ_API_KEY in .env "
                "or enter it in the application sidebar."
            )

        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key.strip())
            except ImportError:
                raise LLMServiceError("The 'groq' package is not installed. Run 'pip install groq'.")
            except Exception as exc:
                raise LLMServiceError(f"Failed to initialize Groq client: {exc}")

        return self._client

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> str:
        """
        Sends a chat completion request to the configured Groq model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 recommended for deterministic code/SQL)
            max_tokens: Maximum tokens in completion

        Returns:
            The raw text response from the model.

        Raises:
            LLMServiceError: If the API call fails or authentication is rejected.
        """
        if self.mock_mode:
            # Mock responder for testing
            return "SELECT * FROM data_table LIMIT 10;"

        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            if not response.choices or not response.choices[0].message.content:
                raise LLMServiceError("Groq model returned an empty response.")

            return response.choices[0].message.content.strip()

        except Exception as exc:
            err_msg = str(exc)
            # Differentiate common failure modes
            if "authentication" in err_msg.lower() or "invalid api key" in err_msg.lower() or "401" in err_msg:
                raise LLMServiceError("Authentication Error: The provided Groq API Key is invalid or expired.")
            elif "rate limit" in err_msg.lower() or "429" in err_msg:
                raise LLMServiceError("Rate Limit Exceeded: Groq API request quota reached. Please try again shortly.")
            elif "model_not_found" in err_msg.lower() or "decommissioned" in err_msg.lower() or "404" in err_msg:
                raise LLMServiceError(
                    f"Model Error: The model '{self.model}' is unavailable or decommissioned. "
                    f"Please switch to '{DEFAULT_GROQ_MODEL}' in settings."
                )
            else:
                raise LLMServiceError(f"Groq API Error: {err_msg}")
