"""
Tests for LLM Service Module
Covers Scenarios:
15. Groq API errors
16. Groq model unavailability
20. Missing API configuration
"""

from unittest.mock import MagicMock, patch
import pytest
from modules.llm_service import GroqLLMService, LLMServiceError


def test_scenario_20_missing_api_configuration():
    """Scenario 20: Missing API configuration raises LLMServiceError with clear instructions."""
    service = GroqLLMService(api_key="", mock_mode=False)
    with pytest.raises(LLMServiceError, match="Missing Groq API Key"):
        service.generate_chat_completion([{"role": "user", "content": "Hello"}])


def test_mock_mode_generates_safe_default():
    """Verifies mock mode provides valid SQL for testing without live API keys."""
    service = GroqLLMService(mock_mode=True)
    res = service.generate_chat_completion([{"role": "user", "content": "What are sales?"}])
    assert "SELECT" in res


def test_scenario_15_groq_api_errors():
    """Scenario 15: Groq API errors (such as 429 Rate Limit or 500) are caught and formatted."""
    service = GroqLLMService(api_key="gsk_test_key_123", mock_mode=False)

    # Mock the client's completions.create to simulate rate limiting
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Rate limit reached: 429 Too Many Requests")
    service._client = mock_client

    with pytest.raises(LLMServiceError, match="Rate Limit Exceeded"):
        service.generate_chat_completion([{"role": "user", "content": "Hello"}])


def test_scenario_16_groq_model_unavailability():
    """Scenario 16: Decommissioned or unavailable Groq models trigger a clear recommendation."""
    service = GroqLLMService(api_key="gsk_test_key_123", model="mixtral-8x7b-32768", mock_mode=False)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Error 404: model_not_found or decommissioned")
    service._client = mock_client

    with pytest.raises(LLMServiceError, match="Model Error: The model 'mixtral-8x7b-32768' is unavailable"):
        service.generate_chat_completion([{"role": "user", "content": "Hello"}])
