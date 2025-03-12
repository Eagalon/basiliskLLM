"""Tests for the AnthropicEngine class."""

from unittest.mock import ANY, MagicMock, patch

import pytest

from basilisk.config.account_config import Account
from basilisk.provider_capability import ProviderCapability
from basilisk.provider_engine.anthropic_engine import AnthropicEngine


class TestAnthropicEngine:
	"""Tests for the AnthropicEngine class."""

	@pytest.fixture
	def mock_anthropic_client(self):
		"""Create a mock Anthropic client."""
		with patch(
			"basilisk.provider_engine.anthropic_engine.Anthropic"
		) as mock_anthropic:
			mock_client = MagicMock()
			mock_anthropic.return_value = mock_client
			yield mock_client

	@pytest.fixture
	def anthropic_engine(self, mock_anthropic_client):
		"""Create an AnthropicEngine instance for testing."""
		account = Account(
			name="test_account", api_key="test_key", provider_id="anthropic"
		)
		return AnthropicEngine(account=account)

	def test_capabilities(self, anthropic_engine):
		"""Test the capabilities property."""
		capabilities = anthropic_engine.capabilities
		assert ProviderCapability.TEXT in capabilities
		assert ProviderCapability.CITATION in capabilities
		assert ProviderCapability.IMAGE in capabilities
		assert ProviderCapability.DOCUMENT in capabilities

	def test_attachment_formats(self, anthropic_engine):
		"""Test the attachment_formats property."""
		formats = anthropic_engine.supported_attachment_formats
		assert "image/jpeg" in formats
		assert "image/png" in formats
		assert "image/gif" in formats
		assert "image/webp" in formats
		assert "application/pdf" in formats

	def test_client_initialization(self):
		"""Test that the client is initialized correctly."""
		test_account = Account(
			name="test_account", api_key="test_key", provider_id="anthropic"
		)
		with patch(
			"basilisk.provider_engine.anthropic_engine.Anthropic"
		) as mock_anthropic:
			mock_client = MagicMock()
			mock_anthropic.return_value = mock_client

			engine = AnthropicEngine(account=test_account)
			assert engine.client == mock_client
		mock_anthropic.assert_called_once_with(
			api_key=test_account.api_key.get_secret_value()
		)

	def test_models_property(self, anthropic_engine):
		"""Test the models property."""
		models = anthropic_engine.models
		assert len(models) > 0
		for model in models:
			assert model.id.startswith("claude-")

	def test_get_attachment_source_image(
		self, anthropic_engine, image_attachment
	):
		"""Test get_attachment_source with an image attachment."""
		result = anthropic_engine.get_attachment_source(image_attachment)
		assert result == {
			"type": "base64",
			"media_type": "image/png",
			"data": ANY,
		}

	def test_get_attachment_source_file(self, anthropic_engine, attachment):
		"""Test get_attachment_source with a file attachment."""
		result = anthropic_engine.get_attachment_source(attachment)
		assert result == {
			"type": "text",
			"media_type": "text/plain",
			"data": ANY,
		}

	def test_convert_message(
		self, anthropic_engine, user_message_factory, image_attachment
	):
		"""Test convert_message function."""
		# Test basic conversion
		msg = user_message_factory()
		result = anthropic_engine.convert_message(msg)
		assert result["role"] == msg.role
		assert result["content"] == [{"type": "text", "text": msg.content}]

		# Test with attachment
		msg = user_message_factory(attachments=[image_attachment])
		result = anthropic_engine.convert_message(msg)
		assert result["role"] == msg.role
		assert len(result["content"]) == 2
		assert result["content"][0]["type"] == "text"
		assert result["content"][0]["text"] == msg.content
		assert result["content"][1]["type"] == "image"
		assert result["content"][1]["source"]["type"] == "base64"
