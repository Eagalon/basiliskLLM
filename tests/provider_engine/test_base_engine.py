"""Tests for the BaseEngine class."""

from unittest.mock import MagicMock

import pytest

from basilisk.conversation import Conversation
from basilisk.provider_ai_model import ProviderAIModel
from basilisk.provider_engine.base_engine import BaseEngine


def test_base_engine_is_abstract():
	"""Test that BaseEngine cannot be instantiated directly."""
	with pytest.raises(TypeError):
		BaseEngine()


def test_get_user_agent():
	"""Test the get_user_agent method."""

	# Create a concrete subclass for testing
	class ConcreteEngine(BaseEngine):
		@property
		def client(self):
			return MagicMock()

		@property
		def models(self):
			return []

		def prepare_message_request(self, *args, **kwargs):
			pass

		def prepare_message_response(self, *args, **kwargs):
			pass

		def completion(self, *args, **kwargs):
			pass

		def completion_response_with_stream(self, *args, **kwargs):
			pass

		def completion_response_without_stream(self, *args, **kwargs):
			pass

	engine = ConcreteEngine(MagicMock())
	user_agent = engine.get_user_agent()

	assert isinstance(user_agent, str)
	assert "basiliskLLM" in user_agent


def test_get_model():
	"""Test the get_model method."""

	# Create a concrete subclass for testing
	class ConcreteEngine(BaseEngine):
		@property
		def client(self):
			return MagicMock()

		@property
		def models(self):
			return [
				ProviderAIModel(
					id="test-model",
					name="Test Model",
					description="A test model",
				),
				ProviderAIModel(
					id="another-model",
					name="Another Model",
					description="Another test model",
				),
			]

		def prepare_message_request(self, *args, **kwargs):
			pass

		def prepare_message_response(self, *args, **kwargs):
			pass

		def completion(self, *args, **kwargs):
			pass

		def completion_response_with_stream(self, *args, **kwargs):
			pass

		def completion_response_without_stream(self, *args, **kwargs):
			pass

	engine = ConcreteEngine(MagicMock())

	# Test getting a valid model
	result = engine.get_model("test-model")
	assert isinstance(result, ProviderAIModel)
	assert result is not None
	assert result.id == "test-model"
	assert result.name == "Test Model"
	assert result.description == "A test model"
	# Test getting an invalid model
	result = engine.get_model("non-existent-model")
	assert result is None


def test_get_messages(message_block_factory, system_message_factory):
	"""Test the get_messages method."""

	# Create a concrete subclass for testing
	class ConcreteEngine(BaseEngine):
		@property
		def client(self):
			return MagicMock()

		@property
		def models(self):
			return []

		def prepare_message_request(self, message):
			return message.model_dump(mode="json")

		def prepare_message_response(self, response):
			return response.model_dump(mode="json")

		def completion(self, *args, **kwargs):
			pass

		def completion_response_with_stream(self, *args, **kwargs):
			pass

		def completion_response_without_stream(self, *args, **kwargs):
			pass

	engine = ConcreteEngine(MagicMock())

	# Test with a new message block and conversation
	system_message = system_message_factory()
	new_block = message_block_factory()
	conversation = Conversation()
	prev_block = message_block_factory(include_response=True)
	conversation.add_block(prev_block)
	messages = engine.get_messages(new_block, conversation, system_message)
	# Check the results
	assert len(messages) == 4  # system + 3 messages from conversation
	assert messages[0]["role"] == "system"
	assert messages[0]["content"] == system_message.content
	assert messages[-1]["role"] == new_block.request.role
	assert messages[-1]["content"] == new_block.request.content
	# Check the previous block
	assert messages[-2]["role"] == prev_block.response.role
	assert messages[-2]["content"] == prev_block.response.content

	# Test without system message
	messages = engine.get_messages(new_block, conversation, None)
	assert (
		len(messages) == 3
	)  # 3 messages from conversation (no system message)
	assert messages[0]["role"] == prev_block.request.role
	assert messages[0]["content"] == prev_block.request.content
	assert messages[-1]["role"] == new_block.request.role
	assert messages[-1]["content"] == new_block.request.content

	messages = engine.get_messages(new_block, Conversation(), system_message)
	assert len(messages) == 2  # system + new message block
	assert messages[0]["role"] == "system"
	assert messages[1]["role"] == new_block.request.role
	assert messages[1]["content"] == new_block.request.content
