"""Unit tests for the Conversation module for the basiliskLLM application."""

import pytest
from pydantic import ValidationError

from basilisk.consts import BSKC_VERSION
from basilisk.conversation import (
	Conversation,
	Message,
	MessageBlock,
	MessageRoleEnum,
)
from basilisk.provider_ai_model import AIModelInfo


class TestAIModelAndMessages:
	"""Tests for AI model and message validation."""

	def test_invalid_ai_model(self):
		"""Test invalid AI model."""
		with pytest.raises(ValidationError) as exc_info:
			AIModelInfo(
				provider_id="invalid_provider", model_id="invalid_model"
			)
			assert exc_info.group_contains(ValueError, "No provider found")

	def test_invalid_msg_role(self):
		"""Test invalid message role."""
		with pytest.raises(ValidationError):
			Message(role="invalid_role", content="test")

	def test_create_message_block(self, ai_model, user_message_factory):
		"""Test creating a message block."""
		user_message = user_message_factory()
		block = MessageBlock(request=user_message, model=ai_model)
		assert block.request.content == user_message.content
		assert block.request.role == MessageRoleEnum.USER
		assert block.response is None
		assert block.model.provider_id == "openai"
		assert block.model.model_id == "test_model"

	def test_create_message_block_with_response(
		self, ai_model, user_message_factory, assistant_message_factory
	):
		"""Test creating a message block with a response."""
		user_message = user_message_factory()
		assistant_message = assistant_message_factory()
		block = MessageBlock(
			request=user_message, response=assistant_message, model=ai_model
		)
		assert block.request.content == user_message.content
		assert block.response.content == assistant_message.content
		assert block.model.provider_id == "openai"
		assert block.model.model_id == "test_model"
		assert block.request.role == MessageRoleEnum.USER
		assert block.response.role == MessageRoleEnum.ASSISTANT


class TestMessageBlockValidation:
	"""Tests for message block validation."""

	def test_invalid_request_role(self, ai_model, assistant_message_factory):
		"""Test invalid request role."""
		assistant_message = assistant_message_factory()
		with pytest.raises(ValidationError) as exc_info:
			MessageBlock(request=assistant_message, model=ai_model)
			assert exc_info.group_contains(
				ValueError, "Request message must be from the user."
			)

	def test_invalid_response_role(self, ai_model, user_message_factory):
		"""Test invalid response role."""
		user_message = user_message_factory()
		with pytest.raises(ValidationError) as exc_info:
			# Using user message as response (invalid)
			MessageBlock(
				request=user_message, response=user_message, model=ai_model
			)
			assert exc_info.group_contains(
				ValueError, "Response message must be from the assistant."
			)

	def test_message_block_no_request(
		self, ai_model, assistant_message_factory
	):
		"""Test message block with no request."""
		assistant_message = assistant_message_factory()
		with pytest.raises(ValidationError):
			MessageBlock(response=assistant_message, model=ai_model)

	def test_message_block_no_attachments_in_response(
		self, ai_model, attachment, faker
	):
		"""Test message block with no attachments in response."""
		msg_content = faker.paragraph()
		req_msg = Message(role=MessageRoleEnum.USER, content=msg_content)
		res_msg = Message(
			role=MessageRoleEnum.ASSISTANT,
			content=msg_content,
			attachments=[attachment],
		)

		with pytest.raises(ValidationError) as exc_info:
			MessageBlock(request=req_msg, response=res_msg, model=ai_model)
			assert exc_info.group_contains(
				ValueError, "Response messages cannot have attachments."
			)


class TestConversationBasics:
	"""Tests for basic conversation functionality."""

	def test_create_empty_conversation(self, empty_conversation):
		"""Test creating an empty conversation."""
		assert empty_conversation.messages == []
		assert empty_conversation.systems == {}
		assert empty_conversation.title is None
		assert empty_conversation.version == BSKC_VERSION

	def test_invalid_min_conversation_version(self, empty_conversation):
		"""Test invalid minimum conversation version."""
		empty_conversation.version = -1
		json = empty_conversation.model_dump_json()
		with pytest.raises(ValidationError):
			Conversation.model_validate_json(json)

	def test_invalid_max_conversation_version(self, empty_conversation):
		"""Test invalid maximum conversation version."""
		empty_conversation.version = BSKC_VERSION + 1
		json = empty_conversation.model_dump_json()
		with pytest.raises(ValidationError):
			Conversation.model_validate_json(json)

	def test_add_block_without_system(
		self, empty_conversation, message_block_factory
	):
		"""Test adding a message block to a conversation without a system message."""
		message_block = message_block_factory()
		empty_conversation.add_block(message_block)

		assert len(empty_conversation.messages) == 1
		assert empty_conversation.messages[0] == message_block
		assert empty_conversation.messages[0].system_index is None
		assert len(empty_conversation.systems) == 0

	def test_add_block_with_system(
		self, empty_conversation, message_block_factory, system_message_factory
	):
		"""Test adding a message block to a conversation with a system message."""
		message_block = message_block_factory()
		system_message = system_message_factory()
		empty_conversation.add_block(message_block, system_message)

		assert len(empty_conversation.messages) == 1
		assert empty_conversation.messages[0] == message_block
		assert empty_conversation.messages[0].system_index == 0
		assert len(empty_conversation.systems) == 1
		assert system_message in empty_conversation.systems


class TestConversationWithMultipleBlocks:
	"""Tests for conversations with multiple message blocks."""

	def test_add_block_with_duplicate_system(
		self, empty_conversation, message_block_factory, system_message_factory
	):
		"""Test adding blocks with duplicate system messages."""
		first_block = message_block_factory()
		second_block = message_block_factory()
		system_message = system_message_factory()
		empty_conversation.add_block(first_block, system_message)
		empty_conversation.add_block(second_block, system_message)
		assert len(empty_conversation.messages) == 2
		assert empty_conversation.messages[0].system_index == 0
		assert empty_conversation.messages[1].system_index == 0
		assert (
			len(empty_conversation.systems) == 1
		)  # Only one unique system message

	def test_add_block_with_multiple_systems(
		self, empty_conversation, message_block_factory, system_message_factory
	):
		"""Test adding blocks with multiple different system messages."""
		first_block = message_block_factory()
		second_block = message_block_factory()
		first_system = system_message_factory()
		second_system = system_message_factory()
		empty_conversation.add_block(first_block, first_system)
		empty_conversation.add_block(second_block, second_system)
		assert len(empty_conversation.messages) == 2
		assert empty_conversation.messages[0].system_index == 0
		assert empty_conversation.messages[1].system_index == 1
		assert len(empty_conversation.systems) == 2
		assert first_system in empty_conversation.systems
		assert second_system in empty_conversation.systems
		assert empty_conversation.messages[0] == first_block
		assert empty_conversation.messages[1] == second_block

	def test_remove_block_without_system(
		self, empty_conversation, message_block_factory
	):
		"""Test removing a message block without a system message."""
		message_block = message_block_factory()
		empty_conversation.add_block(message_block)
		assert len(empty_conversation.messages) == 1
		assert empty_conversation.messages[0] == message_block
		assert empty_conversation.messages[0].system_index is None
		assert len(empty_conversation.systems) == 0
		empty_conversation.remove_block(message_block)
		assert len(empty_conversation.messages) == 0
		assert len(empty_conversation.systems) == 0
		assert message_block not in empty_conversation.messages

	def test_remove_block_with_system(
		self, empty_conversation, message_block_factory, system_message_factory
	):
		"""Test removing a message block with a system message."""
		message_block = message_block_factory()
		system_message = system_message_factory()
		empty_conversation.add_block(message_block, system_message)

		assert len(empty_conversation.messages) == 1
		assert empty_conversation.messages[0].system_index == 0
		assert empty_conversation.messages[0] == message_block
		assert system_message in empty_conversation.systems
		assert len(empty_conversation.systems) == 1
		empty_conversation.remove_block(message_block)
		assert len(empty_conversation.messages) == 0
		assert len(empty_conversation.systems) == 0

	def test_remove_block_with_shared_system(
		self, empty_conversation, message_block_factory, system_message_factory
	):
		"""Test removing a block that shares a system message with another block."""
		first_block = message_block_factory()
		second_block = message_block_factory()
		system_message = system_message_factory()
		empty_conversation.add_block(first_block, system_message)
		empty_conversation.add_block(second_block, system_message)

		assert len(empty_conversation.messages) == 2
		assert empty_conversation.messages[0].system_index == 0
		assert empty_conversation.messages[1].system_index == 0
		assert len(empty_conversation.systems) == 1

		empty_conversation.remove_block(first_block)
		assert len(empty_conversation.messages) == 1
		assert (
			empty_conversation.messages[0].system_index == 0
		)  # Index unchanged
		assert empty_conversation.messages[0] == second_block
		assert (
			len(empty_conversation.systems) == 1
		)  # System still used by remaining block

	def test_remove_block_with_multiple_systems(
		self, empty_conversation, message_block_factory, system_message_factory
	):
		"""Test removing blocks with multiple system messages."""
		first_block = message_block_factory()
		second_block = message_block_factory()
		first_system = system_message_factory()
		second_system = system_message_factory()
		third_block = message_block_factory()
		empty_conversation.add_block(first_block, first_system)
		empty_conversation.add_block(second_block, second_system)
		empty_conversation.add_block(third_block, first_system)

		assert len(empty_conversation.messages) == 3
		assert empty_conversation.messages[0].system_index == 0
		assert empty_conversation.messages[1].system_index == 1
		assert empty_conversation.messages[2].system_index == 0
		assert len(empty_conversation.systems) == 2
		empty_conversation.remove_block(second_block)
		assert len(empty_conversation.messages) == 2
		assert empty_conversation.messages[0].system_index == 0
		assert empty_conversation.messages[1].system_index == 0
		assert len(empty_conversation.systems) == 1
		assert first_system in empty_conversation.systems
		assert second_system not in empty_conversation.systems

	def test_remove_block_with_index_adjustment(
		self, empty_conversation, message_block_factory, system_message_factory
	):
		"""Test system index adjustment when removing a system."""
		first_block = message_block_factory()
		second_block = message_block_factory()
		third_block = message_block_factory()
		first_system = system_message_factory()
		second_system = system_message_factory()
		third_system = system_message_factory()
		empty_conversation.add_block(first_block, first_system)
		empty_conversation.add_block(second_block, second_system)
		empty_conversation.add_block(third_block, third_system)
		assert len(empty_conversation.messages) == 3
		assert empty_conversation.messages[0].system_index == 0
		assert empty_conversation.messages[1].system_index == 1
		assert empty_conversation.messages[2].system_index == 2
		assert len(empty_conversation.systems) == 3

		empty_conversation.remove_block(first_block)
		assert len(empty_conversation.messages) == 2
		assert empty_conversation.messages[0].system_index == 0  # Was 1, now 0
		assert empty_conversation.messages[1].system_index == 1  # Was 2, now 1
		assert len(empty_conversation.systems) == 2
		assert first_system not in empty_conversation.systems
		assert second_system in empty_conversation.systems
		assert third_system in empty_conversation.systems
