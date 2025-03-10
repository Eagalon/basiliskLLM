from unittest.mock import patch
from uuid import uuid4

import pytest
from pydantic_settings import InitSettingsSource

from basilisk.config import Account
from basilisk.config.conversation_profile import (
	ConversationProfile,
	ConversationProfileManager,
)


def test_conversation_profile_initialization():
	profile = ConversationProfile(name="test_profile")
	assert profile.name == "test_profile"
	assert profile.system_prompt == ""
	assert profile.account_info is None
	assert profile.ai_model_info is None
	assert profile.max_tokens is None
	assert profile.temperature is None
	assert profile.top_p is None
	assert profile.stream_mode is True


def test_conversation_profile_set_account():
	account_id = uuid4()
	account = Account(name="test_account", id=account_id, provider_id="ollama")
	profile = ConversationProfile(name="test_profile")
	profile.set_account(account)
	assert profile.account_info == account_id
	assert profile.account == account
	profile.set_account(None)
	assert profile.account_info is None
	assert profile.account is None


def test_conversation_profile_set_model_info():
	profile = ConversationProfile(name="test_profile")
	profile.set_model_info("openai", "gpt-4")
	assert profile.ai_model_info.provider_id == "openai"
	assert profile.ai_model_info.model_id == "gpt-4"


def test_conversation_profile_equality():
	profile1 = ConversationProfile(name="profile1")
	profile2 = ConversationProfile(name="profile2")
	assert profile1 != profile2

	profile2.id = profile1.id
	assert profile1 == profile2


def test_conversation_profile_check_same_provider():
	account = Account(
		name="test_account",
		id=uuid4(),
		provider_id="openai",
		api_key="test_key",
	)
	profile = ConversationProfile(name="test_profile")
	profile.set_model_info("openai", "gpt-4")
	profile.set_account(account)
	profile.check_same_provider()
	with pytest.raises(ValueError):
		profile.set_model_info("anthropic", "claude")
		profile.check_same_provider()


def test_conversation_profile_check_model_params():
	profile = ConversationProfile(name="test_profile")
	profile.check_model_params()
	profile.set_model_info("openai", "gpt-4")
	profile.max_tokens = 100
	profile.temperature = 0.7
	profile.top_p = 0.9
	profile.check_model_params()

	profile.ai_model_info = None
	with pytest.raises(ValueError):
		profile.check_model_params()


def test_conversation_profile_manager_add_remove():
	init_settings = InitSettingsSource(ConversationProfileManager, {})
	with patch(
		"basilisk.config.conversation_profile.ConversationProfileManager.settings_customise_sources",
		return_value=(init_settings,),
	):
		manager = ConversationProfileManager()
		profile = ConversationProfile(name="test_profile")
		manager.add(profile)
		assert len(manager) == 1
		assert manager[0] == profile
		manager.remove(profile)
		assert len(manager) == 0


def test_conversation_profile_manager_default_profile():
	init_settings = InitSettingsSource(ConversationProfileManager, {})
	with patch(
		"basilisk.config.conversation_profile.ConversationProfileManager.settings_customise_sources",
		return_value=(init_settings,),
	):
		manager = ConversationProfileManager()
		assert manager.default_profile is None
		profile = ConversationProfile(name="test_profile")
		manager.add(profile)
		manager.set_default_profile(profile)
		assert manager.default_profile == profile
		manager.set_default_profile(None)
		assert manager.default_profile is None


def test_conversation_profile_manager_get_profile():
	init_settings = InitSettingsSource(ConversationProfileManager, {})
	with patch(
		"basilisk.config.conversation_profile.ConversationProfileManager.settings_customise_sources",
		return_value=(init_settings,),
	):
		manager = ConversationProfileManager()
		profile = ConversationProfile(name="test_profile")
		manager.add(profile)
		retrieved_profile = manager.get_profile(name="test_profile")
		assert retrieved_profile == profile


def test_conversation_profile_manager_save():
	init_settings = InitSettingsSource(ConversationProfileManager, {})
	with patch(
		"basilisk.config.conversation_profile.ConversationProfileManager.settings_customise_sources",
		return_value=(init_settings,),
	):
		manager = ConversationProfileManager()
		profile = ConversationProfile(name="test_profile")
		manager.add(profile)
		with patch(
			"basilisk.config.conversation_profile.save_config_file"
		) as mock_save:
			manager.save()
			mock_save.assert_called_once()
