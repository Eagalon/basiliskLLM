"""Module for managing conversation between users and the bot."""

from .attached_file import (
	URL_PATTERN,
	AttachmentFile,
	AttachmentFileTypes,
	ImageFile,
	NotImageError,
	build_from_url,
	get_mime_type,
	parse_supported_attachment_formats,
)
from .conversation_helper import PROMPT_TITLE
from .conversation_model import (
	Conversation,
	Message,
	MessageBlock,
	MessageRoleEnum,
	SystemMessage,
)

__all__ = [
	"AttachmentFile",
	"AttachmentFileTypes",
	"build_from_url",
	"Conversation",
	"get_mime_type",
	"ImageFile",
	"ImageFileTypes",
	"Message",
	"MessageBlock",
	"MessageRoleEnum",
	"NotImageError",
	"parse_supported_attachment_formats",
	"PROMPT_TITLE",
	"SystemMessage",
	"URL_PATTERN",
]
