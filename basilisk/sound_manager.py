"""Sound manager for playing sound effects in the application.

This module provides a centralized sound management system that handles:
- Loading and caching of sound files
- Asynchronous playback of sound effects
- Looped playback functionality
- Global sound management through singleton pattern

Uses sounddevice library for audio playback and standard wave module to read files.
"""

import logging
import threading
import time
import wave
from functools import cache
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import sounddevice as sd

from .global_vars import resource_path

log = logging.getLogger(__name__)

ALIASES = {
	"chat_request_sent": resource_path
	/ Path("sounds", "chat_request_sent.wav"),
	"chat_response_pending": resource_path
	/ Path("sounds", "chat_response_pending.wav"),
	"chat_response_received": resource_path
	/ Path("sounds", "chat_response_received.wav"),
	"progress": resource_path / Path("sounds", "progress.wav"),
	"recording_started": resource_path
	/ Path("sounds", "recording_started.wav"),
	"recording_stopped": resource_path
	/ Path("sounds", "recording_stopped.wav"),
}


class SoundManager:
	"""Manager class for playing sound effects.

	This class implements a thread-safe sound manager with caching capabilities.
	It supports both one-shot and looped playback of sound effects.
	"""

	def __init__(self):
		"""Initialize the sound manager.

		Sets up:
		- Sound cache for efficient playback
		- Threading support for looped playback
		- Sound playback control
		"""
		self.current_sound = None
		self.loop = False
		self.loop_thread = None
		self.thread_lock = threading.Lock()
		self.sound_cache: Dict[Path, Tuple[np.ndarray, int]] = {}

	def _ensure_sound_loaded(self, file_path: Path) -> Tuple[np.ndarray, int]:
		"""Ensure that the sound file is loaded and cached.

		Args:
			file_path: Path to the sound file

		Returns:
			Tuple of (audio_data, sample_rate)

		Raises:
			IOError: If the sound file could not be loaded
		"""
		if file_path in self.sound_cache:
			return self.sound_cache[file_path]

		try:
			with wave.open(str(file_path), "rb") as wav_file:
				# Get basic info from wav file
				sample_rate = wav_file.getframerate()
				n_frames = wav_file.getnframes()
				n_channels = wav_file.getnchannels()
				sample_width = wav_file.getsampwidth()

				# Read all frames
				frames = wav_file.readframes(n_frames)

				# Convert to numpy array based on sample width
				if sample_width == 2:  # 16-bit
					dtype = np.int16
				elif sample_width == 4:  # 32-bit
					dtype = np.int32
				else:  # Default to 16-bit
					dtype = np.int16

				# Reshape according to number of channels
				audio_data = np.frombuffer(frames, dtype=dtype)
				if n_channels > 1:
					audio_data = audio_data.reshape(-1, n_channels)

				# Normalize data to float for sounddevice
				audio_data = audio_data.astype(np.float32) / np.iinfo(dtype).max

			self.sound_cache[file_path] = (audio_data, sample_rate)
			return audio_data, sample_rate
		except Exception as e:
			raise IOError(f"Failed to load sound: {file_path}, error: {str(e)}")

	def _play_sound_loop(self, data: np.ndarray, sample_rate: int):
		"""Play a sound in a loop until the loop flag is set to False.

		Args:
			data: Audio data as numpy array
			sample_rate: Sample rate of the audio data
		"""
		while self.loop:
			sd.play(data, samplerate=sample_rate, blocking=False)

			# Wait until playback is finished or loop is stopped
			try:
				stream = sd.get_stream()
				while stream and stream.active and self.loop:
					time.sleep(0.1)
			except sd.PortAudioError as e:
				log.error(f"PortAudio error: {e}")
				# Handle case where get_stream might fail
				time.sleep(len(data) / sample_rate)

			if not self.loop:
				sd.stop()

	def play_sound(self, file_path: Union[str, Path], loop: bool = False):
		"""Play a sound effect. If loop is True, the sound will be played in a loop.

		Args:
			file_path: Path to the sound file or a predefined alias from aliases mapping
			loop: Whether to play the sound in a loop
		"""
		with self.thread_lock:
			if isinstance(file_path, str) and file_path in ALIASES:
				file_path = ALIASES[file_path]

			self.stop_sound()

			data, sample_rate = self._ensure_sound_loaded(file_path)

			self.loop = loop

			if loop:
				self.loop_thread = threading.Thread(
					target=self._play_sound_loop,
					args=(data, sample_rate),
					daemon=True,
				)
				self.loop_thread.start()
			else:
				sd.play(data, samplerate=sample_rate)

	def stop_sound(self):
		"""Stop the currently playing sound effect."""
		self.loop = False
		if self.loop_thread is not None:
			self.loop_thread.join(timeout=1)
			self.loop_thread = None
		sd.stop()


@cache
def get_sound_manager() -> SoundManager:
	"""Initialize the global sound manager."""
	log.debug("Initializing sound manager")
	return SoundManager()


def play_sound(file_path: Union[str, Path], loop: bool = False):
	"""Play a sound using the global sound manager. If loop is True, the sound will be played in a loop.

	Args:
		file_path: Path to the sound file or a predefined alias
		loop: Whether to play the sound in a loop
	"""
	log.debug(f"Playing sound: {file_path}, loop: {loop}")
	get_sound_manager().play_sound(file_path, loop)


def stop_sound():
	"""Stop the currently playing sound effect using the global sound manager."""
	get_sound_manager().stop_sound()
