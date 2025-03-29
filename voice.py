from __future__ import annotations

import asyncio
import base64
import threading
from typing import Any, cast

import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime.session import Session

CHUNK_LENGTH_S = 0.05
SAMPLE_RATE = 24000
CHANNELS = 1


class AudioPlayerAsync:
	def __init__(self):
		self.queue = []
		self.lock = threading.Lock()
		self.stream = sd.OutputStream(
			callback=self.callback,
			samplerate=SAMPLE_RATE,
			channels=CHANNELS,
			dtype=np.int16,
			blocksize=int(CHUNK_LENGTH_S * SAMPLE_RATE),
		)
		self.playing = False
		self._frame_count = 0

	def callback(self, outdata, frames, time, status):  # noqa
		with self.lock:
			data = np.empty(0, dtype=np.int16)

			# get next item from queue if there is still space in the buffer
			while len(data) < frames and len(self.queue) > 0:
				item = self.queue.pop(0)
				frames_needed = frames - len(data)
				data = np.concatenate((data, item[:frames_needed]))
				if len(item) > frames_needed:
					self.queue.insert(0, item[frames_needed:])

			self._frame_count += len(data)

			# fill the rest of the frames with zeros if there is no more data
			if len(data) < frames:
				data = np.concatenate(
					(data, np.zeros(frames - len(data), dtype=np.int16))
				)

		outdata[:] = data.reshape(-1, 1)

	def reset_frame_count(self):
		self._frame_count = 0

	def get_frame_count(self):
		return self._frame_count

	def add_data(self, data: bytes):
		with self.lock:
			# bytes is pcm16 single channel audio data, convert to numpy array
			np_data = np.frombuffer(data, dtype=np.int16)
			self.queue.append(np_data)
			if not self.playing:
				self.start()

	def start(self):
		self.playing = True
		self.stream.start()

	def stop(self):
		self.playing = False
		self.stream.stop()
		with self.lock:
			self.queue = []

	def terminate(self):
		self.stream.close()


class TestApp:
	client: AsyncOpenAI
	should_send_audio: asyncio.Event
	audio_player: AudioPlayerAsync
	last_audio_item_id: str | None
	connection: AsyncRealtimeConnection | None
	session: Session | None
	connected: asyncio.Event

	def __init__(self) -> None:
		super().__init__()
		self.connection = None
		self.session = None
		self.client = AsyncOpenAI()
		self.audio_player = AudioPlayerAsync()
		self.last_audio_item_id = None
		self.should_send_audio = asyncio.Event()
		self.connected = asyncio.Event()

	async def handle_realtime_connection(self) -> None:
		async with self.client.beta.realtime.connect(
			model="gpt-4o-realtime-preview"
		) as conn:
			self.connection = conn
			self.connected.set()

			# note: this is the default and can be omitted
			# if you want to manually handle VAD yourself, then set `'turn_detection': None`
			await conn.session.update(
				session={"turn_detection": {"type": "server_vad"}}
			)

			acc_items: dict[str, Any] = {}

			async for event in conn:
				if event.type == "session.created":
					self.session = event.session
					assert event.session.id is not None
					print(f"session: {event.session.id}")
					continue

				if event.type == "session.updated":
					self.session = event.session
					continue

				if event.type == "response.audio.delta":
					if event.item_id != self.last_audio_item_id:
						self.audio_player.reset_frame_count()
						self.last_audio_item_id = event.item_id

					bytes_data = base64.b64decode(event.delta)
					self.audio_player.add_data(bytes_data)
					continue

				if event.type == "response.audio_transcript.delta":
					try:
						text = acc_items[event.item_id]
					except KeyError:
						acc_items[event.item_id] = event.delta
					else:
						acc_items[event.item_id] = text + event.delta

					print(f"item event: {acc_items[event.item_id]}")
					continue

	async def _get_connection(self) -> AsyncRealtimeConnection:
		await self.connected.wait()
		assert self.connection is not None
		return self.connection

	async def send_mic_audio(self) -> None:
		sent_audio = False
		read_size = int(SAMPLE_RATE * 0.02)
		stream = sd.InputStream(
			channels=CHANNELS, samplerate=SAMPLE_RATE, dtype="int16"
		)
		stream.start()

		try:
			while True:
				if stream.read_available < read_size:
					await asyncio.sleep(0)
					continue

				await self.should_send_audio.wait()
				print("recording...")

				data, _ = stream.read(read_size)

				connection = await self._get_connection()
				if not sent_audio:
					asyncio.create_task(
						connection.send({"type": "response.cancel"})
					)
					sent_audio = True

				await connection.input_audio_buffer.append(
					audio=base64.b64encode(cast(Any, data)).decode("utf-8")
				)

				await asyncio.sleep(0)
		except KeyboardInterrupt:
			pass
		finally:
			stream.stop()
			stream.close()


async def main():
	app = TestApp()
	app.should_send_audio.set()
	await asyncio.gather(app.handle_realtime_connection(), app.send_mic_audio())


if __name__ == "__main__":
	asyncio.run(main())
