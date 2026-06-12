"""
60db (60db.ai) streaming text-to-speech over WebSocket, behind the TTSProvider interface.

Protocol (see https://docs.60db.ai/websocket-api/tts):
  connect       -> server: {connecting:true} then {connection_established:{...}}
  create_context-> server: {context_created:{context_id}}
  send_text     -> buffers text
  flush_context -> server: {audio_chunk:{audioContent: <base64 pcm>}} ... {flush_completed}
  close_context -> server: {context_closed} then socket closes

We use LINEAR16 PCM so chunks concatenate directly and can be written straight to a
16-bit PyAudio output stream. (OGG_OPUS would need per-chunk decoding - avoid it here.)
A single context is reused across many speak() calls, which the docs encourage and which
keeps cost down ($0.01 minimum per context).
"""

import asyncio
import base64
import json
import urllib.request
from typing import Optional

import websockets

from .base import AudioHandler, TTSProvider

# The docs' documented default voice. Override via voice_id / SIXTYDB_VOICE_ID env.
DEFAULT_VOICE_ID = "fbb75ed2-975a-40c7-9e06-38e30524a9a1"


class SixtyDBTTS(TTSProvider):
    def __init__(self, api_key: str,
                 voice_id: str = DEFAULT_VOICE_ID,
                 sample_rate: int = 24000,
                 encoding: str = "LINEAR16",
                 speed: float = 1,
                 stability: int = 50,
                 similarity: int = 75,
                 context_id: str = "hoodie",
                 on_audio: Optional[AudioHandler] = None):
        super().__init__(on_audio)
        self._api_key = api_key
        self._voice_id = voice_id
        self._sample_rate = sample_rate
        self._encoding = encoding
        self._speed = speed
        self._stability = stability
        self._similarity = similarity
        self._context_id = context_id

        self._ws = None
        self._recv_task: Optional[asyncio.Task] = None
        self._authenticated = asyncio.Event()  # set on connection_established
        self._context_ready = asyncio.Event()  # set on context_created

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def _url(self) -> str:
        # NOTE: the docs use ws:// (cleartext). Swap to wss:// if/when 60db offers TLS.
        return f"ws://api.60db.ai/ws/tts?apiKey={self._api_key}"

    async def connect(self) -> None:
        self._authenticated.clear()
        self._context_ready.clear()

        # max_size=None: audio_chunk messages can be large; don't let the client cap them.
        self._ws = await websockets.connect(self._url, max_size=None)
        self._recv_task = asyncio.create_task(self._receive_loop())

        # Wait for the server to authenticate us, then open a synthesis context.
        await self._authenticated.wait()
        await self._send({
            "create_context": {
                "context_id": self._context_id,
                "voice_id": self._voice_id,
                "audio_config": {
                    "audio_encoding": self._encoding,
                    "sample_rate_hertz": self._sample_rate,
                },
                "speed": self._speed,
                "stability": self._stability,
                "similarity": self._similarity,
            }
        })
        await self._context_ready.wait()
        print(f"60db TTS context '{self._context_id}' ready (voice {self._voice_id})")

    async def _send(self, message: dict) -> None:
        await self._ws.send(json.dumps(message))

    async def _receive_loop(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)

                if "connection_established" in msg:
                    info = msg["connection_established"]
                    print(f"60db connected (credit balance: {info.get('credit_balance')})")
                    self._authenticated.set()
                elif "context_created" in msg:
                    self._context_ready.set()
                elif "audio_chunk" in msg:
                    pcm = base64.b64decode(msg["audio_chunk"]["audioContent"])
                    self._emit_audio(pcm)
                elif "flush_completed" in msg:
                    pass  # all audio for the last flush has been delivered
                elif "context_closed" in msg:
                    break
                elif "error" in msg:
                    print(f"60db TTS error: {msg['error'].get('message')}")
                # ignore {connecting: true} and anything else
        except websockets.ConnectionClosed:
            pass

    # speak() auto-(re)connects so a dropped socket or an idle timeout never wedges the
    # hoodie - it just dials again and re-opens the context before synthesizing.
    async def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        for attempt in (1, 2):
            try:
                if self._ws is None or self._ws.closed or not self._context_ready.is_set():
                    await self.connect()
                await self._send({"send_text": {"context_id": self._context_id, "text": text}})
                await self._send({"flush_context": {"context_id": self._context_id}})
                return
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                print(f"60db speak attempt {attempt} failed ({e}); reconnecting")
                self._context_ready.clear()
                self._ws = None
        print("60db TTS: giving up on this line")

    async def close(self) -> None:
        if self._ws is not None and not self._ws.closed:
            try:
                await self._send({"close_context": {"context_id": self._context_id}})
                await self._ws.close()
            except (websockets.ConnectionClosed, OSError):
                pass
        if self._recv_task is not None:
            self._recv_task.cancel()


def fetch_my_voices(api_key: str) -> list:
    """
    Convenience: list the cloned/professional voices on your 60db account so you can
    grab a voice_id. Uses the REST endpoint (GET /myvoices) - no SDK needed.
    """
    req = urllib.request.Request(
        "https://api.60db.ai/myvoices",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req) as resp:
        payload = json.loads(resp.read().decode())
    return payload.get("data", [])


if __name__ == "__main__":
    # Run `python -m providers.sixtydb_tts` (with SIXTYDB_API_KEY set) to list your voices.
    import os

    key = os.environ.get("SIXTYDB_API_KEY")
    if not key:
        raise SystemExit("Set SIXTYDB_API_KEY in your environment first.")
    for v in fetch_my_voices(key):
        labels = v.get("labels", {})
        print(f"{v['voice_id']}  {v['name']:<24} "
              f"{labels.get('language_name', '?')}/{labels.get('gender', '?')}  "
              f"[{v.get('model')}]")
