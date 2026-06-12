"""
Deepgram streaming STT, hidden behind the STTProvider interface.

All the Deepgram-SDK-specific code lives HERE and nowhere else - the original
author's wish ("consolidate all the Deepgram SDK funsies in case the spec changes").
The rest of the app only ever sees a normalized TranscriptResult.
"""

from typing import Optional

from deepgram import Deepgram
from deepgram.transcription import LiveTranscription

from .base import STTProvider, TranscriptHandler, TranscriptResult


class DeepgramSTT(STTProvider):
    def __init__(self, api_key: str, sample_rate: int,
                 language: str = "en-US",
                 on_transcript: Optional[TranscriptHandler] = None):
        super().__init__(on_transcript)
        self._deepgram = Deepgram(api_key)
        self._sample_rate = sample_rate
        self._language = language
        self._live: Optional[LiveTranscription] = None

    # We can't reuse a Deepgram live socket once it's been finished - reopening
    # just builds a fresh one. Has nothing to do with request IDs (those split on
    # pauses between statements).
    async def connect(self) -> None:
        try:
            self._live = await self._deepgram.transcription.live({
                "language": self._language,
                "encoding": "linear16",
                "sample_rate": self._sample_rate,   # must match the PyAudio capture config
                "punctuate": True,
                "interim_results": False,
                "diarize": True,                     # TODO: distinguish wearer from interlocutor
            })
        except Exception as e:
            print(f"Could not open Deepgram socket: {e}")
            raise

        self._live.registerHandler(
            self._live.event.CLOSE,
            lambda c: print(f"Deepgram connection closed with code {c}."))
        self._live.registerHandler(
            self._live.event.TRANSCRIPT_RECEIVED, self._on_raw_response)

    # The SDK fires this for every message. We parse out the bits we care about
    # and normalize them; status/metadata messages lack these keys and are dropped.
    def _on_raw_response(self, response) -> None:
        try:
            alternative = response["channel"]["alternatives"][0]
            result = TranscriptResult(
                transcript=alternative["transcript"],
                words=alternative["words"],
                request_id=response["metadata"]["request_id"],
                is_final=response["is_final"],
            )
        except (KeyError, IndexError, TypeError):
            # Status messages, metadata, confirmation, and who knows what else.
            return
        self._emit(result)

    async def send_audio(self, pcm: bytes) -> None:
        if self._live is not None:
            self._live.send(pcm)

    async def finish(self) -> None:
        if self._live is not None:
            await self._live.finish()

    @property
    def done(self) -> bool:
        return self._live is None or self._live.done
