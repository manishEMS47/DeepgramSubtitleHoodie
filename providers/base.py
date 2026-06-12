"""
Provider abstraction for the Subtitle Hoodie.

The hoodie has two independent jobs:
  - STT (speech-to-text): turn the wearer's voice into subtitles on the chest display.
  - TTS (text-to-speech): speak finalized lines back out loud in a chosen voice.

By hiding each vendor behind a tiny interface, the rest of the app never touches a
vendor-specific response shape. Swapping Deepgram for 60db (or vice versa) is a
one-line change in main.py, and the display/audio code stays identical.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class TranscriptResult:
    """
    The normalized output of ANY STT provider.

    `words` is a list of plain dicts, each with at least 'word', 'start', 'end'
    (seconds since the start of this request). This matches what SubtitleDisplay
    already expects, so the display logic doesn't care which vendor produced it.
    """
    transcript: str
    words: list = field(default_factory=list)
    request_id: str = "0"
    is_final: bool = False


# Type alias for the callback an STT provider fires for every transcript it receives.
TranscriptHandler = Callable[[TranscriptResult], None]
# Type alias for the callback a TTS provider fires for every raw PCM audio chunk it gets back.
AudioHandler = Callable[[bytes], None]


class STTProvider(ABC):
    """
    A streaming speech-to-text provider.

    Lifecycle: connect() -> send_audio() repeatedly -> finish().
    A provider may be reconnected (connect() again) after finish() - the hoodie
    tears the socket down during silence to save credit and reopens it on speech.
    """

    def __init__(self, on_transcript: Optional[TranscriptHandler] = None):
        self._on_transcript = on_transcript

    def set_transcript_handler(self, handler: TranscriptHandler) -> None:
        self._on_transcript = handler

    def _emit(self, result: TranscriptResult) -> None:
        if self._on_transcript is not None:
            self._on_transcript(result)

    @abstractmethod
    async def connect(self) -> None:
        """Open (or reopen) the streaming connection."""

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None:
        """Stream a chunk of raw PCM audio to the provider."""

    @abstractmethod
    async def finish(self) -> None:
        """Signal end-of-stream and let the provider flush its final results."""

    @property
    @abstractmethod
    def done(self) -> bool:
        """True if there is no live connection to send audio on."""


class TTSProvider(ABC):
    """
    A streaming text-to-speech provider.

    Lifecycle: connect() -> speak() repeatedly -> close().
    Every chunk of synthesized audio is handed back through the `on_audio`
    callback as raw little-endian 16-bit PCM, ready to write to an output device.
    """

    def __init__(self, on_audio: Optional[AudioHandler] = None):
        self._on_audio = on_audio

    def set_audio_handler(self, handler: AudioHandler) -> None:
        self._on_audio = handler

    def _emit_audio(self, pcm: bytes) -> None:
        if self._on_audio is not None:
            self._on_audio(pcm)

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate (Hz) of the PCM handed to the audio callback - drives the output device."""

    @abstractmethod
    async def connect(self) -> None:
        """Open the connection and prepare a synthesis context."""

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Synthesize `text`; audio arrives asynchronously via the on_audio callback."""

    @abstractmethod
    async def close(self) -> None:
        """Flush remaining audio and close the connection."""
