"""Pluggable STT/TTS providers for the Subtitle Hoodie."""

from .base import STTProvider, TTSProvider, TranscriptResult
from .deepgram_stt import DeepgramSTT
from .sixtydb_tts import SixtyDBTTS, fetch_my_voices

__all__ = [
    "STTProvider",
    "TTSProvider",
    "TranscriptResult",
    "DeepgramSTT",
    "SixtyDBTTS",
    "fetch_my_voices",
]
