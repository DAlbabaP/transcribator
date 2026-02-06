"""
WhisperX pipeline wrapper for transcription, alignment, and diarization.
"""

from typing import Dict, List, Optional, Tuple


from dataclasses import dataclass


@dataclass
class MergedSegment:
    """Объединенный сегмент с транскрипцией и спикером"""
    start: float
    end: float
    text: str
    speaker: str
    confidence: float

    def to_dict(self) -> Dict:
        """Преобразование в словарь"""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
            "confidence": self.confidence,
        }




def run_whisperx_pipeline(
    audio_file: str,
    model_size: str,
    device: str,
    compute_type: str,
    batch_size: int,
    language: Optional[str],
    align: bool,
    diarize: bool,
    hf_token: Optional[str],
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> Tuple[List[MergedSegment], Dict]:
    """
    Run WhisperX transcription + optional alignment + diarization.

    Returns:
        (merged_segments, metadata)
    """
    import os
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torchaudio
    if not hasattr(torchaudio, "AudioMetaData"):
        try:
            from torchaudio.backend.common import AudioMetaData as _AudioMetaData
            torchaudio.AudioMetaData = _AudioMetaData
            _patch_status = "assigned_from_backend_common"
            _patch_error = None
        except Exception as _e:
            try:
                from torchaudio._backend.common import AudioMetaData as _AudioMetaData
                torchaudio.AudioMetaData = _AudioMetaData
                _patch_status = "assigned_from__backend_common"
                _patch_error = None
            except Exception as _e2:
                try:
                    from torchaudio._backend import AudioMetaData as _AudioMetaData  # type: ignore
                    torchaudio.AudioMetaData = _AudioMetaData
                    _patch_status = "assigned_from__backend"
                    _patch_error = None
                except Exception as _e3:
                    try:
                        from dataclasses import dataclass
                        @dataclass
                        class _AudioMetaData:
                            sample_rate: int
                            num_frames: int
                            num_channels: int
                            bits_per_sample: int
                            encoding: str
                        torchaudio.AudioMetaData = _AudioMetaData
                        _patch_status = "assigned_fallback_dataclass"
                        _patch_error = None
                    except Exception as _e4:
                        _patch_status = "failed"
                        _patch_error = repr((_e, _e2, _e3, _e4))

    if not hasattr(torchaudio, "list_audio_backends"):
        try:
            from torchaudio._backend.utils import list_audio_backends as _list_audio_backends
            torchaudio.list_audio_backends = _list_audio_backends
            _backend_patch_status = "assigned_from__backend_utils"
            _backend_patch_error = None
        except Exception as _e_b:
            try:
                def _fallback_list_audio_backends():
                    backends = []
                    try:
                        import soundfile  # noqa: F401
                        backends.append("soundfile")
                    except ImportError:
                        pass
                    if not backends:
                        backends.append("default")
                    return backends
                torchaudio.list_audio_backends = _fallback_list_audio_backends
                _backend_patch_status = "assigned_fallback_detector"
                _backend_patch_error = None
            except Exception as _e_b2:
                _backend_patch_status = "failed"
                _backend_patch_error = repr((_e_b, _e_b2))
    import whisperx

    print("🔎 WhisperX: загрузка аудио...")
    audio = whisperx.load_audio(audio_file)

    print(f"🧠 WhisperX: загрузка модели ({model_size}) на {device}...")
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    transcribe_kwargs: Dict = {"batch_size": batch_size}
    if language:
        transcribe_kwargs["language"] = language

    print("📝 WhisperX: транскрибация...")
    result = model.transcribe(audio, **transcribe_kwargs)

    detected_language = result.get("language") or language or "unknown"
    print(f"   Обнаруженный язык: {detected_language}")

    if align:
        print("⏱️ WhisperX: выравнивание (alignment)...")
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )

    diarize_kwargs: Dict = {}
    if num_speakers is not None:
        diarize_kwargs["min_speakers"] = num_speakers
        diarize_kwargs["max_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

    if diarize:
        from whisperx.diarize import DiarizationPipeline
        if not hf_token or hf_token == "your_huggingface_token_here":
            raise ValueError("Не указан HuggingFace токен (HF_TOKEN) для diarization")

        print("🎙️ WhisperX: diarization...")
        if device != "cuda":
            raise ValueError(f"Diarization поддерживается только на CUDA GPU. Получено устройство: {device}")
        
        diarize_model = DiarizationPipeline(
            use_auth_token=hf_token,
            device=device
        )
        diarize_segments = diarize_model(audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    merged_segments: List[MergedSegment] = []
    for seg in result["segments"]:
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker")
        if not speaker:
            speaker = "SPEAKER_00" if not diarize else "UNKNOWN"
        raw_confidence = seg.get("confidence")
        if raw_confidence is None:
            raw_confidence = 1.0 if speaker != "UNKNOWN" else 0.0
        confidence = float(raw_confidence)
        merged_segments.append(
            MergedSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=text,
                speaker=speaker,
                confidence=confidence
            )
        )

    metadata_out = {
        "language": detected_language,
        "model": model_size,
        "device": device,
        "compute_type": compute_type,
        "batch_size": batch_size,
        "aligned": align,
        "diarization_enabled": diarize,
    }

    return merged_segments, metadata_out
