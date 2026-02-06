"""
Модуль транскрибации аудио с использованием faster-whisper

Функциональность:
- Инициализация модели Whisper (faster-whisper)
- Транскрибация аудио с временными метками
- Поддержка настройки языка
- Возврат сегментов с timestamp'ами
"""

from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from faster_whisper import WhisperModel
import os


@dataclass
class TranscriptionSegment:
    """Сегмент транскрибации с временными метками"""
    start: float  # время начала в секундах
    end: float    # время окончания в секундах
    text: str     # распознанный текст
    
    def to_dict(self) -> Dict:
        """Преобразование в словарь"""
        return {
            'start': self.start,
            'end': self.end,
            'text': self.text
        }


class WhisperTranscriber:
    """Класс для транскрибации аудио с помощью Whisper"""
    
    # Доступные размеры моделей
    MODEL_SIZES = ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']
    
    def __init__(
        self,
        model_size: str = 'small',
        device: Literal['cuda'] = 'cuda',
        compute_type: str = 'float16',
        download_root: Optional[str] = None,
        cpu_threads: int = 0,
        num_workers: int = 1
    ):
        """
        Инициализация транскрибера
        
        Args:
            model_size: Размер модели ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
            device: Устройство для вычислений (только 'cuda')
            compute_type: Тип вычислений ('float16', 'float32')
                         - 'float16' оптимален для GPU
            download_root: Директория для кэширования моделей
            cpu_threads: Количество потоков CPU (0 = автоопределение)
            num_workers: Количество параллельных воркеров
        """
        if model_size not in self.MODEL_SIZES:
            raise ValueError(
                f"Недопустимый размер модели: {model_size}. "
                f"Доступные: {', '.join(self.MODEL_SIZES)}"
            )
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        
        print(f"⏳ Загрузка модели Whisper ({model_size}) на {device}...")
        print(f"   Размер модели: ~{self._get_model_size_mb(model_size)}")
        print(f"   Это может занять время при первом запуске...")
        
        # Инициализация модели с оптимизацией
        from tqdm import tqdm
        import time
        
        with tqdm(total=100, desc=f"Загрузка {model_size}", bar_format='{l_bar}{bar}| {n_fmt}%') as pbar:
            # Начинаем загрузку
            pbar.update(10)
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
                cpu_threads=cpu_threads,
                num_workers=num_workers
            )
            pbar.update(90)
        
        print(f"✓ Модель Whisper загружена!")
        if cpu_threads > 0:
            print(f"  └─ CPU потоков: {cpu_threads}")
        if num_workers > 1:
            print(f"  └─ Воркеров: {num_workers}")
    
    def transcribe(
        self,
        audio_file: str,
        language: str = 'ru',
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        vad_filter: bool = True,
        word_timestamps: bool = False,
        condition_on_previous_text: bool = True,
        compression_ratio_threshold: Optional[float] = None,
        log_prob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        hallucination_silence_threshold: Optional[float] = None,
        vad_parameters: Optional[Dict] = None
    ) -> List[TranscriptionSegment]:
        """
        Транскрибация аудио файла
        
        Args:
            audio_file: Путь к аудио файлу
            language: Код языка ('ru' для русского)
            beam_size: Размер beam search (выше = точнее, но медленнее)
            best_of: Количество кандидатов (выше = точнее, но медленнее)
            temperature: Температура семплирования (0.0 = детерминированно)
            vad_filter: Использовать Voice Activity Detection для фильтрации тишины
            word_timestamps: Возвращать временные метки для отдельных слов
            condition_on_previous_text: Учитывать предыдущий текст при декодировании
            compression_ratio_threshold: Порог для детекта повторов (None = дефолт модели)
            log_prob_threshold: Порог низкой уверенности (None = дефолт модели)
            no_speech_threshold: Порог тишины (None = дефолт модели)
            hallucination_silence_threshold: Порог тишины для подавления галлюцинаций
            vad_parameters: Параметры VAD фильтра (dict)
            
        Returns:
            Список сегментов транскрибации
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Аудио файл не найден: {audio_file}")
        
        print(f"Начинаем транскрибацию: {audio_file}")
        print(f"Язык: {language}, VAD фильтр: {vad_filter}")
        
        # Запуск транскрибации
        transcribe_kwargs = {
            "language": language,
            "beam_size": beam_size,
            "best_of": best_of,
            "temperature": temperature,
            "vad_filter": vad_filter,
            "word_timestamps": word_timestamps,
            "condition_on_previous_text": condition_on_previous_text
        }
        if compression_ratio_threshold is not None:
            transcribe_kwargs["compression_ratio_threshold"] = compression_ratio_threshold
        if log_prob_threshold is not None:
            transcribe_kwargs["log_prob_threshold"] = log_prob_threshold
        if no_speech_threshold is not None:
            transcribe_kwargs["no_speech_threshold"] = no_speech_threshold
        if hallucination_silence_threshold is not None:
            transcribe_kwargs["hallucination_silence_threshold"] = hallucination_silence_threshold
        if vad_parameters is not None:
            transcribe_kwargs["vad_parameters"] = vad_parameters

        segments_iterator, info = self.model.transcribe(
            audio_file,
            **transcribe_kwargs
        )
        
        # Информация о распознавании
        print(f"Обнаруженный язык: {info.language} (вероятность: {info.language_probability:.2f})")
        print(f"Длительность аудио: {info.duration:.2f} секунд ({info.duration/60:.1f} минут)")
        if hasattr(info, "duration_after_vad") and info.duration_after_vad is not None:
            print(f"Длительность после VAD: {info.duration_after_vad:.2f} секунд ({info.duration_after_vad/60:.1f} минут)")
        
        # Собираем сегменты с прогресс-баром
        segments = []
        
        # Создаем прогресс-бар
        from tqdm import tqdm
        pbar = tqdm(
            total=int(info.duration),
            desc="Транскрибация",
            unit="сек",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} сек [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        last_end = 0
        for segment in segments_iterator:
            segments.append(
                TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip()
                )
            )
            # Обновляем прогресс-бар
            progress = int(segment.end - last_end)
            if progress > 0:
                pbar.update(progress)
                last_end = segment.end
        
        pbar.close()
        print(f"✓ Транскрибация завершена! Всего сегментов: {len(segments)}")
        
        return segments
    
    def transcribe_with_progress(
        self,
        audio_file: str,
        language: str = 'ru',
        **kwargs
    ) -> tuple[List[TranscriptionSegment], Dict]:
        """
        Транскрибация с детальной информацией
        
        Returns:
            Кортеж (сегменты, метаданные)
        """
        segments = self.transcribe(audio_file, language=language, **kwargs)
        
        metadata = {
            'model': self.model_size,
            'language': language,
            'device': self.device,
            'num_segments': len(segments),
            'total_duration': segments[-1].end if segments else 0.0
        }
        
        return segments, metadata
    
    @staticmethod
    def _get_model_size_mb(model_size: str) -> str:
        """Примерный размер модели в MB"""
        sizes = {
            'tiny': '75',
            'base': '145',
            'small': '466',
            'medium': '1.5GB',
            'large-v2': '3GB',
            'large-v3': '3GB'
        }
        return sizes.get(model_size, 'неизвестно')
    
    def get_supported_languages(self) -> List[str]:
        """Получить список поддерживаемых языков"""
        # Whisper поддерживает 99 языков
        return [
            'ru',  # Русский
            'en',  # Английский
            'uk',  # Украинский
            'be',  # Белорусский
            'kk',  # Казахский
            # ... и многие другие
            # Полный список: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
        ]


def transcribe_audio(
    audio_file: str,
    model_size: str = 'small',
    language: str = 'ru',
    device: str = 'cuda'
) -> List[TranscriptionSegment]:
    """
    Вспомогательная функция для быстрой транскрибации
    
    Args:
        audio_file: Путь к аудио файлу
        model_size: Размер модели Whisper
        language: Код языка
        device: Устройство (только 'cuda')
        
    Returns:
        Список сегментов транскрибации
    """
    transcriber = WhisperTranscriber(
        model_size=model_size,
        device=device
    )
    return transcriber.transcribe(audio_file, language=language)
