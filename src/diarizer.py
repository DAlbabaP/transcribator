"""
Модуль определения спикеров (Speaker Diarization) с использованием pyannote.audio

Функциональность:
- Инициализация pyannote.audio pipeline
- Определение количества спикеров и их временных меток
- Возврат сегментов со спикерами
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import os
from pyannote.audio import Pipeline
import torch
import torchaudio


@dataclass
class SpeakerSegment:
    """Сегмент с информацией о спикере"""
    start: float  # время начала в секундах
    end: float    # время окончания в секундах
    speaker: str  # идентификатор спикера ('SPEAKER_00', 'SPEAKER_01', etc.)
    
    def to_dict(self) -> Dict:
        """Преобразование в словарь"""
        return {
            'start': self.start,
            'end': self.end,
            'speaker': self.speaker
        }


class SpeakerDiarizer:
    """Класс для определения спикеров в аудио"""
    
    def __init__(
        self,
        hf_token: str,
        device: Optional[str] = None,
        pipeline_name: str = "pyannote/speaker-diarization-3.1"
    ):
        """
        Инициализация diarizer
        
        Args:
            hf_token: HuggingFace access token
            device: Устройство ('cpu', 'cuda' или None для автоопределения)
            pipeline_name: Название pipeline на HuggingFace Hub
            
        Примечание:
            Для использования pyannote.audio необходимо:
            1. Зарегистрироваться на HuggingFace (https://huggingface.co)
            2. Получить access token (https://huggingface.co/settings/tokens)
            3. Принять условия использования моделей:
               - https://huggingface.co/pyannote/speaker-diarization-3.1
               - https://huggingface.co/pyannote/segmentation-3.0
        """
        if not hf_token or hf_token == "your_huggingface_token_here":
            raise ValueError(
                "Необходим HuggingFace access token!\n"
                "1. Зарегистрируйтесь на https://huggingface.co\n"
                "2. Создайте токен: https://huggingface.co/settings/tokens\n"
                "3. Примите лицензии моделей:\n"
                "   - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "   - https://huggingface.co/pyannote/segmentation-3.0\n"
                "4. Укажите токен в .env файле или передайте в конструктор"
            )
        
        self.hf_token = hf_token
        
        # Определяем устройство
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        print(f"⏳ Загрузка pyannote.audio pipeline на {device}...")
        print(f"   Размер моделей: ~1.5 GB")
        print(f"   Это может занять время при первом запуске...")
        
        try:
            from tqdm import tqdm
            
            # Инициализация pipeline с прогресс-баром
            with tqdm(total=100, desc="Загрузка pyannote", bar_format='{l_bar}{bar}| {n_fmt}%') as pbar:
                pbar.update(10)
                # В новой версии pyannote.audio используется 'token' вместо 'use_auth_token'
                try:
                    self.pipeline = Pipeline.from_pretrained(
                        pipeline_name,
                        token=hf_token
                    )
                except TypeError:
                    # Для старых версий
                    self.pipeline = Pipeline.from_pretrained(
                        pipeline_name,
                        use_auth_token=hf_token
                    )
                pbar.update(70)
                
                # Перемещаем на нужное устройство
                self.pipeline.to(self.device)
                pbar.update(20)
            
            print(f"✓ Pipeline pyannote.audio загружен!")
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                raise RuntimeError(
                    "Ошибка авторизации HuggingFace!\n"
                    "Проверьте:\n"
                    "1. Токен правильный?\n"
                    "2. Вы приняли лицензии моделей?\n"
                    "   - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                    "   - https://huggingface.co/pyannote/segmentation-3.0"
                )
            else:
                raise RuntimeError(f"Ошибка при загрузке pipeline: {error_msg}")
    
    def diarize(
        self,
        audio_file: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        num_speakers: Optional[int] = None
    ) -> List[SpeakerSegment]:
        """
        Определение спикеров в аудио файле
        
        Args:
            audio_file: Путь к аудио файлу
            min_speakers: Минимальное количество спикеров (опционально)
            max_speakers: Максимальное количество спикеров (опционально)
            num_speakers: Точное количество спикеров (опционально)
            
        Returns:
            Список сегментов со спикерами
            
        Примечание:
            Если количество спикеров не указано, pipeline определит автоматически
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Аудио файл не найден: {audio_file}")
        
        print(f"🎤 Начинаем определение спикеров...")
        
        # Параметры для pipeline
        params = {}
        if num_speakers is not None:
            params['num_speakers'] = num_speakers
            print(f"   Ожидаемое количество спикеров: {num_speakers}")
        elif min_speakers is not None or max_speakers is not None:
            if min_speakers is not None:
                params['min_speakers'] = min_speakers
                print(f"   Минимум спикеров: {min_speakers}")
            if max_speakers is not None:
                params['max_speakers'] = max_speakers
                print(f"   Максимум спикеров: {max_speakers}")
        else:
            print("   Количество спикеров будет определено автоматически")
        
        # Запуск diarization с прогресс-баром
        from tqdm import tqdm
        
        try:
            print("   Загрузка аудио в память...")
            # Загружаем аудио напрямую в память (обход проблемы с AudioDecoder на Windows)
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # Конвертируем в моно если нужно
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Создаем словарь с аудио данными
            audio_in_memory = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
            
            print("   Анализ аудио...")
            with tqdm(total=100, desc="Diarization", bar_format='{l_bar}{bar}| {n_fmt}%') as pbar:
                pbar.update(10)
                diarization = self.pipeline(audio_in_memory, **params)
                pbar.update(90)
        except Exception as e:
            raise RuntimeError(f"Ошибка при определении спикеров: {str(e)}")
        
        # Собираем сегменты
        print("   Обработка результатов...")
        
        segments = []
        
        # pyannote.audio 4.0+ возвращает DiarizeOutput dataclass
        # Проверяем разные варианты структуры данных
        if hasattr(diarization, 'segments'):
            # Прямой список сегментов
            for seg in diarization.segments:
                segments.append(
                    SpeakerSegment(
                        start=seg.start,
                        end=seg.end,
                        speaker=seg.speaker
                    )
                )
        elif hasattr(diarization, 'diarization'):
            # Вложенный Annotation объект
            annotation = diarization.diarization
            for segment, track, label in annotation.itertracks(yield_label=True):
                segments.append(
                    SpeakerSegment(
                        start=segment.start,
                        end=segment.end,
                        speaker=label
                    )
                )
        else:
            # Для dataclass с полями (pyannote.audio 4.0+)
            import dataclasses
            if dataclasses.is_dataclass(diarization):
                # Ищем поле с Annotation данными
                for field in dataclasses.fields(diarization):
                    value = getattr(diarization, field.name)
                    if hasattr(value, 'itertracks'):
                        for segment, track, label in value.itertracks(yield_label=True):
                            segments.append(
                                SpeakerSegment(
                                    start=segment.start,
                                    end=segment.end,
                                    speaker=label
                                )
                            )
                        break
            else:
                # Последняя попытка - прямой Annotation объект
                for segment, track, label in diarization.itertracks(yield_label=True):
                    segments.append(
                        SpeakerSegment(
                            start=segment.start,
                            end=segment.end,
                            speaker=label
                        )
                    )
        
        # Получаем уникальных спикеров
        unique_speakers = sorted(set(seg.speaker for seg in segments))
        
        print(f"✓ Определение спикеров завершено!")
        print(f"  └─ Обнаружено спикеров: {len(unique_speakers)}")
        print(f"  └─ Всего сегментов: {len(segments)}")
        
        return segments
    
    def diarize_with_stats(
        self,
        audio_file: str,
        **kwargs
    ) -> tuple[List[SpeakerSegment], Dict]:
        """
        Определение спикеров с детальной статистикой
        
        Returns:
            Кортеж (сегменты, статистика)
        """
        segments = self.diarize(audio_file, **kwargs)
        
        # Собираем статистику
        unique_speakers = sorted(set(seg.speaker for seg in segments))
        
        # Общее время говорения для каждого спикера
        speaker_duration = {}
        for speaker in unique_speakers:
            total_time = sum(
                seg.end - seg.start 
                for seg in segments 
                if seg.speaker == speaker
            )
            speaker_duration[speaker] = total_time
        
        stats = {
            'num_speakers': len(unique_speakers),
            'speakers': unique_speakers,
            'num_segments': len(segments),
            'speaker_duration': speaker_duration,
            'total_duration': segments[-1].end if segments else 0.0
        }
        
        return segments, stats
    
    def get_speaker_timeline(self, segments: List[SpeakerSegment]) -> Dict[str, List[tuple]]:
        """
        Получить временную линию для каждого спикера
        
        Args:
            segments: Список сегментов со спикерами
            
        Returns:
            Словарь {speaker: [(start, end), ...]}
        """
        timeline = {}
        for segment in segments:
            if segment.speaker not in timeline:
                timeline[segment.speaker] = []
            timeline[segment.speaker].append((segment.start, segment.end))
        
        return timeline


def diarize_audio(
    audio_file: str,
    hf_token: str,
    num_speakers: Optional[int] = None,
    device: Optional[str] = None
) -> List[SpeakerSegment]:
    """
    Вспомогательная функция для быстрого определения спикеров
    
    Args:
        audio_file: Путь к аудио файлу
        hf_token: HuggingFace access token
        num_speakers: Количество спикеров (опционально)
        device: Устройство ('cpu' или 'cuda')
        
    Returns:
        Список сегментов со спикерами
    """
    diarizer = SpeakerDiarizer(hf_token=hf_token, device=device)
    return diarizer.diarize(audio_file, num_speakers=num_speakers)
