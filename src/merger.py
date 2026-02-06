"""
Модуль объединения результатов транскрибации и определения спикеров

Функциональность:
- Объединение сегментов транскрипции с информацией о спикерах
- Алгоритм поиска спикера по временному пересечению
- Разрешение конфликтов при пересечении нескольких спикеров
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from .transcriber import TranscriptionSegment
from .diarizer import SpeakerSegment


@dataclass
class MergedSegment:
    """Объединенный сегмент с транскрипцией и спикером"""
    start: float      # время начала в секундах
    end: float        # время окончания в секундах
    text: str         # распознанный текст
    speaker: str      # идентификатор спикера или 'UNKNOWN'
    confidence: float # уверенность в назначении спикера (0.0 - 1.0)
    
    def to_dict(self) -> Dict:
        """Преобразование в словарь"""
        return {
            'start': self.start,
            'end': self.end,
            'text': self.text,
            'speaker': self.speaker,
            'confidence': self.confidence
        }


class TranscriptionMerger:
    """Класс для объединения транскрипции и diarization"""
    
    def __init__(self, min_overlap_ratio: float = 0.5, unknown_policy: str = "keep"):
        """
        Инициализация merger
        
        Args:
            min_overlap_ratio: Минимальное отношение пересечения для назначения спикера (0.0 - 1.0)
                              Если пересечение меньше этого значения, спикер будет UNKNOWN
            unknown_policy: Политика для сегментов ниже порога пересечения:
                            - "keep": оставлять UNKNOWN
                            - "best_overlap": назначать спикера с максимальным пересечением
        """
        if not 0.0 <= min_overlap_ratio <= 1.0:
            raise ValueError("min_overlap_ratio должен быть между 0.0 и 1.0")
        if unknown_policy not in {"keep", "best_overlap"}:
            raise ValueError("unknown_policy должен быть 'keep' или 'best_overlap'")
        
        self.min_overlap_ratio = min_overlap_ratio
        self.unknown_policy = unknown_policy
    
    def merge(
        self,
        transcription: List[TranscriptionSegment],
        diarization: List[SpeakerSegment]
    ) -> List[MergedSegment]:
        """
        Объединение транскрипции и diarization
        
        Args:
            transcription: Список сегментов транскрипции
            diarization: Список сегментов с спикерами
            
        Returns:
            Список объединенных сегментов
        """
        if not transcription:
            raise ValueError("Список транскрипции пуст")
        
        if not diarization:
            print("Внимание: список diarization пуст, все сегменты будут помечены как UNKNOWN")
            return [
                MergedSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker='UNKNOWN',
                    confidence=0.0
                )
                for seg in transcription
            ]
        
        print(f"Объединяем {len(transcription)} сегментов транскрипции с {len(diarization)} сегментами diarization...")
        
        merged_segments = []
        
        for trans_seg in transcription:
            # Находим спикера для этого сегмента транскрипции
            speaker, confidence = self._find_speaker(trans_seg, diarization)
            
            merged_segments.append(
                MergedSegment(
                    start=trans_seg.start,
                    end=trans_seg.end,
                    text=trans_seg.text,
                    speaker=speaker,
                    confidence=confidence
                )
            )
        
        # Статистика
        unknown_count = sum(1 for seg in merged_segments if seg.speaker == 'UNKNOWN')
        if unknown_count > 0:
            print(f"Внимание: {unknown_count} сегментов не удалось сопоставить со спикером")
        
        print(f"Объединение завершено! Всего сегментов: {len(merged_segments)}")
        
        return merged_segments
    
    def _find_speaker(
        self,
        trans_seg: TranscriptionSegment,
        diarization: List[SpeakerSegment]
    ) -> tuple[str, float]:
        """
        Найти спикера для сегмента транскрипции
        
        Args:
            trans_seg: Сегмент транскрипции
            diarization: Список сегментов с спикерами
            
        Returns:
            Кортеж (идентификатор спикера, уверенность)
        """
        best_speaker = 'UNKNOWN'
        max_overlap = 0.0
        
        # Длительность сегмента транскрипции
        trans_duration = trans_seg.end - trans_seg.start
        
        # Проверяем пересечение со всеми сегментами diarization
        for diar_seg in diarization:
            # Вычисляем пересечение
            overlap_start = max(trans_seg.start, diar_seg.start)
            overlap_end = min(trans_seg.end, diar_seg.end)
            
            if overlap_start < overlap_end:
                # Есть пересечение
                overlap_duration = overlap_end - overlap_start
                overlap_ratio = overlap_duration / trans_duration
                
                # Если это лучшее пересечение, запоминаем
                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio
                    best_speaker = diar_seg.speaker
        
        # Определяем уверенность
        if max_overlap < self.min_overlap_ratio:
            # Пересечение слишком мало
            if self.unknown_policy == "best_overlap" and max_overlap > 0.0:
                return best_speaker, max_overlap
            return 'UNKNOWN', max_overlap
        
        return best_speaker, max_overlap
    
    def get_statistics(self, merged_segments: List[MergedSegment]) -> Dict:
        """
        Получить статистику по объединенным сегментам
        
        Args:
            merged_segments: Список объединенных сегментов
            
        Returns:
            Словарь со статистикой
        """
        # Уникальные спикеры (кроме UNKNOWN)
        speakers = sorted(set(
            seg.speaker for seg in merged_segments 
            if seg.speaker != 'UNKNOWN'
        ))
        
        # Количество сегментов на спикера
        speaker_counts = {}
        speaker_durations = {}
        speaker_word_counts = {}
        
        for speaker in speakers + ['UNKNOWN']:
            segs = [s for s in merged_segments if s.speaker == speaker]
            speaker_counts[speaker] = len(segs)
            speaker_durations[speaker] = sum(s.end - s.start for s in segs)
            speaker_word_counts[speaker] = sum(len(s.text.split()) for s in segs)
        
        # Средняя уверенность
        confidences = [seg.confidence for seg in merged_segments if seg.speaker != 'UNKNOWN']
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'num_speakers': len(speakers),
            'speakers': speakers,
            'unknown_segments': speaker_counts.get('UNKNOWN', 0),
            'speaker_segments': speaker_counts,
            'speaker_duration': speaker_durations,
            'speaker_word_count': speaker_word_counts,
            'average_confidence': avg_confidence,
            'total_segments': len(merged_segments)
        }
    
    def group_by_speaker(
        self,
        merged_segments: List[MergedSegment],
        merge_consecutive: bool = True
    ) -> Dict[str, List[MergedSegment]]:
        """
        Группировать сегменты по спикерам
        
        Args:
            merged_segments: Список объединенных сегментов
            merge_consecutive: Объединять ли последовательные сегменты одного спикера
            
        Returns:
            Словарь {speaker: [segments]}
        """
        grouped = {}
        
        if not merge_consecutive:
            # Простая группировка
            for seg in merged_segments:
                if seg.speaker not in grouped:
                    grouped[seg.speaker] = []
                grouped[seg.speaker].append(seg)
        else:
            # Объединяем последовательные сегменты
            current_speaker = None
            current_segments = []
            
            for seg in merged_segments:
                if seg.speaker != current_speaker:
                    # Смена спикера
                    if current_segments:
                        # Сохраняем предыдущую группу
                        if current_speaker not in grouped:
                            grouped[current_speaker] = []
                        grouped[current_speaker].append(
                            self._merge_consecutive_segments(current_segments)
                        )
                    
                    current_speaker = seg.speaker
                    current_segments = [seg]
                else:
                    # Тот же спикер
                    current_segments.append(seg)
            
            # Сохраняем последнюю группу
            if current_segments:
                if current_speaker not in grouped:
                    grouped[current_speaker] = []
                grouped[current_speaker].append(
                    self._merge_consecutive_segments(current_segments)
                )
        
        return grouped
    
    def _merge_consecutive_segments(
        self,
        segments: List[MergedSegment]
    ) -> MergedSegment:
        """
        Объединить последовательные сегменты одного спикера
        
        Args:
            segments: Список сегментов для объединения
            
        Returns:
            Объединенный сегмент
        """
        if not segments:
            raise ValueError("Список сегментов пуст")
        
        if len(segments) == 1:
            return segments[0]
        
        # Объединяем текст
        combined_text = ' '.join(seg.text for seg in segments)
        
        # Средняя уверенность
        avg_confidence = sum(seg.confidence for seg in segments) / len(segments)
        
        return MergedSegment(
            start=segments[0].start,
            end=segments[-1].end,
            text=combined_text,
            speaker=segments[0].speaker,
            confidence=avg_confidence
        )


def merge_transcription_and_speakers(
    transcription: List[TranscriptionSegment],
    diarization: List[SpeakerSegment],
    min_overlap_ratio: float = 0.5,
    unknown_policy: str = "keep"
) -> List[MergedSegment]:
    """
    Вспомогательная функция для объединения транскрипции и diarization
    
    Args:
        transcription: Список сегментов транскрипции
        diarization: Список сегментов с спикерами
        min_overlap_ratio: Минимальное отношение пересечения
        unknown_policy: Политика для сегментов ниже порога пересечения
        
    Returns:
        Список объединенных сегментов
    """
    merger = TranscriptionMerger(
        min_overlap_ratio=min_overlap_ratio,
        unknown_policy=unknown_policy
    )
    return merger.merge(transcription, diarization)
