"""
Экспорт транскрипции в текстовый формат

Формат:
[SPEAKER_00] (00:00:05 - 00:00:12)
Это пример транскрибированного текста от первого спикера.

[SPEAKER_01] (00:00:13 - 00:00:20)
А это текст от второго спикера.
"""

from typing import List
from pathlib import Path
from ..merger import MergedSegment


def format_timestamp(seconds: float) -> str:
    """
    Форматирование временной метки в HH:MM:SS формат
    
    Args:
        seconds: Время в секундах
        
    Returns:
        Отформатированная строка времени
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def export_to_text(
    segments: List[MergedSegment],
    output_file: str,
    show_confidence: bool = False,
    group_by_speaker: bool = True
) -> None:
    """
    Экспорт транскрипции в текстовый файл
    
    Args:
        segments: Список объединенных сегментов
        output_file: Путь к выходному файлу
        show_confidence: Показывать ли уверенность в определении спикера
        group_by_speaker: Группировать ли последовательные сегменты одного спикера
    """
    if not segments:
        raise ValueError("Список сегментов пуст")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if group_by_speaker:
            # Группируем последовательные сегменты одного спикера
            current_speaker = None
            current_texts = []
            current_start = None
            current_end = None
            
            for seg in segments:
                if seg.speaker != current_speaker:
                    # Записываем предыдущую группу
                    if current_texts:
                        _write_speaker_block(
                            f, current_speaker, current_start, current_end,
                            ' '.join(current_texts), show_confidence,
                            sum([s.confidence for s in segments if s.speaker == current_speaker]) / 
                            len([s for s in segments if s.speaker == current_speaker])
                        )
                        f.write('\n')
                    
                    # Начинаем новую группу
                    current_speaker = seg.speaker
                    current_texts = [seg.text]
                    current_start = seg.start
                    current_end = seg.end
                else:
                    # Тот же спикер
                    current_texts.append(seg.text)
                    current_end = seg.end
            
            # Записываем последнюю группу
            if current_texts:
                _write_speaker_block(
                    f, current_speaker, current_start, current_end,
                    ' '.join(current_texts), show_confidence,
                    sum([s.confidence for s in segments if s.speaker == current_speaker]) / 
                    len([s for s in segments if s.speaker == current_speaker])
                )
        else:
            # Каждый сегмент отдельно
            for seg in segments:
                _write_speaker_block(
                    f, seg.speaker, seg.start, seg.end,
                    seg.text, show_confidence, seg.confidence
                )
                f.write('\n')
    
    print(f"Текстовый файл сохранен: {output_file}")


def _write_speaker_block(
    f,
    speaker: str,
    start: float,
    end: float,
    text: str,
    show_confidence: bool,
    confidence: float
) -> None:
    """Записать блок текста спикера"""
    time_range = f"{format_timestamp(start)} - {format_timestamp(end)}"
    
    if show_confidence and speaker != 'UNKNOWN':
        header = f"[{speaker}] ({time_range}) [confidence: {confidence:.2f}]"
    else:
        header = f"[{speaker}] ({time_range})"
    
    f.write(f"{header}\n")
    f.write(f"{text}\n")


def export_to_simple_text(
    segments: List[MergedSegment],
    output_file: str,
    include_speakers: bool = True
) -> None:
    """
    Экспорт в простой текстовый формат без временных меток
    
    Args:
        segments: Список объединенных сегментов
        output_file: Путь к выходному файлу
        include_speakers: Включать ли метки спикеров
    """
    if not segments:
        raise ValueError("Список сегментов пуст")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        current_speaker = None
        
        for seg in segments:
            # Добавляем метку спикера при смене
            if include_speakers and seg.speaker != current_speaker:
                if current_speaker is not None:
                    f.write('\n\n')
                f.write(f"[{seg.speaker}]\n")
                current_speaker = seg.speaker
            
            f.write(f"{seg.text} ")
    
    print(f"Простой текстовый файл сохранен: {output_file}")
