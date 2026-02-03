"""
Экспорт транскрипции в JSON формат

Формат:
{
  "metadata": {
    "file": "input.mp3",
    "duration": 120.5,
    "num_speakers": 2,
    "language": "ru",
    "model": "faster-whisper-small"
  },
  "segments": [
    {
      "start": 5.0,
      "end": 12.0,
      "speaker": "SPEAKER_00",
      "text": "Текст",
      "confidence": 0.95
    }
  ]
}
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from ..merger import MergedSegment


def export_to_json(
    segments: List[MergedSegment],
    output_file: str,
    metadata: Optional[Dict] = None,
    pretty: bool = True
) -> None:
    """
    Экспорт транскрипции в JSON файл
    
    Args:
        segments: Список объединенных сегментов
        output_file: Путь к выходному файлу
        metadata: Дополнительные метаданные
        pretty: Форматировать ли JSON с отступами
    """
    if not segments:
        raise ValueError("Список сегментов пуст")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Собираем статистику
    speakers = sorted(set(seg.speaker for seg in segments if seg.speaker != 'UNKNOWN'))
    
    # Формируем структуру данных
    data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'duration': segments[-1].end if segments else 0.0,
            'num_segments': len(segments),
            'num_speakers': len(speakers),
            'speakers': speakers,
            **(metadata or {})
        },
        'segments': [seg.to_dict() for seg in segments]
    }
    
    # Записываем в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False)
    
    print(f"JSON файл сохранен: {output_file}")


def export_to_json_compact(
    segments: List[MergedSegment],
    output_file: str,
    include_confidence: bool = False
) -> None:
    """
    Экспорт в компактный JSON формат (только сегменты)
    
    Args:
        segments: Список объединенных сегментов
        output_file: Путь к выходному файлу
        include_confidence: Включать ли поле confidence
    """
    if not segments:
        raise ValueError("Список сегментов пуст")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Формируем компактную структуру
    compact_segments = []
    for seg in segments:
        item = {
            'start': seg.start,
            'end': seg.end,
            'speaker': seg.speaker,
            'text': seg.text
        }
        if include_confidence:
            item['confidence'] = seg.confidence
        compact_segments.append(item)
    
    # Записываем в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(compact_segments, f, ensure_ascii=False, indent=2)
    
    print(f"Компактный JSON файл сохранен: {output_file}")


def load_from_json(input_file: str) -> tuple[List[MergedSegment], Dict]:
    """
    Загрузка транскрипции из JSON файла
    
    Args:
        input_file: Путь к JSON файлу
        
    Returns:
        Кортеж (сегменты, метаданные)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Проверяем формат
    if isinstance(data, dict) and 'segments' in data:
        # Полный формат
        segments = [
            MergedSegment(
                start=seg['start'],
                end=seg['end'],
                text=seg['text'],
                speaker=seg['speaker'],
                confidence=seg.get('confidence', 0.0)
            )
            for seg in data['segments']
        ]
        metadata = data.get('metadata', {})
    elif isinstance(data, list):
        # Компактный формат
        segments = [
            MergedSegment(
                start=seg['start'],
                end=seg['end'],
                text=seg['text'],
                speaker=seg['speaker'],
                confidence=seg.get('confidence', 0.0)
            )
            for seg in data
        ]
        metadata = {}
    else:
        raise ValueError("Неподдерживаемый формат JSON файла")
    
    return segments, metadata
