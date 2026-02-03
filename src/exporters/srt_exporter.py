"""
Экспорт транскрипции в форматы субтитров (SRT и VTT)

SRT формат:
1
00:00:05,000 --> 00:00:12,000
[SPEAKER_00] Это пример транскрибированного текста

2
00:00:13,000 --> 00:00:20,000
[SPEAKER_01] А это текст от второго спикера

VTT формат:
WEBVTT

00:00:05.000 --> 00:00:12.000
[SPEAKER_00] Это пример транскрибированного текста

00:00:13.000 --> 00:00:20.000
[SPEAKER_01] А это текст от второго спикера
"""

from typing import List
from pathlib import Path
from ..merger import MergedSegment


def format_srt_timestamp(seconds: float) -> str:
    """
    Форматирование временной метки для SRT (HH:MM:SS,mmm)
    
    Args:
        seconds: Время в секундах
        
    Returns:
        Отформатированная строка времени
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """
    Форматирование временной метки для VTT (HH:MM:SS.mmm)
    
    Args:
        seconds: Время в секундах
        
    Returns:
        Отформатированная строка времени
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def export_to_srt(
    segments: List[MergedSegment],
    output_file: str,
    include_speakers: bool = True,
    max_line_length: int = 42
) -> None:
    """
    Экспорт транскрипции в SRT файл
    
    Args:
        segments: Список объединенных сегментов
        output_file: Путь к выходному файлу
        include_speakers: Включать ли метки спикеров
        max_line_length: Максимальная длина строки (для переноса)
    """
    if not segments:
        raise ValueError("Список сегментов пуст")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, seg in enumerate(segments, start=1):
            # Номер субтитра
            f.write(f"{idx}\n")
            
            # Временной диапазон
            start_time = format_srt_timestamp(seg.start)
            end_time = format_srt_timestamp(seg.end)
            f.write(f"{start_time} --> {end_time}\n")
            
            # Текст
            if include_speakers:
                text = f"[{seg.speaker}] {seg.text}"
            else:
                text = seg.text
            
            # Разбиваем длинные строки
            lines = _split_text(text, max_line_length)
            for line in lines:
                f.write(f"{line}\n")
            
            # Пустая строка между субтитрами
            f.write("\n")
    
    print(f"SRT файл сохранен: {output_file}")


def export_to_vtt(
    segments: List[MergedSegment],
    output_file: str,
    include_speakers: bool = True,
    max_line_length: int = 42
) -> None:
    """
    Экспорт транскрипции в WebVTT файл
    
    Args:
        segments: Список объединенных сегментов
        output_file: Путь к выходному файлу
        include_speakers: Включать ли метки спикеров
        max_line_length: Максимальная длина строки (для переноса)
    """
    if not segments:
        raise ValueError("Список сегментов пуст")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Заголовок VTT
        f.write("WEBVTT\n\n")
        
        for seg in segments:
            # Временной диапазон
            start_time = format_vtt_timestamp(seg.start)
            end_time = format_vtt_timestamp(seg.end)
            f.write(f"{start_time} --> {end_time}\n")
            
            # Текст
            if include_speakers:
                text = f"[{seg.speaker}] {seg.text}"
            else:
                text = seg.text
            
            # Разбиваем длинные строки
            lines = _split_text(text, max_line_length)
            for line in lines:
                f.write(f"{line}\n")
            
            # Пустая строка между субтитрами
            f.write("\n")
    
    print(f"VTT файл сохранен: {output_file}")


def _split_text(text: str, max_length: int) -> List[str]:
    """
    Разбить текст на строки заданной длины
    
    Args:
        text: Текст для разбиения
        max_length: Максимальная длина строки
        
    Returns:
        Список строк
    """
    if len(text) <= max_length:
        return [text]
    
    lines = []
    words = text.split()
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + (1 if current_line else 0)  # +1 для пробела
        
        if current_length + word_length <= max_length:
            current_line.append(word)
            current_length += word_length
        else:
            # Сохраняем текущую строку
            if current_line:
                lines.append(' '.join(current_line))
            
            # Начинаем новую строку
            current_line = [word]
            current_length = len(word)
    
    # Добавляем последнюю строку
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def export_to_srt_with_speaker_colors(
    segments: List[MergedSegment],
    output_file: str
) -> None:
    """
    Экспорт в SRT с цветовой разметкой спикеров (поддерживается не всеми плеерами)
    
    Args:
        segments: Список объединенных сегментов
        output_file: Путь к выходному файлу
    """
    if not segments:
        raise ValueError("Список сегментов пуст")
    
    # Цвета для спикеров
    speaker_colors = {}
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    # Назначаем цвета спикерам
    unique_speakers = sorted(set(seg.speaker for seg in segments))
    for idx, speaker in enumerate(unique_speakers):
        speaker_colors[speaker] = colors[idx % len(colors)]
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, seg in enumerate(segments, start=1):
            f.write(f"{idx}\n")
            
            start_time = format_srt_timestamp(seg.start)
            end_time = format_srt_timestamp(seg.end)
            f.write(f"{start_time} --> {end_time}\n")
            
            # Добавляем цветовую разметку
            color = speaker_colors.get(seg.speaker, '#FFFFFF')
            f.write(f'<font color="{color}">[{seg.speaker}]</font> {seg.text}\n')
            f.write("\n")
    
    print(f"SRT файл с цветами сохранен: {output_file}")
