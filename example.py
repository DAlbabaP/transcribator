"""
Пример программного использования Transcribator

Этот скрипт показывает как использовать модули напрямую из Python кода
"""

from src.audio_processor import AudioProcessor
from src.transcriber import WhisperTranscriber
from src.diarizer import SpeakerDiarizer
from src.merger import TranscriptionMerger
from src.exporters.text_exporter import export_to_text
from src.exporters.json_exporter import export_to_json
from src.exporters.srt_exporter import export_to_srt
import os
from dotenv import load_dotenv


def transcribe_file_simple(audio_file: str, output_dir: str = "./output"):
    """
    Простой пример: транскрибация без определения спикеров
    """
    print("=== Простая транскрибация ===")
    
    # 1. Предобработка аудио
    processor = AudioProcessor()
    processed_audio = processor.preprocess_audio(audio_file)
    
    # 2. Транскрибация
    transcriber = WhisperTranscriber(model_size='small', device='cpu')
    segments = transcriber.transcribe(processed_audio, language='ru')
    
    # 3. Экспорт (без спикеров)
    from src.merger import MergedSegment
    merged_segments = [
        MergedSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker='SPEAKER_00',
            confidence=1.0
        )
        for seg in segments
    ]
    
    # Сохраняем результаты
    os.makedirs(output_dir, exist_ok=True)
    export_to_text(merged_segments, f"{output_dir}/transcript.txt")
    
    print(f"✓ Готово! Результаты в {output_dir}")
    return merged_segments


def transcribe_file_with_speakers(audio_file: str, hf_token: str, output_dir: str = "./output"):
    """
    Полный пример: транскрибация с определением спикеров
    """
    print("=== Транскрибация с определением спикеров ===")
    
    # 1. Предобработка аудио
    print("\n[1/5] Предобработка аудио...")
    processor = AudioProcessor()
    processed_audio = processor.preprocess_audio(audio_file)
    
    # 2. Транскрибация
    print("\n[2/5] Транскрибация...")
    transcriber = WhisperTranscriber(model_size='small', device='cpu')
    transcription = transcriber.transcribe(processed_audio, language='ru')
    print(f"✓ Получено {len(transcription)} сегментов")
    
    # 3. Определение спикеров
    print("\n[3/5] Определение спикеров...")
    diarizer = SpeakerDiarizer(hf_token=hf_token, device='cpu')
    diarization = diarizer.diarize(processed_audio)
    print(f"✓ Определено спикеров: {len(set(seg.speaker for seg in diarization))}")
    
    # 4. Объединение результатов
    print("\n[4/5] Объединение результатов...")
    merger = TranscriptionMerger(min_overlap_ratio=0.5)
    merged = merger.merge(transcription, diarization)
    stats = merger.get_statistics(merged)
    print(f"✓ Объединено: {len(merged)} сегментов, {stats['num_speakers']} спикеров")
    
    # 5. Экспорт во все форматы
    print("\n[5/5] Экспорт результатов...")
    os.makedirs(output_dir, exist_ok=True)
    
    export_to_text(merged, f"{output_dir}/transcript.txt")
    export_to_json(merged, f"{output_dir}/transcript.json")
    export_to_srt(merged, f"{output_dir}/transcript.srt")
    
    print(f"\n✓ Готово! Результаты в {output_dir}")
    print(f"  Файлы: transcript.txt, transcript.json, transcript.srt")
    
    return merged


def analyze_transcription(segments):
    """
    Анализ транскрипции
    """
    print("\n=== Анализ транскрипции ===")
    
    # Уникальные спикеры
    speakers = set(seg.speaker for seg in segments)
    print(f"Количество спикеров: {len(speakers)}")
    
    # Статистика по спикерам
    for speaker in sorted(speakers):
        speaker_segs = [s for s in segments if s.speaker == speaker]
        total_time = sum(s.end - s.start for s in speaker_segs)
        total_words = sum(len(s.text.split()) for s in speaker_segs)
        
        print(f"\n{speaker}:")
        print(f"  Сегментов: {len(speaker_segs)}")
        print(f"  Время говорения: {total_time:.1f} сек ({total_time/60:.1f} мин)")
        print(f"  Слов: {total_words}")
        print(f"  Первые слова: {speaker_segs[0].text[:50]}...")


if __name__ == '__main__':
    # Загружаем переменные окружения
    load_dotenv()
    
    # Пример 1: Простая транскрибация (без спикеров)
    # Раскомментируйте для использования:
    # segments = transcribe_file_simple("path/to/audio.mp3")
    
    # Пример 2: Полная транскрибация с определением спикеров
    # Раскомментируйте и укажите свои данные:
    # HF_TOKEN = os.getenv('HF_TOKEN')
    # segments = transcribe_file_with_speakers(
    #     audio_file="path/to/audio.mp3",
    #     hf_token=HF_TOKEN,
    #     output_dir="./output"
    # )
    # analyze_transcription(segments)
    
    print("Примеры готовы к использованию!")
    print("Раскомментируйте нужный пример и запустите скрипт.")
