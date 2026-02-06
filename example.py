"""
Пример программного использования Transcribator

Этот скрипт показывает как использовать модули напрямую из Python кода
"""

from src.audio_processor import AudioProcessor
from src.whisperx_pipeline import run_whisperx_pipeline
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
    
    # 2. WhisperX (без diarization)
    merged_segments, _ = run_whisperx_pipeline(
        audio_file=processed_audio,
        model_size="small",
        device="cuda",
        compute_type="float16",
        batch_size=16,
        language="ru",
        align=True,
        diarize=False,
        hf_token=None,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None
    )
    
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
    
    # 2. WhisperX (с diarization)
    print("\n[2/3] WhisperX...")
    merged, _ = run_whisperx_pipeline(
        audio_file=processed_audio,
        model_size="small",
        device="cuda",
        compute_type="float16",
        batch_size=16,
        language="ru",
        align=True,
        diarize=True,
        hf_token=hf_token,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None
    )
    print(f"✓ Объединено: {len(merged)} сегментов")
    
    # 3. Экспорт во все форматы
    print("\n[3/3] Экспорт результатов...")
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
