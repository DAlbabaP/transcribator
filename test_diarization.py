"""
Тестовый скрипт для проверки только diarization (этап 3)
"""

import os
from dotenv import load_dotenv
from src.diarizer import SpeakerDiarizer

# Загружаем переменные окружения
load_dotenv()

# Путь к файлу
audio_file = r"C:\Users\pokho\Downloads\Telegram Desktop\Запись_встречи_02_02_2026_13_06_35_запись.mp3"

print("="*80)
print("ТЕСТ ОПРЕДЕЛЕНИЯ СПИКЕРОВ (DIARIZATION)")
print("="*80)
print(f"Файл: {audio_file}")
print()

# Получаем токен
hf_token = os.getenv('HF_TOKEN')
if not hf_token or hf_token == 'your_huggingface_token_here':
    print("ОШИБКА: HF_TOKEN не найден в .env файле!")
    exit(1)

try:
    # Инициализация diarizer
    print("Инициализация pyannote.audio...")
    diarizer = SpeakerDiarizer(hf_token=hf_token, device='cpu')
    
    # Запуск diarization
    print("\nНачинаем определение спикеров...")
    diarization_segments = diarizer.diarize(
        audio_file,
        num_speakers=3
    )
    
    # Показываем результаты
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ")
    print("="*80)
    
    # Группируем по спикерам
    speakers = {}
    for seg in diarization_segments:
        if seg.speaker not in speakers:
            speakers[seg.speaker] = []
        speakers[seg.speaker].append(seg)
    
    for speaker in sorted(speakers.keys()):
        segs = speakers[speaker]
        total_time = sum(s.end - s.start for s in segs)
        print(f"\n{speaker}:")
        print(f"  Сегментов: {len(segs)}")
        print(f"  Общее время: {total_time:.1f} сек ({total_time/60:.1f} мин)")
        print(f"  Первый сегмент: {segs[0].start:.1f}с - {segs[0].end:.1f}с")
    
    print("\n✓ Тест завершен успешно!")
    
except Exception as e:
    print(f"\n❌ ОШИБКА: {str(e)}")
    import traceback
    traceback.print_exc()
