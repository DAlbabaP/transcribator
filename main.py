"""
Transcribator - Локальный транскрибатор для русского языка с определением спикеров

Использование:
    python main.py input.mp3 --output-dir ./output --formats text json srt
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Импорт модулей транскрибатора
from src.audio_processor import AudioProcessor
from src.transcriber import WhisperTranscriber
from src.diarizer import SpeakerDiarizer
from src.merger import TranscriptionMerger
from src.exporters.text_exporter import export_to_text
from src.exporters.json_exporter import export_to_json
from src.exporters.srt_exporter import export_to_srt, export_to_vtt


def main():
    """Главная функция CLI"""
    
    # Загружаем переменные окружения
    load_dotenv()
    
    # Парсинг аргументов
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Проверка входного файла
    if not os.path.exists(args.input_file):
        print(f"Ошибка: файл не найден: {args.input_file}")
        sys.exit(1)
    
    # Создаем директорию для результатов
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем имя файла без расширения
    input_filename = Path(args.input_file).stem
    
    print("=" * 80)
    print("TRANSCRIBATOR - Транскрибация с определением спикеров")
    print("=" * 80)
    print(f"Входной файл: {args.input_file}")
    print(f"Выходная директория: {args.output_dir}")
    print(f"Модель Whisper: {args.model}")
    print(f"Язык: {args.language}")
    print(f"Устройство: {args.device}")
    print(f"Форматы экспорта: {', '.join(args.formats)}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Шаг 1: Предобработка аудио
        print("\n" + "="*80)
        print("[1/5] ПРЕДОБРАБОТКА АУДИО")
        print("="*80)
        processor = AudioProcessor()
        processed_audio = processor.preprocess_audio(args.input_file)
        audio_duration = processor.get_audio_duration(processed_audio)
        print(f"✓ Предобработка завершена!")
        print(f"  └─ Длительность: {audio_duration:.2f} сек ({audio_duration/60:.1f} мин)")
        print(f"  └─ Формат: WAV 16kHz моно")
        
        # Шаг 2: Транскрибация
        print("\n" + "="*80)
        print(f"[2/5] ТРАНСКРИБАЦИЯ (модель: {args.model})")
        print("="*80)
        
        # Определяем оптимальное количество потоков
        cpu_count = os.cpu_count() or 4
        cpu_threads = args.cpu_threads if args.cpu_threads > 0 else cpu_count
        num_workers = args.num_workers if args.num_workers > 0 else min(4, cpu_count)
        
        print(f"⚙️  Оптимизация CPU:")
        print(f"  └─ Доступно ядер: {cpu_count}")
        print(f"  └─ Используется потоков: {cpu_threads}")
        print(f"  └─ Параллельных воркеров: {num_workers}")
        print()
        
        transcriber = WhisperTranscriber(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers
        )
        transcription_segments = transcriber.transcribe(
            processed_audio,
            language=args.language,
            vad_filter=args.vad_filter
        )
        print(f"\n✓ Транскрибация завершена!")
        print(f"  └─ Получено сегментов: {len(transcription_segments)}")
        print(f"  └─ Общее время текста: {sum(s.end - s.start for s in transcription_segments):.1f} сек")
        
        # Шаг 3: Определение спикеров (если включено)
        diarization_segments = None
        if not args.no_diarization:
            print("\n" + "="*80)
            print("[3/5] ОПРЕДЕЛЕНИЕ СПИКЕРОВ")
            print("="*80)
            
            # Получаем HuggingFace токен
            hf_token = args.hf_token or os.getenv('HF_TOKEN')
            if not hf_token or hf_token == 'your_huggingface_token_here':
                print("Ошибка: не указан HuggingFace токен!")
                print("Укажите токен через --hf-token или в .env файле (HF_TOKEN)")
                print("Как получить токен:")
                print("1. Зарегистрируйтесь на https://huggingface.co")
                print("2. Создайте токен: https://huggingface.co/settings/tokens")
                print("3. Примите лицензии моделей:")
                print("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
                print("   - https://huggingface.co/pyannote/segmentation-3.0")
                sys.exit(1)
            
            # Используем CPU для diarization если GPU не поддерживает архитектуру
            # (например, RTX 5060 Ti с sm_120 не поддерживается PyTorch 2.5.1)
            diarizer_device = args.device
            if args.device == 'cuda':
                import torch
                # Проверяем поддержку GPU для pyannote.audio
                if torch.cuda.is_available():
                    device_capability = torch.cuda.get_device_capability(0)
                    # sm_120 и новее не поддерживаются PyTorch 2.5.1
                    if device_capability[0] >= 12:
                        print(f"⚠️  GPU архитектура sm_{device_capability[0]}{device_capability[1]} не поддерживается PyTorch")
                        print(f"   Используем CPU для определения спикеров (транскрибация на GPU работает)")
                        diarizer_device = 'cpu'
            
            diarizer = SpeakerDiarizer(hf_token=hf_token, device=diarizer_device)
            diarization_segments = diarizer.diarize(
                processed_audio,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers
            )
        else:
            print("\n" + "="*80)
            print("[3/5] ОПРЕДЕЛЕНИЕ СПИКЕРОВ - ПРОПУЩЕНО")
            print("="*80)
            print("⚠️  Использован параметр --no-diarization")
        
        # Шаг 4: Объединение результатов
        print("\n" + "="*80)
        print("[4/5] ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*80)
        if diarization_segments:
            print("🔗 Объединяем транскрипцию со спикерами...")
            merger = TranscriptionMerger(min_overlap_ratio=args.min_overlap)
            merged_segments = merger.merge(transcription_segments, diarization_segments)
            stats = merger.get_statistics(merged_segments)
            print(f"\n✓ Объединение завершено!")
            print(f"  └─ Всего сегментов: {len(merged_segments)}")
            print(f"  └─ Спикеров: {stats['num_speakers']}")
            if stats['unknown_segments'] > 0:
                print(f"  └─ ⚠️  Неопределенных сегментов: {stats['unknown_segments']}")
        else:
            # Без diarization - просто используем транскрипцию
            from src.merger import MergedSegment
            merged_segments = [
                MergedSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker='SPEAKER_00',
                    confidence=1.0
                )
                for seg in transcription_segments
            ]
            print(f"✓ Сегментов транскрипции: {len(merged_segments)}")
        
        # Шаг 5: Экспорт результатов
        print("\n" + "="*80)
        print("[5/5] ЭКСПОРТ РЕЗУЛЬТАТОВ")
        print("="*80)
        
        # Метаданные для экспорта
        metadata = {
            'source_file': os.path.basename(args.input_file),
            'model': args.model,
            'language': args.language,
            'duration': audio_duration,
            'diarization_enabled': not args.no_diarization
        }
        
        exported_files = []
        
        # Экспорт в разные форматы
        if 'text' in args.formats or 'all' in args.formats:
            text_file = output_dir / f"{input_filename}.txt"
            export_to_text(
                merged_segments,
                str(text_file),
                show_confidence=args.show_confidence,
                group_by_speaker=True
            )
            exported_files.append(str(text_file))
        
        if 'json' in args.formats or 'all' in args.formats:
            json_file = output_dir / f"{input_filename}.json"
            export_to_json(
                merged_segments,
                str(json_file),
                metadata=metadata,
                pretty=True
            )
            exported_files.append(str(json_file))
        
        if 'srt' in args.formats or 'all' in args.formats:
            srt_file = output_dir / f"{input_filename}.srt"
            export_to_srt(
                merged_segments,
                str(srt_file),
                include_speakers=not args.no_diarization
            )
            exported_files.append(str(srt_file))
        
        if 'vtt' in args.formats or 'all' in args.formats:
            vtt_file = output_dir / f"{input_filename}.vtt"
            export_to_vtt(
                merged_segments,
                str(vtt_file),
                include_speakers=not args.no_diarization
            )
            exported_files.append(str(vtt_file))
        
        # Удаляем временный файл если он был создан
        if processed_audio != args.input_file:
            try:
                os.remove(processed_audio)
            except:
                pass
        
        # Итоги
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("✓ ТРАНСКРИБАЦИЯ ЗАВЕРШЕНА!")
        print("=" * 80)
        print(f"Время выполнения: {elapsed_time:.2f} секунд ({elapsed_time/60:.1f} минут)")
        print(f"Обработано аудио: {audio_duration:.2f} секунд")
        print(f"Скорость обработки: {audio_duration/elapsed_time:.2f}x реального времени")
        print(f"\nСозданные файлы:")
        for file_path in exported_files:
            print(f"  - {file_path}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nОперация прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nОшибка: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_argument_parser() -> argparse.ArgumentParser:
    """Создание парсера аргументов командной строки"""
    
    parser = argparse.ArgumentParser(
        description='Transcribator - Транскрибация аудио с определением спикеров',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Базовое использование
  python main.py audio.mp3

  # С указанием формата и модели
  python main.py audio.mp3 --model medium --formats json srt

  # Без определения спикеров
  python main.py audio.mp3 --no-diarization

  # С явным указанием количества спикеров
  python main.py audio.mp3 --num-speakers 3

  # Для английского языка
  python main.py audio.mp3 --language en

Больше информации: https://github.com/yourusername/transcribator
        """
    )
    
    # Обязательные аргументы
    parser.add_argument(
        'input_file',
        type=str,
        help='Путь к входному аудио или видео файлу'
    )
    
    # Опциональные аргументы
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Директория для результатов (по умолчанию: ./output)'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['text', 'json', 'srt', 'vtt', 'all'],
        default=['all'],
        help='Форматы экспорта (по умолчанию: all)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
        default=os.getenv('WHISPER_MODEL', 'small'),
        help='Размер модели Whisper (по умолчанию: small)'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default=os.getenv('DEFAULT_LANGUAGE', 'ru'),
        help='Язык транскрибации (по умолчанию: ru)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Устройство для вычислений (по умолчанию: cpu)'
    )
    
    parser.add_argument(
        '--compute-type',
        type=str,
        choices=['int8', 'float16', 'float32'],
        default='int8',
        help='Тип вычислений (int8 для CPU, float16 для GPU)'
    )
    
    parser.add_argument(
        '--cpu-threads',
        type=int,
        default=0,
        help='Количество потоков CPU (0 = все доступные)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Количество параллельных воркеров (0 = автоопределение)'
    )
    
    # Параметры diarization
    parser.add_argument(
        '--no-diarization',
        action='store_true',
        help='Отключить определение спикеров'
    )
    
    parser.add_argument(
        '--num-speakers',
        type=int,
        default=None,
        help='Точное количество спикеров (опционально)'
    )
    
    parser.add_argument(
        '--min-speakers',
        type=int,
        default=None,
        help='Минимальное количество спикеров (опционально)'
    )
    
    parser.add_argument(
        '--max-speakers',
        type=int,
        default=None,
        help='Максимальное количество спикеров (опционально)'
    )
    
    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace access token (или через HF_TOKEN в .env)'
    )
    
    # Дополнительные параметры
    parser.add_argument(
        '--min-overlap',
        type=float,
        default=0.5,
        help='Минимальное пересечение для назначения спикера (0.0-1.0)'
    )
    
    parser.add_argument(
        '--vad-filter',
        action='store_true',
        default=True,
        help='Использовать Voice Activity Detection'
    )
    
    parser.add_argument(
        '--show-confidence',
        action='store_true',
        help='Показывать уверенность в текстовом экспорте'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Transcribator 0.1.0'
    )
    
    return parser


if __name__ == '__main__':
    main()
