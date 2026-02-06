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
from src.whisperx_pipeline import run_whisperx_pipeline
from src.exporters.text_exporter import export_to_text
from src.exporters.json_exporter import export_to_json
from src.exporters.srt_exporter import export_to_srt, export_to_vtt
from src.command_store import save_command, list_commands


def main():
    """Главная функция CLI"""
    
    # Загружаем переменные окружения
    load_dotenv()
    
    # Проверка доступности CUDA перед запуском
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ Ошибка: CUDA недоступна!")
            print("Проект настроен только на работу с CUDA GPU.")
            print("\nУбедитесь, что:")
            print("1. Установлен PyTorch с поддержкой CUDA")
            print("2. Установлены драйверы NVIDIA")
            print("3. GPU поддерживает CUDA")
            print("\nСм. инструкции в INSTALL.md")
            sys.exit(1)
    except ImportError:
        print("❌ Ошибка: PyTorch не установлен!")
        print("Установите PyTorch с поддержкой CUDA.")
        print("См. инструкции в INSTALL.md")
        sys.exit(1)
    
    # Парсинг аргументов
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.list_commands:
        commands = list_commands()
        if not commands:
            print("Сохраненных команд нет.")
        else:
            print("Сохраненные команды:")
            for item in commands:
                desc = f" - {item['description']}" if item.get("description") else ""
                print(f"  • {item['name']}{desc}")
                print(f"    {item['command']}")
        sys.exit(0)

    if not args.input_file:
        print("Ошибка: не указан входной файл")
        parser.print_help()
        sys.exit(1)

    # Принудительно используем CUDA и float16
    if args.device != 'cuda':
        print("Ошибка: проект настроен только на работу с CUDA GPU")
        sys.exit(1)
    if args.compute_type == 'int8':
        args.compute_type = 'float16'
    
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
    print("TRANSCRIBATOR - WhisperX транскрибация с определением спикеров")
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
        print("[1/3] ПРЕДОБРАБОТКА АУДИО")
        print("="*80)
        processor = AudioProcessor()
        processed_audio = processor.preprocess_audio(args.input_file)
        audio_duration = processor.get_audio_duration(processed_audio)
        print(f"✓ Предобработка завершена!")
        print(f"  └─ Длительность: {audio_duration:.2f} сек ({audio_duration/60:.1f} мин)")
        print(f"  └─ Формат: WAV 16kHz моно")
        
        # Шаг 2: WhisperX (транскрибация + alignment + diarization)
        print("\n" + "="*80)
        print(f"[2/3] WHISPERX (модель: {args.model})")
        print("="*80)

        hf_token = args.hf_token or os.getenv('HF_TOKEN')

        merged_segments, wx_metadata = run_whisperx_pipeline(
            audio_file=processed_audio,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            language=args.language,
            align=args.align,
            diarize=not args.no_diarization,
            hf_token=hf_token,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers
        )

        print(f"\n✓ WhisperX завершен!")
        print(f"  └─ Получено сегментов: {len(merged_segments)}")
        
        # Шаг 3: Экспорт результатов
        print("\n" + "="*80)
        print("[3/3] ЭКСПОРТ РЕЗУЛЬТАТОВ")
        print("="*80)
        
        # Метаданные для экспорта
        metadata = {
            'source_file': os.path.basename(args.input_file),
            'model': wx_metadata.get('model', args.model),
            'language': wx_metadata.get('language', args.language),
            'duration': audio_duration,
            'diarization_enabled': not args.no_diarization,
            'aligned': wx_metadata.get('aligned', args.align),
            'batch_size': wx_metadata.get('batch_size', args.batch_size)
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
            except OSError as e:
                print(f"⚠️  Не удалось удалить временный файл: {processed_audio} ({e})")
        
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

        if args.save_command:
            command_str = " ".join(sys.argv)
            save_command(args.save_command, command_str)
            print(f"\n✓ Команда сохранена как: {args.save_command}")
        
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
    
    # Входной файл
    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',
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
        choices=['cuda'],
        default='cuda',
        help='Устройство для вычислений (только CUDA GPU)'
    )
    
    parser.add_argument(
        '--compute-type',
        type=str,
        choices=['int8', 'float16', 'float32'],
        default='int8',
        help='Тип вычислений (float16 для CUDA GPU, float32 для максимальной точности)'
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
        '--batch-size',
        type=int,
        default=16,
        help='Batch size для WhisperX (выше = быстрее, больше VRAM)'
    )

    parser.add_argument(
        '--align',
        dest='align',
        action='store_true',
        help='Включить выравнивание (word-level timestamps)'
    )
    parser.add_argument(
        '--no-align',
        dest='align',
        action='store_false',
        help='Выключить выравнивание (alignment)'
    )
    parser.set_defaults(align=True)
    
    parser.add_argument(
        '--show-confidence',
        action='store_true',
        help='Показывать уверенность в текстовом экспорте'
    )

    parser.add_argument(
        '--save-command',
        type=str,
        default=None,
        help='Сохранить команду запуска под указанным именем'
    )

    parser.add_argument(
        '--list-commands',
        action='store_true',
        help='Показать сохраненные команды и выйти'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Transcribator 0.1.0'
    )
    
    return parser


if __name__ == '__main__':
    main()
