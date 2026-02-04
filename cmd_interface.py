"""
Интерактивный CMD интерфейс для Transcribator

Использование:
    python cmd_interface.py
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Импорт модулей транскрибатора
from src.audio_processor import AudioProcessor
from src.transcriber import WhisperTranscriber
from src.diarizer import SpeakerDiarizer
from src.merger import TranscriptionMerger
from src.exporters.text_exporter import export_to_text
from src.exporters.json_exporter import export_to_json
from src.exporters.srt_exporter import export_to_srt, export_to_vtt


class TranscribatorInterface:
    """Интерактивный интерфейс для транскрибатора"""
    
    def __init__(self):
        """Инициализация интерфейса"""
        load_dotenv()
        
        # Глобальные настройки (не меняются от файла к файлу)
        self.global_settings = {
            'output_dir': './output',
            'device': 'cpu',
            'compute_type': 'int8',
            'hf_token': os.getenv('HF_TOKEN'),
            'min_overlap': 0.5,
            'vad_filter': True,
            'show_confidence': False,
            'cpu_threads': 0,
            'num_workers': 0
        }
        
        # Настройки по умолчанию для файлов (могут быть переопределены при запуске)
        self.default_file_settings = {
            'model': os.getenv('WHISPER_MODEL', 'small'),
            'language': os.getenv('DEFAULT_LANGUAGE', 'ru'),
            'formats': ['all'],
            'no_diarization': False,
            'num_speakers': None,
            'min_speakers': None,
            'max_speakers': None
        }
        
        # Текущий выбранный файл
        self.selected_file = None
        
        self.models = ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']
        self.languages = ['ru', 'en', 'uk', 'be', 'kk']
        self.devices = ['cpu', 'cuda']
        self.compute_types = ['int8', 'float16', 'float32']
        self.export_formats = ['text', 'json', 'srt', 'vtt', 'all']
    
    def clear_screen(self):
        """Очистка экрана"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title: str = "TRANSCRIBATOR"):
        """Печать заголовка"""
        print("=" * 80)
        print(f"  {title}")
        print("=" * 80)
        print()
    
    def print_menu(self, items: List[tuple], title: str = "МЕНЮ"):
        """Печать меню"""
        self.print_header(title)
        for i, (key, description) in enumerate(items, 1):
            print(f"  {i}. {description}")
        print()
        print("  0. Выход")
        print("=" * 80)
    
    def get_choice(self, max_choice: int) -> int:
        """Получение выбора пользователя"""
        while True:
            try:
                choice = input(f"\nВыберите опцию (0-{max_choice}): ").strip()
                choice = int(choice)
                if 0 <= choice <= max_choice:
                    return choice
                else:
                    print(f"❌ Неверный выбор. Введите число от 0 до {max_choice}")
            except ValueError:
                print("❌ Введите число")
            except KeyboardInterrupt:
                print("\n\nОперация отменена")
                return 0
    
    def get_input(self, prompt: str, default: Optional[str] = None, validator=None) -> str:
        """Получение ввода от пользователя"""
        if default:
            prompt_text = f"{prompt} [{default}]: "
        else:
            prompt_text = f"{prompt}: "
        
        while True:
            try:
                value = input(prompt_text).strip()
                if not value and default:
                    return default
                if not value:
                    print("❌ Значение не может быть пустым")
                    continue
                if validator:
                    if not validator(value):
                        continue
                return value
            except KeyboardInterrupt:
                print("\n\nОперация отменена")
                return ""
    
    def select_file(self) -> Optional[str]:
        """Выбор файла для обработки"""
        self.clear_screen()
        self.print_header("ВЫБОР ФАЙЛА")
        
        print("Введите путь к аудио или видео файлу:")
        print("(или нажмите Enter для просмотра файлов в папке input/)")
        print()
        
        file_path = input("Путь к файлу: ").strip()
        
        # Если пусто, показываем файлы в input/
        if not file_path:
            input_dir = Path('./input')
            if input_dir.exists():
                files = list(input_dir.glob('*'))
                audio_video_files = [
                    f for f in files 
                    if f.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac', '.mp4', '.avi', '.mkv', '.mov']
                ]
                
                if audio_video_files:
                    print("\nНайденные файлы в папке input/:")
                    for i, file in enumerate(audio_video_files, 1):
                        size_mb = file.stat().st_size / (1024 * 1024)
                        print(f"  {i}. {file.name} ({size_mb:.1f} MB)")
                    
                    choice = self.get_choice(len(audio_video_files))
                    if choice > 0:
                        file_path = str(audio_video_files[choice - 1])
                else:
                    print("❌ Файлы не найдены в папке input/")
                    input("\nНажмите Enter для продолжения...")
                    return None
            else:
                print("❌ Папка input/ не существует")
                input("\nНажмите Enter для продолжения...")
                return None
        
        # Проверка существования файла
        if not os.path.exists(file_path):
            print(f"❌ Файл не найден: {file_path}")
            input("\nНажмите Enter для продолжения...")
            return None
        
        self.selected_file = file_path
        print(f"✓ Файл выбран: {file_path}")
        input("\nНажмите Enter для продолжения...")
        return file_path
    
    def configure_settings(self):
        """Настройка глобальных параметров"""
        while True:
            self.clear_screen()
            self.print_header("ГЛОБАЛЬНЫЕ НАСТРОЙКИ")
            
            print("Текущие глобальные настройки:")
            print(f"  Выходная папка: {self.global_settings['output_dir']}")
            print(f"  Устройство: {self.global_settings['device']}")
            print(f"  Тип вычислений: {self.global_settings['compute_type']}")
            print(f"  HuggingFace токен: {'установлен' if self.global_settings['hf_token'] else 'не установлен'}")
            print(f"  Минимальное пересечение: {self.global_settings['min_overlap']}")
            print(f"  VAD фильтр: {'включен' if self.global_settings['vad_filter'] else 'выключен'}")
            print(f"  Показывать уверенность: {'да' if self.global_settings['show_confidence'] else 'нет'}")
            print()
            print("Настройки по умолчанию для файлов:")
            print(f"  Модель: {self.default_file_settings['model']}")
            print(f"  Язык: {self.default_file_settings['language']}")
            print(f"  Форматы: {', '.join(self.default_file_settings['formats'])}")
            print(f"  Определение спикеров: {'выключено' if self.default_file_settings['no_diarization'] else 'включено'}")
            print()
            
            menu_items = [
                ('1', 'Изменить выходную папку'),
                ('2', 'Изменить устройство (CPU/CUDA)'),
                ('3', 'Изменить тип вычислений'),
                ('4', 'Настроить HuggingFace токен'),
                ('5', 'Изменить минимальное пересечение'),
                ('6', 'Включить/выключить VAD фильтр'),
                ('7', 'Показывать уверенность'),
                ('8', 'Изменить настройки по умолчанию для файлов'),
            ]
            
            self.print_menu(menu_items, "ГЛОБАЛЬНЫЕ НАСТРОЙКИ")
            choice = self.get_choice(len(menu_items))
            
            if choice == 0:
                break
            elif choice == 1:
                self._configure_output_dir()
            elif choice == 2:
                self._configure_device()
            elif choice == 3:
                self._configure_compute_type()
            elif choice == 4:
                self._configure_hf_token()
            elif choice == 5:
                self._configure_min_overlap()
            elif choice == 6:
                self._toggle_vad_filter()
            elif choice == 7:
                self._toggle_show_confidence()
            elif choice == 8:
                self._configure_default_file_settings()
    
    def _configure_output_dir(self):
        """Настройка выходной папки"""
        self.clear_screen()
        self.print_header("ВЫХОДНАЯ ПАПКА")
        new_dir = self.get_input("Введите путь к папке для результатов", self.global_settings['output_dir'])
        if new_dir:
            self.global_settings['output_dir'] = new_dir
            Path(new_dir).mkdir(parents=True, exist_ok=True)
            print(f"✓ Выходная папка установлена: {new_dir}")
            input("\nНажмите Enter для продолжения...")
    
    def _configure_device(self):
        """Настройка устройства"""
        self.clear_screen()
        self.print_header("УСТРОЙСТВО")
        print("Доступные устройства:")
        for i, device in enumerate(self.devices, 1):
            marker = " ← текущее" if device == self.global_settings['device'] else ""
            name = "CPU" if device == 'cpu' else "CUDA (GPU)"
            print(f"  {i}. {name} ({device}){marker}")
        print()
        
        choice = self.get_choice(len(self.devices))
        if choice > 0:
            self.global_settings['device'] = self.devices[choice - 1]
            # Автоматически меняем compute_type для GPU
            if self.global_settings['device'] == 'cuda' and self.global_settings['compute_type'] == 'int8':
                self.global_settings['compute_type'] = 'float16'
            print(f"✓ Устройство установлено: {self.global_settings['device']}")
            input("\nНажмите Enter для продолжения...")
    
    def _configure_compute_type(self):
        """Настройка типа вычислений"""
        self.clear_screen()
        self.print_header("ТИП ВЫЧИСЛЕНИЙ")
        print("Доступные типы:")
        for i, ct in enumerate(self.compute_types, 1):
            marker = " ← текущий" if ct == self.global_settings['compute_type'] else ""
            desc = {
                'int8': 'Оптимально для CPU',
                'float16': 'Оптимально для GPU',
                'float32': 'Максимальная точность (медленнее)'
            }.get(ct, '')
            print(f"  {i}. {ct} - {desc}{marker}")
        print()
        
        choice = self.get_choice(len(self.compute_types))
        if choice > 0:
            self.global_settings['compute_type'] = self.compute_types[choice - 1]
            print(f"✓ Тип вычислений установлен: {self.global_settings['compute_type']}")
            input("\nНажмите Enter для продолжения...")
    
    def _configure_hf_token(self):
        """Настройка HuggingFace токена"""
        self.clear_screen()
        self.print_header("HUGGINGFACE ТОКЕН")
        
        current = self.global_settings['hf_token']
        if current:
            masked = current[:8] + "..." + current[-4:] if len(current) > 12 else "***"
            print(f"Текущий токен: {masked}")
        else:
            print("Токен не установлен")
        print()
        print("Введите новый токен (или нажмите Enter чтобы оставить текущий):")
        
        new_token = input("Токен: ").strip()
        if new_token:
            self.global_settings['hf_token'] = new_token
            print("✓ Токен обновлен")
        else:
            print("Токен не изменен")
        
        input("\nНажмите Enter для продолжения...")
    
    def _configure_min_overlap(self):
        """Настройка минимального пересечения"""
        self.clear_screen()
        self.print_header("МИНИМАЛЬНОЕ ПЕРЕСЕЧЕНИЕ")
        print(f"Текущее значение: {self.global_settings['min_overlap']}")
        print("Рекомендуемое значение: 0.5 (50% пересечения)")
        print()
        
        new_value = input(f"Введите новое значение (0.0-1.0) [{self.global_settings['min_overlap']}]: ").strip()
        if new_value:
            try:
                value = float(new_value)
                if 0.0 <= value <= 1.0:
                    self.global_settings['min_overlap'] = value
                    print(f"✓ Минимальное пересечение установлено: {value}")
                else:
                    print("❌ Значение должно быть от 0.0 до 1.0")
            except:
                print("❌ Неверное значение")
        else:
            print("Значение не изменено")
        
        input("\nНажмите Enter для продолжения...")
    
    def _toggle_vad_filter(self):
        """Переключение VAD фильтра"""
        self.clear_screen()
        self.print_header("VAD ФИЛЬТР")
        current = "включен" if self.global_settings['vad_filter'] else "выключен"
        print(f"Текущее состояние: {current}")
        print()
        print("1. Включить VAD фильтр")
        print("2. Выключить VAD фильтр")
        print()
        
        choice = self.get_choice(2)
        if choice == 1:
            self.global_settings['vad_filter'] = True
            print("✓ VAD фильтр включен")
        elif choice == 2:
            self.global_settings['vad_filter'] = False
            print("✓ VAD фильтр выключен")
        
        input("\nНажмите Enter для продолжения...")
    
    def _toggle_show_confidence(self):
        """Переключение показа уверенности"""
        self.clear_screen()
        self.print_header("ПОКАЗ УВЕРЕННОСТИ")
        current = "включен" if self.global_settings['show_confidence'] else "выключен"
        print(f"Текущее состояние: {current}")
        print()
        print("1. Включить показ уверенности")
        print("2. Выключить показ уверенности")
        print()
        
        choice = self.get_choice(2)
        if choice == 1:
            self.global_settings['show_confidence'] = True
            print("✓ Показ уверенности включен")
        elif choice == 2:
            self.global_settings['show_confidence'] = False
            print("✓ Показ уверенности выключен")
        
        input("\nНажмите Enter для продолжения...")
    
    def _configure_default_file_settings(self):
        """Настройка значений по умолчанию для файлов"""
        while True:
            self.clear_screen()
            self.print_header("НАСТРОЙКИ ПО УМОЛЧАНИЮ ДЛЯ ФАЙЛОВ")
            
            print("Текущие значения по умолчанию:")
            print(f"  Модель: {self.default_file_settings['model']}")
            print(f"  Язык: {self.default_file_settings['language']}")
            print(f"  Форматы: {', '.join(self.default_file_settings['formats'])}")
            print(f"  Определение спикеров: {'выключено' if self.default_file_settings['no_diarization'] else 'включено'}")
            print()
            
            menu_items = [
                ('1', 'Изменить модель по умолчанию'),
                ('2', 'Изменить язык по умолчанию'),
                ('3', 'Изменить форматы по умолчанию'),
                ('4', 'Включить/выключить определение спикеров по умолчанию'),
            ]
            
            self.print_menu(menu_items, "НАСТРОЙКИ ПО УМОЛЧАНИЮ")
            choice = self.get_choice(len(menu_items))
            
            if choice == 0:
                break
            elif choice == 1:
                self._configure_default_model()
            elif choice == 2:
                self._configure_default_language()
            elif choice == 3:
                self._configure_default_formats()
            elif choice == 4:
                self._toggle_default_diarization()
    
    def _configure_default_model(self):
        """Настройка модели по умолчанию"""
        self.clear_screen()
        self.print_header("МОДЕЛЬ ПО УМОЛЧАНИЮ")
        print("Доступные модели:")
        for i, model in enumerate(self.models, 1):
            marker = " ← текущая" if model == self.default_file_settings['model'] else ""
            print(f"  {i}. {model}{marker}")
        print()
        
        choice = self.get_choice(len(self.models))
        if choice > 0:
            self.default_file_settings['model'] = self.models[choice - 1]
            print(f"✓ Модель по умолчанию установлена: {self.default_file_settings['model']}")
            input("\nНажмите Enter для продолжения...")
    
    def _configure_default_language(self):
        """Настройка языка по умолчанию"""
        self.clear_screen()
        self.print_header("ЯЗЫК ПО УМОЛЧАНИЮ")
        print("Доступные языки:")
        lang_names = {'ru': 'Русский', 'en': 'Английский', 'uk': 'Украинский', 
                     'be': 'Белорусский', 'kk': 'Казахский'}
        for i, lang in enumerate(self.languages, 1):
            marker = " ← текущий" if lang == self.default_file_settings['language'] else ""
            name = lang_names.get(lang, lang)
            print(f"  {i}. {lang} ({name}){marker}")
        print()
        print("Или введите код языка вручную (например: de, fr, es)")
        
        choice = self.get_choice(len(self.languages))
        if choice > 0:
            self.default_file_settings['language'] = self.languages[choice - 1]
        else:
            custom_lang = input("Введите код языка: ").strip().lower()
            if custom_lang:
                self.default_file_settings['language'] = custom_lang
        
        print(f"✓ Язык по умолчанию установлен: {self.default_file_settings['language']}")
        input("\nНажмите Enter для продолжения...")
    
    def _configure_default_formats(self):
        """Настройка форматов по умолчанию"""
        self.clear_screen()
        self.print_header("ФОРМАТЫ ПО УМОЛЧАНИЮ")
        print("Текущие форматы:", ', '.join(self.default_file_settings['formats']))
        print()
        print("Доступные форматы:")
        format_names = {
            'text': 'Текст (.txt)',
            'json': 'JSON (.json)',
            'srt': 'SRT субтитры (.srt)',
            'vtt': 'WebVTT субтитры (.vtt)',
            'all': 'Все форматы'
        }
        for i, fmt in enumerate(self.export_formats, 1):
            selected = "✓" if fmt in self.default_file_settings['formats'] or 'all' in self.default_file_settings['formats'] else " "
            print(f"  {i}. [{selected}] {format_names.get(fmt, fmt)}")
        print()
        print("Введите номера форматов через запятую (например: 1,2,3) или 5 для всех:")
        
        try:
            choices = input("Выбор: ").strip()
            if choices == '5':
                self.default_file_settings['formats'] = ['all']
            else:
                selected = [int(x.strip()) for x in choices.split(',')]
                self.default_file_settings['formats'] = [self.export_formats[i-1] for i in selected if 1 <= i <= len(self.export_formats)]
            
            if self.default_file_settings['formats']:
                print(f"✓ Форматы по умолчанию установлены: {', '.join(self.default_file_settings['formats'])}")
            else:
                print("❌ Не выбрано ни одного формата, установлен 'all'")
                self.default_file_settings['formats'] = ['all']
        except:
            print("❌ Ошибка ввода, форматы не изменены")
        
        input("\nНажмите Enter для продолжения...")
    
    def _toggle_default_diarization(self):
        """Переключение определения спикеров по умолчанию"""
        self.clear_screen()
        self.print_header("ОПРЕДЕЛЕНИЕ СПИКЕРОВ ПО УМОЛЧАНИЮ")
        current = "включено" if not self.default_file_settings['no_diarization'] else "выключено"
        print(f"Текущее состояние: {current}")
        print()
        print("1. Включить определение спикеров по умолчанию")
        print("2. Выключить определение спикеров по умолчанию")
        print()
        
        choice = self.get_choice(2)
        if choice == 1:
            self.default_file_settings['no_diarization'] = False
            print("✓ Определение спикеров по умолчанию включено")
        elif choice == 2:
            self.default_file_settings['no_diarization'] = True
            print("✓ Определение спикеров по умолчанию выключено")
        
        input("\nНажмите Enter для продолжения...")
    
    def show_summary(self):
        """Показать сводку настроек"""
        self.clear_screen()
        self.print_header("СВОДКА НАСТРОЕК")
        
        print("Глобальные настройки:")
        print(f"  Выходная папка: {self.global_settings['output_dir']}")
        print(f"  Устройство: {self.global_settings['device']}")
        print(f"  Тип вычислений: {self.global_settings['compute_type']}")
        print(f"  HuggingFace токен: {'установлен' if self.global_settings['hf_token'] else 'не установлен'}")
        print()
        print("Настройки по умолчанию для файлов:")
        print(f"  Модель: {self.default_file_settings['model']}")
        print(f"  Язык: {self.default_file_settings['language']}")
        print(f"  Форматы: {', '.join(self.default_file_settings['formats'])}")
        print(f"  Определение спикеров: {'выключено' if self.default_file_settings['no_diarization'] else 'включено'}")
        print()
        print(f"Выбранный файл: {self.selected_file or '❌ не выбран'}")
        print()
        
        if not self.global_settings['hf_token']:
            print("⚠️  ВНИМАНИЕ: HuggingFace токен не установлен!")
            print("   Определение спикеров может не работать.")
            print()
        
        input("Нажмите Enter для продолжения...")
    
    def request_file_settings(self) -> Optional[Dict]:
        """Запрос параметров для конкретного файла"""
        self.clear_screen()
        self.print_header("НАСТРОЙКА ПАРАМЕТРОВ ДЛЯ ФАЙЛА")
        
        print(f"Файл: {self.selected_file}")
        print()
        print("Настройте параметры для этого файла (нажмите Enter для значений по умолчанию):")
        print()
        
        # Инициализируем настройки значениями по умолчанию
        file_settings = {
            'model': self.default_file_settings['model'],
            'language': self.default_file_settings['language'],
            'formats': self.default_file_settings['formats'].copy(),
            'no_diarization': self.default_file_settings['no_diarization'],
            'num_speakers': None,
            'min_speakers': None,
            'max_speakers': None
        }
        
        # Модель
        print("1. Модель Whisper")
        print("Доступные модели:")
        for i, model in enumerate(self.models, 1):
            marker = " ← по умолчанию" if model == file_settings['model'] else ""
            print(f"   {i}. {model}{marker}")
        print("   (нажмите Enter для значения по умолчанию)")
        choice = input("Выбор: ").strip()
        if choice:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(self.models):
                    file_settings['model'] = self.models[idx]
            except:
                pass
        
        # Язык
        print()
        print("2. Язык транскрибации")
        lang_names = {'ru': 'Русский', 'en': 'Английский', 'uk': 'Украинский', 
                     'be': 'Белорусский', 'kk': 'Казахский'}
        print("Доступные языки:")
        for i, lang in enumerate(self.languages, 1):
            marker = " ← по умолчанию" if lang == file_settings['language'] else ""
            name = lang_names.get(lang, lang)
            print(f"   {i}. {lang} ({name}){marker}")
        print("   (нажмите Enter для значения по умолчанию, или введите код языка)")
        choice = input("Выбор: ").strip()
        if choice:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(self.languages):
                    file_settings['language'] = self.languages[idx]
            except:
                # Возможно, это код языка
                if len(choice) == 2:
                    file_settings['language'] = choice.lower()
        print()
        
        # Форматы экспорта
        print("3. Форматы экспорта")
        format_names = {
            'text': 'Текст (.txt)',
            'json': 'JSON (.json)',
            'srt': 'SRT субтитры (.srt)',
            'vtt': 'WebVTT субтитры (.vtt)',
            'all': 'Все форматы'
        }
        print("Доступные форматы:")
        for i, fmt in enumerate(self.export_formats, 1):
            selected = "✓" if fmt in file_settings['formats'] or 'all' in file_settings['formats'] else " "
            print(f"   {i}. [{selected}] {format_names.get(fmt, fmt)}")
        print("   (нажмите Enter для значения по умолчанию)")
        print("   Введите номера через запятую (например: 1,2,3) или 5 для всех:")
        choice = input("Выбор: ").strip()
        if choice:
            try:
                if choice == '5':
                    file_settings['formats'] = ['all']
                else:
                    selected = [int(x.strip()) for x in choice.split(',')]
                    file_settings['formats'] = [self.export_formats[i-1] for i in selected if 1 <= i <= len(self.export_formats)]
                    if not file_settings['formats']:
                        file_settings['formats'] = ['all']
            except:
                pass
        print()
        
        # Определение спикеров
        print("4. Определение спикеров")
        current = "включено" if not file_settings['no_diarization'] else "выключено"
        print(f"   Текущее (по умолчанию): {current}")
        print("   1. Включить")
        print("   2. Выключить")
        print("   (нажмите Enter для значения по умолчанию)")
        choice = input("Выбор: ").strip()
        if choice == '1':
            file_settings['no_diarization'] = False
        elif choice == '2':
            file_settings['no_diarization'] = True
        
        # Параметры спикеров (если включено)
        if not file_settings['no_diarization']:
            print()
            print("5. Параметры спикеров (опционально)")
            print("   (нажмите Enter чтобы пропустить)")
            print("   a) Точное количество спикеров")
            print("   b) Диапазон (мин/макс)")
            choice = input("Выбор (a/b или Enter): ").strip().lower()
            if choice == 'a':
                num = input("Количество спикеров: ").strip()
                try:
                    file_settings['num_speakers'] = int(num)
                    file_settings['min_speakers'] = None
                    file_settings['max_speakers'] = None
                except:
                    pass
            elif choice == 'b':
                min_sp = input("Минимум: ").strip()
                max_sp = input("Максимум: ").strip()
                try:
                    file_settings['min_speakers'] = int(min_sp) if min_sp else None
                    file_settings['max_speakers'] = int(max_sp) if max_sp else None
                    file_settings['num_speakers'] = None
                except:
                    pass
        
        # Показываем итоговые настройки
        print()
        print("=" * 80)
        print("ИТОГОВЫЕ НАСТРОЙКИ ДЛЯ ЭТОГО ФАЙЛА:")
        print("=" * 80)
        print(f"  Модель: {file_settings['model']}")
        print(f"  Язык: {file_settings['language']}")
        print(f"  Форматы: {', '.join(file_settings['formats'])}")
        print(f"  Определение спикеров: {'выключено' if file_settings['no_diarization'] else 'включено'}")
        if not file_settings['no_diarization']:
            if file_settings['num_speakers']:
                print(f"  Количество спикеров: {file_settings['num_speakers']}")
            if file_settings['min_speakers'] or file_settings['max_speakers']:
                print(f"  Диапазон: {file_settings['min_speakers'] or '?'} - {file_settings['max_speakers'] or '?'}")
        print("=" * 80)
        print()
        
        confirm = input("Начать транскрибацию с этими настройками? (y/n) [y]: ").strip().lower()
        if confirm and confirm != 'y':
            return None
        
        return file_settings
    
    def run_transcription(self):
        """Запуск транскрибации"""
        if not self.selected_file:
            self.clear_screen()
            self.print_header("ОШИБКА")
            print("❌ Файл не выбран!")
            print("Сначала выберите файл для обработки.")
            input("\nНажмите Enter для продолжения...")
            return
        
        if not os.path.exists(self.selected_file):
            self.clear_screen()
            self.print_header("ОШИБКА")
            print(f"❌ Файл не найден: {self.selected_file}")
            input("\nНажмите Enter для продолжения...")
            return
        
        # Запрашиваем параметры для этого файла
        file_settings = self.request_file_settings()
        if not file_settings:
            return  # Пользователь отменил
        
        # Объединяем глобальные и файловые настройки
        settings = {**self.global_settings, **file_settings, 'input_file': self.selected_file}
        
        # Создаем директорию для результатов
        output_dir = Path(settings['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Получаем имя файла без расширения
        input_filename = Path(settings['input_file']).stem
        
        self.clear_screen()
        self.print_header("ТРАНСКРИБАЦИЯ")
        print(f"Файл: {settings['input_file']}")
        print(f"Модель: {settings['model']}")
        print(f"Язык: {settings['language']}")
        print()
        print("Начинаем обработку...")
        print("=" * 80)
        print()
        
        start_time = time.time()
        
        try:
            # Шаг 1: Предобработка аудио
            print("[1/5] ПРЕДОБРАБОТКА АУДИО")
            print("-" * 80)
            processor = AudioProcessor()
            processed_audio = processor.preprocess_audio(settings['input_file'])
            audio_duration = processor.get_audio_duration(processed_audio)
            print(f"✓ Предобработка завершена! Длительность: {audio_duration:.2f} сек ({audio_duration/60:.1f} мин)")
            print()
            
            # Шаг 2: Транскрибация
            print("[2/5] ТРАНСКРИБАЦИЯ")
            print("-" * 80)
            
            cpu_count = os.cpu_count() or 4
            cpu_threads = settings['cpu_threads'] if settings['cpu_threads'] > 0 else cpu_count
            num_workers = settings['num_workers'] if settings['num_workers'] > 0 else min(4, cpu_count)
            
            transcriber = WhisperTranscriber(
                model_size=settings['model'],
                device=settings['device'],
                compute_type=settings['compute_type'],
                cpu_threads=cpu_threads,
                num_workers=num_workers
            )
            transcription_segments = transcriber.transcribe(
                processed_audio,
                language=settings['language'],
                vad_filter=settings['vad_filter']
            )
            print(f"✓ Транскрибация завершена! Сегментов: {len(transcription_segments)}")
            print()
            
            # Шаг 3: Определение спикеров
            diarization_segments = None
            if not settings['no_diarization']:
                print("[3/5] ОПРЕДЕЛЕНИЕ СПИКЕРОВ")
                print("-" * 80)
                
                hf_token = settings['hf_token']
                if not hf_token or hf_token == 'your_huggingface_token_here':
                    print("❌ Ошибка: не указан HuggingFace токен!")
                    print("Укажите токен в настройках.")
                    input("\nНажмите Enter для продолжения...")
                    return
                
                diarizer_device = settings['device']
                if settings['device'] == 'cuda':
                    import torch
                    if torch.cuda.is_available():
                        device_capability = torch.cuda.get_device_capability(0)
                        if device_capability[0] >= 12:
                            print(f"⚠️  GPU архитектура не поддерживается, используем CPU")
                            diarizer_device = 'cpu'
                
                diarizer = SpeakerDiarizer(hf_token=hf_token, device=diarizer_device)
                diarization_segments = diarizer.diarize(
                    processed_audio,
                    num_speakers=settings['num_speakers'],
                    min_speakers=settings['min_speakers'],
                    max_speakers=settings['max_speakers']
                )
                print(f"✓ Определение спикеров завершено!")
                print()
            else:
                print("[3/5] ОПРЕДЕЛЕНИЕ СПИКЕРОВ - ПРОПУЩЕНО")
                print("-" * 80)
                print()
            
            # Шаг 4: Объединение результатов
            print("[4/5] ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ")
            print("-" * 80)
            if diarization_segments:
                merger = TranscriptionMerger(min_overlap_ratio=settings['min_overlap'])
                merged_segments = merger.merge(transcription_segments, diarization_segments)
                stats = merger.get_statistics(merged_segments)
                print(f"✓ Объединение завершено! Сегментов: {len(merged_segments)}, Спикеров: {stats['num_speakers']}")
            else:
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
            print()
            
            # Шаг 5: Экспорт результатов
            print("[5/5] ЭКСПОРТ РЕЗУЛЬТАТОВ")
            print("-" * 80)
            
            metadata = {
                'source_file': os.path.basename(settings['input_file']),
                'model': settings['model'],
                'language': settings['language'],
                'duration': audio_duration,
                'diarization_enabled': not settings['no_diarization']
            }
            
            exported_files = []
            
            if 'text' in settings['formats'] or 'all' in settings['formats']:
                text_file = output_dir / f"{input_filename}.txt"
                export_to_text(
                    merged_segments,
                    str(text_file),
                    show_confidence=settings['show_confidence'],
                    group_by_speaker=True
                )
                exported_files.append(str(text_file))
                print(f"✓ Создан: {text_file}")
            
            if 'json' in settings['formats'] or 'all' in settings['formats']:
                json_file = output_dir / f"{input_filename}.json"
                export_to_json(
                    merged_segments,
                    str(json_file),
                    metadata=metadata,
                    pretty=True
                )
                exported_files.append(str(json_file))
                print(f"✓ Создан: {json_file}")
            
            if 'srt' in settings['formats'] or 'all' in settings['formats']:
                srt_file = output_dir / f"{input_filename}.srt"
                export_to_srt(
                    merged_segments,
                    str(srt_file),
                    include_speakers=not settings['no_diarization']
                )
                exported_files.append(str(srt_file))
                print(f"✓ Создан: {srt_file}")
            
            if 'vtt' in settings['formats'] or 'all' in settings['formats']:
                vtt_file = output_dir / f"{input_filename}.vtt"
                export_to_vtt(
                    merged_segments,
                    str(vtt_file),
                    include_speakers=not settings['no_diarization']
                )
                exported_files.append(str(vtt_file))
                print(f"✓ Создан: {vtt_file}")
            
            # Удаляем временный файл
            if processed_audio != settings['input_file']:
                try:
                    os.remove(processed_audio)
                except:
                    pass
            
            # Итоги
            elapsed_time = time.time() - start_time
            print()
            print("=" * 80)
            print("✓ ТРАНСКРИБАЦИЯ ЗАВЕРШЕНА!")
            print("=" * 80)
            print(f"Время выполнения: {elapsed_time:.2f} секунд ({elapsed_time/60:.1f} минут)")
            print(f"Обработано аудио: {audio_duration:.2f} секунд")
            if elapsed_time > 0:
                print(f"Скорость обработки: {audio_duration/elapsed_time:.2f}x реального времени")
            print(f"\nСозданные файлы:")
            for file_path in exported_files:
                print(f"  - {file_path}")
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\n\n❌ Операция прервана пользователем")
        except Exception as e:
            print(f"\n\n❌ Ошибка: {str(e)}")
            import traceback
            traceback.print_exc()
        
        input("\nНажмите Enter для продолжения...")
    
    def main_menu(self):
        """Главное меню"""
        while True:
            self.clear_screen()
            self.print_header()
            
            menu_items = [
                ('1', 'Выбрать файл для обработки'),
                ('2', 'Настроить параметры'),
                ('3', 'Показать сводку настроек'),
                ('4', 'Запустить транскрибацию'),
            ]
            
            self.print_menu(menu_items)
            choice = self.get_choice(len(menu_items))
            
            if choice == 0:
                self.clear_screen()
                print("До свидания!")
                break
            elif choice == 1:
                self.select_file()
            elif choice == 2:
                self.configure_settings()
            elif choice == 3:
                self.show_summary()
            elif choice == 4:
                self.run_transcription()


def main():
    """Главная функция"""
    try:
        interface = TranscribatorInterface()
        interface.main_menu()
    except KeyboardInterrupt:
        print("\n\nПрограмма завершена")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nКритическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

