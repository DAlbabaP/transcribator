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
from src.whisperx_pipeline import run_whisperx_pipeline
from src.exporters.text_exporter import export_to_text
from src.exporters.json_exporter import export_to_json
from src.exporters.srt_exporter import export_to_srt, export_to_vtt
from src.command_store import save_command, list_commands, load_settings, save_settings


class TranscribatorInterface:
    """Интерактивный интерфейс для транскрибатора"""
    
    def __init__(self):
        """Инициализация интерфейса"""
        load_dotenv()
        
        default_device = self._detect_default_device()
        default_compute = "float16"  # Только для CUDA GPU

        # Глобальные настройки (не меняются от файла к файлу)
        self.global_settings = {
            'output_dir': './output',
            'device': default_device,
            'compute_type': default_compute,
            'hf_token': os.getenv('HF_TOKEN'),
            'batch_size': 16,
            'align': True,
            'show_confidence': False
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
        self.last_input_file = None
        
        self.models = ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']
        self.languages = ['ru', 'en', 'uk', 'be', 'kk']
        self.devices = ['cuda']  # Только CUDA GPU
        self.compute_types = ['float16', 'float32']  # Только для CUDA GPU
        self.export_formats = ['text', 'json', 'srt', 'vtt', 'all']

        self._load_saved_settings()
        self._coerce_device_compute()

    def _detect_default_device(self) -> str:
        """Проверка доступности CUDA. Проект работает только на GPU."""
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA недоступна. Проект настроен только на работу с CUDA GPU.\n"
                    "Убедитесь, что:\n"
                    "1. Установлен PyTorch с поддержкой CUDA\n"
                    "2. Установлены драйверы NVIDIA\n"
                    "3. GPU поддерживает CUDA"
                )
            return "cuda"
        except ImportError:
            raise ImportError(
                "PyTorch не установлен. Установите PyTorch с поддержкой CUDA.\n"
                "См. инструкции в INSTALL.md"
            )

    def _coerce_device_compute(self):
        """Принудительно используем CUDA и float16"""
        if self.global_settings['device'] != 'cuda':
            raise ValueError(f"Проект настроен только на CUDA GPU. Получено устройство: {self.global_settings['device']}")
        if self.global_settings['compute_type'] == 'int8':
            self.global_settings['compute_type'] = 'float16'

    def _load_saved_settings(self):
        saved = load_settings()
        if not isinstance(saved, dict):
            return
        global_saved = saved.get("global")
        if isinstance(global_saved, dict):
            self.global_settings.update(global_saved)
        defaults_saved = saved.get("defaults")
        if isinstance(defaults_saved, dict):
            self.default_file_settings.update(defaults_saved)
        last_input = saved.get("last_input_file")
        if isinstance(last_input, str) and last_input:
            self.last_input_file = last_input

        # Принудительно используем CUDA
        if self.global_settings.get('device') not in self.devices:
            self.global_settings['device'] = self._detect_default_device()
        if self.global_settings.get('device') != 'cuda':
            raise ValueError(f"Проект настроен только на CUDA GPU. Получено устройство: {self.global_settings.get('device')}")
        if self.global_settings.get('compute_type') not in self.compute_types:
            self.global_settings['compute_type'] = "float16"  # Только для CUDA GPU
        if self.default_file_settings.get('model') not in self.models:
            self.default_file_settings['model'] = os.getenv('WHISPER_MODEL', 'small')
        if not isinstance(self.default_file_settings.get('formats'), list):
            self.default_file_settings['formats'] = ['all']

    def _save_settings(self):
        payload = {
            "global": self.global_settings,
            "defaults": self.default_file_settings,
            "last_input_file": self.last_input_file,
        }
        save_settings(payload)
    
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

    def _parse_yes_no(self, value: str, default: bool) -> bool:
        if not value:
            return default
        return value.strip().lower() in {"y", "yes", "да", "д"}

    def _parse_formats(self, raw: str) -> List[str]:
        raw = raw.strip().lower()
        if not raw:
            return self.default_file_settings['formats'].copy()
        if raw == "all":
            return ["all"]
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        selected = [p for p in parts if p in self.export_formats and p != "all"]
        return selected if selected else ["all"]

    def run_linear(self):
        """Линейный запуск без меню"""
        self.clear_screen()
        self.print_header("TRANSCRIBATOR")
        print("Нажимайте Enter для значений по умолчанию.")
        print()

        file_path = self.get_input("Путь к файлу", self.last_input_file)
        if not file_path:
            return
        if not os.path.exists(file_path):
            print(f"❌ Файл не найден: {file_path}")
            input("\nНажмите Enter для продолжения...")
            return
        self.selected_file = file_path
        self.last_input_file = file_path

        output_dir = self.get_input("Выходная папка", self.global_settings['output_dir'])

        # Устройство всегда CUDA
        device = "cuda"
        print(f"Устройство: CUDA (GPU)")

        default_compute = self.global_settings['compute_type']
        if default_compute == "int8":
            default_compute = "float16"

        def compute_validator(v: str) -> bool:
            if v in self.compute_types:
                return True
            print(f"❌ Допустимо: {', '.join(self.compute_types)}")
            return False

        compute_type = self.get_input(f"Тип вычислений ({'/'.join(self.compute_types)})", default_compute, validator=compute_validator)

        def batch_validator(v: str) -> bool:
            try:
                return int(v) > 0
            except ValueError:
                print("❌ Введите целое число > 0")
                return False

        batch_size = int(self.get_input("Batch size", str(self.global_settings['batch_size']), validator=batch_validator))

        align_raw = self.get_input(
            "Alignment (y/n)",
            "y" if self.global_settings['align'] else "n",
        )
        show_conf_raw = self.get_input(
            "Показывать уверенность (y/n)",
            "y" if self.global_settings['show_confidence'] else "n",
        )

        def model_validator(v: str) -> bool:
            if v in self.models:
                return True
            print(f"❌ Допустимо: {', '.join(self.models)}")
            return False

        model = self.get_input("Модель", self.default_file_settings['model'], validator=model_validator)
        language = self.get_input("Язык (код)", self.default_file_settings['language'])

        formats_raw = self.get_input(
            "Форматы (text,json,srt,vtt,all)",
            ",".join(self.default_file_settings['formats']) if self.default_file_settings['formats'] else "all"
        )
        formats = self._parse_formats(formats_raw)

        diarization_raw = self.get_input(
            "Определение спикеров (y/n)",
            "y" if not self.default_file_settings['no_diarization'] else "n",
        )
        no_diarization = not self._parse_yes_no(diarization_raw, default=not self.default_file_settings['no_diarization'])

        num_speakers_raw = input("Количество спикеров (опционально): ").strip()
        min_speakers_raw = input("Минимум спикеров (опционально): ").strip()
        max_speakers_raw = input("Максимум спикеров (опционально): ").strip()

        num_speakers = int(num_speakers_raw) if num_speakers_raw.isdigit() else None
        min_speakers = int(min_speakers_raw) if min_speakers_raw.isdigit() else None
        max_speakers = int(max_speakers_raw) if max_speakers_raw.isdigit() else None
        if num_speakers is not None:
            min_speakers = None
            max_speakers = None

        hf_token = input("HF токен (Enter = оставить текущий): ").strip()
        if not hf_token:
            hf_token = self.global_settings['hf_token']

        self.global_settings.update({
            "output_dir": output_dir,
            "device": device,
            "compute_type": compute_type,
            "batch_size": batch_size,
            "align": self._parse_yes_no(align_raw, self.global_settings['align']),
            "show_confidence": self._parse_yes_no(show_conf_raw, self.global_settings['show_confidence']),
            "hf_token": hf_token,
        })
        self._coerce_device_compute()

        self.default_file_settings.update({
            "model": model,
            "language": language,
            "formats": formats,
            "no_diarization": no_diarization,
        })

        self._save_settings()

        file_settings = {
            "model": model,
            "language": language,
            "formats": formats,
            "no_diarization": no_diarization,
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }

        self._run_with_settings(file_settings)
    
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
            print(f"  Batch size: {self.global_settings['batch_size']}")
            print(f"  Alignment: {'включено' if self.global_settings['align'] else 'выключено'}")
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
                ('2', 'Информация об устройстве (CUDA)'),
                ('3', 'Изменить тип вычислений'),
                ('4', 'Настроить HuggingFace токен'),
                ('5', 'Изменить batch size'),
                ('6', 'Включить/выключить alignment'),
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
                self._configure_batch_size()
            elif choice == 6:
                self._toggle_alignment()
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
        """Настройка устройства (только CUDA)"""
        self.clear_screen()
        self.print_header("УСТРОЙСТВО")
        print("Проект настроен только на работу с CUDA GPU.")
        print(f"Текущее устройство: CUDA (GPU)")
        print()
        
        # Проверяем доступность CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA доступна")
                print(f"  └─ GPU: {torch.cuda.get_device_name(0)}")
                print(f"  └─ CUDA версия: {torch.version.cuda}")
            else:
                print("❌ CUDA недоступна!")
                print("Убедитесь, что установлен PyTorch с поддержкой CUDA и драйверы NVIDIA.")
        except ImportError:
            print("❌ PyTorch не установлен!")
            print("Установите PyTorch с поддержкой CUDA.")
        except Exception as e:
            print(f"❌ Ошибка при проверке CUDA: {e}")
        
        input("\nНажмите Enter для продолжения...")
    
    def _configure_compute_type(self):
        """Настройка типа вычислений"""
        self.clear_screen()
        self.print_header("ТИП ВЫЧИСЛЕНИЙ")
        print("Доступные типы:")
        for i, ct in enumerate(self.compute_types, 1):
            marker = " ← текущий" if ct == self.global_settings['compute_type'] else ""
            desc = {
                'float16': 'Оптимально для GPU (рекомендуется)',
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
    
    def _configure_batch_size(self):
        """Настройка batch size"""
        self.clear_screen()
        self.print_header("BATCH SIZE")
        print(f"Текущее значение: {self.global_settings['batch_size']}")
        print("Рекомендация: 8-24 для CUDA GPU (зависит от доступной VRAM)")
        print()

        new_value = input(f"Введите новое значение [{self.global_settings['batch_size']}]: ").strip()
        if new_value:
            try:
                value = int(new_value)
                if value > 0:
                    self.global_settings['batch_size'] = value
                    print(f"✓ Batch size установлен: {value}")
                else:
                    print("❌ Значение должно быть > 0")
            except ValueError:
                print("❌ Неверное значение")
        else:
            print("Значение не изменено")

        input("\nНажмите Enter для продолжения...")

    def _toggle_alignment(self):
        """Переключение alignment"""
        self.clear_screen()
        self.print_header("ALIGNMENT")
        current = "включено" if self.global_settings['align'] else "выключено"
        print(f"Текущее состояние: {current}")
        print()
        print("1. Включить alignment (word-level timestamps)")
        print("2. Выключить alignment")
        print()

        choice = self.get_choice(2)
        if choice == 1:
            self.global_settings['align'] = True
            print("✓ Alignment включен")
        elif choice == 2:
            self.global_settings['align'] = False
            print("✓ Alignment выключен")

        input("\nНажмите Enter для продолжения...")

    def _build_cli_command(self, settings: Dict) -> str:
        def quote(value: str) -> str:
            if " " in value or "(" in value or ")" in value:
                return f"\"{value}\""
            return value

        cmd = ["python", "main.py", quote(settings['input_file'])]
        cmd += ["--output-dir", quote(settings['output_dir'])]
        cmd += ["--model", settings['model']]
        cmd += ["--language", settings['language']]
        cmd += ["--device", settings['device']]
        cmd += ["--compute-type", settings['compute_type']]
        cmd += ["--batch-size", str(settings['batch_size'])]
        cmd += ["--no-align"] if not settings['align'] else ["--align"]

        if settings['no_diarization']:
            cmd.append("--no-diarization")
        else:
            if settings.get('num_speakers'):
                cmd += ["--num-speakers", str(settings['num_speakers'])]
            else:
                if settings.get('min_speakers') is not None:
                    cmd += ["--min-speakers", str(settings['min_speakers'])]
                if settings.get('max_speakers') is not None:
                    cmd += ["--max-speakers", str(settings['max_speakers'])]
            if settings.get('hf_token') and settings['hf_token'] != 'your_huggingface_token_here':
                cmd += ["--hf-token", settings['hf_token']]

        if 'all' in settings['formats']:
            cmd += ["--formats", "all"]
        else:
            cmd += ["--formats"] + settings['formats']

        if settings.get('show_confidence'):
            cmd.append("--show-confidence")

        return " ".join(cmd)

    def _save_cli_command_prompt(self, settings: Dict):
        """Сохранить команду CLI"""
        print()
        choice = input("Сохранить команду для CLI? (y/n) [n]: ").strip().lower()
        if choice != 'y':
            return
        name = input("Имя команды (например: meeting_fast): ").strip()
        if not name:
            print("❌ Имя не указано, команда не сохранена.")
            return
        command_str = self._build_cli_command(settings)
        try:
            save_command(name, command_str)
            print(f"✓ Команда сохранена: {name}")
        except Exception as e:
            print(f"❌ Не удалось сохранить: {e}")

    def _print_saved_commands(self):
        """Показать сохраненные команды"""
        self.clear_screen()
        self.print_header("СОХРАНЕННЫЕ КОМАНДЫ")
        commands = list_commands()
        if not commands:
            print("Сохраненных команд нет.")
        else:
            for item in commands:
                desc = f" - {item['description']}" if item.get("description") else ""
                print(f"• {item['name']}{desc}")
                print(f"  {item['command']}")
                print()
        input("Нажмите Enter для продолжения...")

    def _quick_preset_settings(self) -> Optional[Dict]:
        """Быстрый выбор пресета"""
        self.clear_screen()
        self.print_header("БЫСТРЫЙ СТАРТ")
        print("Выберите пресет качества:")
        print("1. Быстро (small, alignment off, batch 8)")
        print("2. Сбалансировано (small, alignment on, batch 16)")
        print("3. Точно (medium, alignment on, batch 16)")
        print()
        choice = self.get_choice(3)
        if choice == 0:
            return None

        file_settings = {
            'model': self.default_file_settings['model'],
            'language': self.default_file_settings['language'],
            'formats': self.default_file_settings['formats'].copy(),
            'no_diarization': self.default_file_settings['no_diarization'],
            'num_speakers': None,
            'min_speakers': None,
            'max_speakers': None
        }

        if choice == 1:
            file_settings['model'] = 'small'
            self.global_settings['batch_size'] = 8
            self.global_settings['align'] = False
        elif choice == 2:
            file_settings['model'] = 'small'
            self.global_settings['batch_size'] = 16
            self.global_settings['align'] = True
        elif choice == 3:
            file_settings['model'] = 'medium'
            self.global_settings['batch_size'] = 16
            self.global_settings['align'] = True

        print()
        print("Определение спикеров:")
        print("1. Включить")
        print("2. Выключить")
        sp = self.get_choice(2)
        if sp == 2:
            file_settings['no_diarization'] = True
        elif sp == 1:
            file_settings['no_diarization'] = False
            print("Количество спикеров (опционально)")
            print("1. Указать точное")
            print("2. Указать диапазон")
            print("3. Не указывать")
            sp_choice = self.get_choice(3)
            if sp_choice == 1:
                num = input("Сколько спикеров: ").strip()
                try:
                    file_settings['num_speakers'] = int(num)
                except ValueError:
                    pass
            elif sp_choice == 2:
                min_sp = input("Минимум: ").strip()
                max_sp = input("Максимум: ").strip()
                try:
                    file_settings['min_speakers'] = int(min_sp) if min_sp else None
                    file_settings['max_speakers'] = int(max_sp) if max_sp else None
                except ValueError:
                    pass

        return file_settings

    def _simple_wizard_settings(self) -> Optional[Dict]:
        """Пошаговый мастер настроек"""
        self.clear_screen()
        self.print_header("МАСТЕР НАСТРОЕК")
        print("Ответьте на 4 простых вопроса.")
        print()

        file_settings = {
            'model': self.default_file_settings['model'],
            'language': self.default_file_settings['language'],
            'formats': self.default_file_settings['formats'].copy(),
            'no_diarization': self.default_file_settings['no_diarization'],
            'num_speakers': None,
            'min_speakers': None,
            'max_speakers': None
        }

        print("1) Качество распознавания:")
        print("1. Быстро (small)")
        print("2. Сбалансировано (small)")
        print("3. Максимально точно (medium)")
        quality = self.get_choice(3)
        if quality == 1:
            file_settings['model'] = 'small'
            self.global_settings['batch_size'] = 8
            self.global_settings['align'] = False
        elif quality == 2:
            file_settings['model'] = 'small'
            self.global_settings['batch_size'] = 16
            self.global_settings['align'] = True
        elif quality == 3:
            file_settings['model'] = 'medium'
            self.global_settings['batch_size'] = 16
            self.global_settings['align'] = True

        print()
        lang = input("2) Язык (например ru, en) [ru]: ").strip().lower()
        if lang:
            file_settings['language'] = lang

        print()
        print("3) Нужны ли спикеры?")
        print("1. Да")
        print("2. Нет")
        sp = self.get_choice(2)
        if sp == 2:
            file_settings['no_diarization'] = True
        else:
            file_settings['no_diarization'] = False

        print()
        fmt = input("4) Форматы (txt/json/srt/vtt/all) [all]: ").strip().lower()
        if fmt:
            if fmt == "all":
                file_settings['formats'] = ['all']
            else:
                parts = [p.strip() for p in fmt.split(",") if p.strip()]
                if parts:
                    file_settings['formats'] = parts

        return file_settings

    def _quick_start_menu(self):
        """Меню быстрого старта"""
        if not self.selected_file:
            self.select_file()
            if not self.selected_file:
                return

        self.clear_screen()
        self.print_header("БЫСТРЫЙ СТАРТ")
        print("Выберите способ настройки:")
        print("1. Пресеты (быстро/сбалансировано/точно)")
        print("2. Мастер (4 вопроса)")
        print()
        choice = self.get_choice(2)
        if choice == 0:
            return

        if choice == 1:
            file_settings = self._quick_preset_settings()
        else:
            file_settings = self._simple_wizard_settings()

        if not file_settings:
            return

        self._run_with_settings(file_settings)
    
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
        except ValueError:
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
        print(f"  Batch size: {self.global_settings['batch_size']}")
        print(f"  Alignment: {'включено' if self.global_settings['align'] else 'выключено'}")
        print(f"  Показывать уверенность: {'да' if self.global_settings['show_confidence'] else 'нет'}")
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
            except ValueError:
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
            except ValueError:
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
            except ValueError:
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
                except ValueError:
                    pass
            elif choice == 'b':
                min_sp = input("Минимум: ").strip()
                max_sp = input("Максимум: ").strip()
                try:
                    file_settings['min_speakers'] = int(min_sp) if min_sp else None
                    file_settings['max_speakers'] = int(max_sp) if max_sp else None
                    file_settings['num_speakers'] = None
                except ValueError:
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
    
    def _run_with_settings(self, file_settings: Dict):
        """Запуск транскрибации с готовыми настройками"""
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

        # Объединяем глобальные и файловые настройки
        settings = {**self.global_settings, **file_settings, 'input_file': self.selected_file}

        # Принудительно используем CUDA и float16
        if settings['device'] != 'cuda':
            raise ValueError(f"Проект настроен только на CUDA GPU. Получено устройство: {settings['device']}")
        if settings['compute_type'] == 'int8':
            settings['compute_type'] = 'float16'
        
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
            print("[1/3] ПРЕДОБРАБОТКА АУДИО")
            print("-" * 80)
            processor = AudioProcessor()
            processed_audio = processor.preprocess_audio(settings['input_file'])
            audio_duration = processor.get_audio_duration(processed_audio)
            print(f"✓ Предобработка завершена! Длительность: {audio_duration:.2f} сек ({audio_duration/60:.1f} мин)")
            print()
            
            # Шаг 2: WhisperX
            print("[2/3] WHISPERX")
            print("-" * 80)
            merged_segments, wx_metadata = run_whisperx_pipeline(
                audio_file=processed_audio,
                model_size=settings['model'],
                device=settings['device'],
                compute_type=settings['compute_type'],
                batch_size=settings['batch_size'],
                language=settings['language'],
                align=settings['align'],
                diarize=not settings['no_diarization'],
                hf_token=settings['hf_token'],
                num_speakers=settings['num_speakers'],
                min_speakers=settings['min_speakers'],
                max_speakers=settings['max_speakers']
            )
            print(f"✓ WhisperX завершен! Сегментов: {len(merged_segments)}")
            print()

            # Шаг 3: Экспорт результатов
            print("[3/3] ЭКСПОРТ РЕЗУЛЬТАТОВ")
            print("-" * 80)
            
            metadata = {
                'source_file': os.path.basename(settings['input_file']),
                'model': wx_metadata.get('model', settings['model']),
                'language': wx_metadata.get('language', settings['language']),
                'duration': audio_duration,
                'diarization_enabled': not settings['no_diarization'],
                'aligned': wx_metadata.get('aligned', settings['align']),
                'batch_size': wx_metadata.get('batch_size', settings['batch_size'])
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
                except OSError as e:
                    print(f"⚠️  Не удалось удалить временный файл: {processed_audio} ({e})")
            
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
            self._save_cli_command_prompt(settings)
            
        except KeyboardInterrupt:
            print("\n\n❌ Операция прервана пользователем")
        except Exception as e:
            print(f"\n\n❌ Ошибка: {str(e)}")
            import traceback
            traceback.print_exc()
        
        input("\nНажмите Enter для продолжения...")

    def run_transcription(self):
        """Запуск транскрибации"""
        # Запрашиваем параметры для этого файла
        file_settings = self.request_file_settings()
        if not file_settings:
            return  # Пользователь отменил
        self._run_with_settings(file_settings)
    
    def main_menu(self):
        """Главное меню"""
        while True:
            self.clear_screen()
            self.print_header()
            
            menu_items = [
                ('1', 'Выбрать файл для обработки'),
                ('2', 'Настроить параметры'),
                ('3', 'Быстрый старт (пресеты/мастер)'),
                ('4', 'Показать сводку настроек'),
                ('5', 'Запустить транскрибацию'),
                ('6', 'Сохраненные команды'),
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
                self._quick_start_menu()
            elif choice == 4:
                self.show_summary()
            elif choice == 5:
                self.run_transcription()
            elif choice == 6:
                self._print_saved_commands()


def main():
    """Главная функция"""
    try:
        interface = TranscribatorInterface()
        interface.run_linear()
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

