"""
Модуль предобработки аудио файлов

Функциональность:
- Извлечение аудио из видео файлов
- Конвертация в WAV формат
- Ресемплинг в 16kHz (требование Whisper)
- Конвертация в моно
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
from pydub import AudioSegment
import ffmpeg


class AudioProcessor:
    """Класс для предобработки аудио файлов"""
    
    SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac']
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm']
    TARGET_SAMPLE_RATE = 16000  # 16kHz для Whisper
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Args:
            temp_dir: Директория для временных файлов. Если None, используется системная.
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
    def preprocess_audio(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        Полная предобработка аудио файла
        
        Args:
            input_file: Путь к входному файлу (аудио или видео)
            output_file: Путь к выходному файлу. Если None, создается временный файл.
            
        Returns:
            Путь к обработанному WAV файлу (16kHz, моно)
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_file}")
        
        file_extension = input_path.suffix.lower()
        
        # Определяем, это видео или аудио
        if file_extension in self.SUPPORTED_VIDEO_FORMATS:
            print(f"📹 Обнаружен видео файл, извлекаем аудио...")
            audio_file = self._extract_audio_from_video(str(input_path))
        elif file_extension in self.SUPPORTED_AUDIO_FORMATS:
            print(f"🎵 Обнаружен аудио файл ({file_extension})")
            audio_file = str(input_path)
        else:
            raise ValueError(
                f"Неподдерживаемый формат файла: {file_extension}\n"
                f"Поддерживаемые аудио: {', '.join(self.SUPPORTED_AUDIO_FORMATS)}\n"
                f"Поддерживаемые видео: {', '.join(self.SUPPORTED_VIDEO_FORMATS)}"
            )
        
        # Конвертируем в WAV с нужными параметрами
        print(f"🔄 Конвертация: {file_extension} → WAV (16kHz, моно)...")
        processed_file = self._convert_to_wav(audio_file, output_file)
        
        # Удаляем временный файл, если он был создан
        if file_extension in self.SUPPORTED_VIDEO_FORMATS and audio_file != str(input_path):
            try:
                os.remove(audio_file)
            except OSError as e:
                print(f"⚠️  Не удалось удалить временный файл: {audio_file} ({e})")
        return processed_file
    
    def _extract_audio_from_video(self, video_file: str) -> str:
        """
        Извлечение аудио дорожки из видео файла
        
        Args:
            video_file: Путь к видео файлу
            
        Returns:
            Путь к извлеченному аудио файлу
        """
        output_file = os.path.join(
            self.temp_dir,
            f"extracted_audio_{os.getpid()}.wav"
        )
        
        try:
            # Используем ffmpeg для извлечения аудио
            stream = ffmpeg.input(video_file)
            stream = ffmpeg.output(stream, output_file, acodec='pcm_s16le', ac=1, ar=self.TARGET_SAMPLE_RATE)
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            return output_file
        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Ошибка при извлечении аудио из видео: {error_message}")
    
    def _convert_to_wav(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        Конвертация аудио в WAV формат с нужными параметрами
        
        Args:
            input_file: Путь к входному аудио файлу
            output_file: Путь к выходному WAV файлу. Если None, создается временный.
            
        Returns:
            Путь к WAV файлу
        """
        if output_file is None:
            output_file = os.path.join(
                self.temp_dir,
                f"processed_audio_{os.getpid()}.wav"
            )
        
        try:
            # Загружаем аудио
            audio = AudioSegment.from_file(input_file)
            
            # Конвертируем в моно если нужно
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Ресемплинг в 16kHz
            if audio.frame_rate != self.TARGET_SAMPLE_RATE:
                audio = audio.set_frame_rate(self.TARGET_SAMPLE_RATE)
            
            # Сохраняем как WAV
            audio.export(
                output_file,
                format='wav',
                parameters=['-acodec', 'pcm_s16le']
            )
            
            return output_file
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при конвертации аудио: {str(e)}")
    
    def get_audio_duration(self, audio_file: str) -> float:
        """
        Получение длительности аудио файла в секундах
        
        Args:
            audio_file: Путь к аудио файлу
            
        Returns:
            Длительность в секундах
        """
        try:
            audio = AudioSegment.from_file(audio_file)
            return len(audio) / 1000.0  # pydub возвращает в миллисекундах
        except Exception as e:
            raise RuntimeError(f"Ошибка при определении длительности: {str(e)}")


def extract_audio(input_file: str, output_file: str) -> str:
    """
    Вспомогательная функция для извлечения аудио из видео
    
    Args:
        input_file: Путь к входному файлу
        output_file: Путь к выходному файлу
        
    Returns:
        Путь к извлеченному аудио
    """
    processor = AudioProcessor()
    return processor._extract_audio_from_video(input_file)


def convert_to_wav(input_file: str, output_file: Optional[str] = None) -> str:
    """
    Вспомогательная функция для конвертации в WAV
    
    Args:
        input_file: Путь к входному файлу
        output_file: Путь к выходному файлу
        
    Returns:
        Путь к WAV файлу
    """
    processor = AudioProcessor()
    return processor._convert_to_wav(input_file, output_file)


def preprocess_audio(input_file: str, output_file: Optional[str] = None) -> str:
    """
    Вспомогательная функция для полной предобработки
    
    Args:
        input_file: Путь к входному файлу
        output_file: Путь к выходному файлу
        
    Returns:
        Путь к обработанному файлу
    """
    processor = AudioProcessor()
    return processor.preprocess_audio(input_file, output_file)
