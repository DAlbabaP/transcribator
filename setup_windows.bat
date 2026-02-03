@echo off
echo ================================================================================
echo TRANSCRIBATOR - Скрипт автоматической установки (Windows)
echo ================================================================================
echo.

REM Проверка Python
echo [1/6] Проверка Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: Python не найден!
    echo Установите Python 3.10 или новее с https://www.python.org/downloads/
    echo При установке отметьте "Add Python to PATH"
    pause
    exit /b 1
)
python --version
echo OK: Python установлен
echo.

REM Проверка FFmpeg
echo [2/6] Проверка FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: FFmpeg не найден!
    echo Установите FFmpeg:
    echo   1. Через Chocolatey: choco install ffmpeg
    echo   2. Или скачайте с https://ffmpeg.org/download.html
    pause
    exit /b 1
)
echo OK: FFmpeg установлен
echo.

REM Создание виртуального окружения
echo [3/6] Создание виртуального окружения...
if exist venv (
    echo Виртуальное окружение уже существует, пропускаем...
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ОШИБКА при создании виртуального окружения
        pause
        exit /b 1
    )
    echo OK: Виртуальное окружение создано
)
echo.

REM Активация окружения
echo [4/6] Активация окружения...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ОШИБКА при активации окружения
    pause
    exit /b 1
)
echo OK: Окружение активировано
echo.

REM Обновление pip
echo [5/6] Обновление pip...
python -m pip install --upgrade pip
echo.

REM Установка зависимостей
echo [6/6] Установка зависимостей (это займет время)...
echo ВАЖНО: Устанавливаем в правильном порядке для избежания конфликтов
echo.
echo Шаг 1/6: PyTorch...
pip install torch==2.10.0 torchaudio==2.10.0
echo.
echo Шаг 2/6: pyannote.audio...
pip install pyannote.audio==4.0.3
echo.
echo Шаг 3/6: faster-whisper...
pip install faster-whisper==1.2.1
echo.
echo Шаг 4/6: onnxruntime...
pip install onnxruntime
echo.
echo Шаг 5/6: Аудио библиотеки...
pip install pydub ffmpeg-python
echo.
echo Шаг 6/6: Утилиты...
pip install python-dotenv
echo.

REM Создание .env файла
echo Настройка конфигурации...
if not exist .env (
    copy .env.example .env
    echo OK: Создан файл .env
    echo ВАЖНО: Отредактируйте .env и укажите ваш HuggingFace токен!
) else (
    echo Файл .env уже существует, пропускаем...
)
echo.

REM Проверка установки
echo Проверка установки...
python main.py --help >nul 2>&1
if %errorlevel% neq 0 (
    echo ВНИМАНИЕ: Возможны проблемы с установкой
) else (
    echo OK: Программа готова к использованию
)
echo.

echo ================================================================================
echo УСТАНОВКА ЗАВЕРШЕНА!
echo ================================================================================
echo.
echo Что дальше:
echo 1. Отредактируйте файл .env и укажите ваш HuggingFace токен
echo    - Зарегистрируйтесь на https://huggingface.co
echo    - Создайте токен: https://huggingface.co/settings/tokens
echo    - Примите лицензии моделей (см. .env.example)
echo.
echo 2. Запустите первую транскрибацию:
echo    python main.py ваш_файл.mp3
echo.
echo 3. Для справки:
echo    python main.py --help
echo.
echo Подробная документация в README.md
echo ================================================================================
pause
