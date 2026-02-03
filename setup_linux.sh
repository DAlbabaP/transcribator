#!/bin/bash

echo "================================================================================"
echo "TRANSCRIBATOR - Скрипт автоматической установки (Linux/macOS)"
echo "================================================================================"
echo ""

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция для вывода ошибок
error() {
    echo -e "${RED}ОШИБКА: $1${NC}"
    exit 1
}

# Функция для вывода успеха
success() {
    echo -e "${GREEN}OK: $1${NC}"
}

# Функция для вывода предупреждений
warning() {
    echo -e "${YELLOW}ВНИМАНИЕ: $1${NC}"
}

# [1/6] Проверка Python
echo "[1/6] Проверка Python..."
if ! command -v python3 &> /dev/null; then
    error "Python не найден! Установите Python 3.10 или новее"
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    error "Требуется Python 3.10 или новее. Установлена версия: $PYTHON_VERSION"
fi

success "Python $PYTHON_VERSION установлен"
echo ""

# [2/6] Проверка FFmpeg
echo "[2/6] Проверка FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    warning "FFmpeg не найден!"
    echo "Установите FFmpeg:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    read -p "Продолжить без FFmpeg? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    success "FFmpeg установлен"
fi
echo ""

# [3/6] Создание виртуального окружения
echo "[3/6] Создание виртуального окружения..."
if [ -d "venv" ]; then
    warning "Виртуальное окружение уже существует, пропускаем..."
else
    python3 -m venv venv || error "Не удалось создать виртуальное окружение"
    success "Виртуальное окружение создано"
fi
echo ""

# [4/6] Активация окружения
echo "[4/6] Активация окружения..."
source venv/bin/activate || error "Не удалось активировать окружение"
success "Окружение активировано"
echo ""

# [5/6] Обновление pip
echo "[5/6] Обновление pip..."
pip install --upgrade pip
echo ""

# [6/6] Установка зависимостей
echo "[6/6] Установка зависимостей (это займет время)..."
echo "ВАЖНО: Устанавливаем в правильном порядке для избежания конфликтов"
echo ""

echo "Шаг 1/6: PyTorch..."
pip install torch==2.10.0 torchaudio==2.10.0 || error "Не удалось установить PyTorch"
echo ""

echo "Шаг 2/6: pyannote.audio..."
pip install pyannote.audio==4.0.3 || error "Не удалось установить pyannote.audio"
echo ""

echo "Шаг 3/6: faster-whisper..."
pip install faster-whisper==1.2.1 || error "Не удалось установить faster-whisper"
echo ""

echo "Шаг 4/6: onnxruntime..."
pip install onnxruntime || error "Не удалось установить onnxruntime"
echo ""

echo "Шаг 5/6: Аудио библиотеки..."
pip install pydub ffmpeg-python || error "Не удалось установить аудио библиотеки"
echo ""

echo "Шаг 6/6: Утилиты..."
pip install python-dotenv || error "Не удалось установить утилиты"
echo ""

# Создание .env файла
echo "Настройка конфигурации..."
if [ ! -f .env ]; then
    cp .env.example .env
    success "Создан файл .env"
    warning "ВАЖНО: Отредактируйте .env и укажите ваш HuggingFace токен!"
else
    warning "Файл .env уже существует, пропускаем..."
fi
echo ""

# Проверка установки
echo "Проверка установки..."
if python main.py --help &> /dev/null; then
    success "Программа готова к использованию"
else
    warning "Возможны проблемы с установкой"
fi
echo ""

echo "================================================================================"
echo "УСТАНОВКА ЗАВЕРШЕНА!"
echo "================================================================================"
echo ""
echo "Что дальше:"
echo "1. Активируйте окружение:"
echo "   source venv/bin/activate"
echo ""
echo "2. Отредактируйте файл .env и укажите ваш HuggingFace токен:"
echo "   - Зарегистрируйтесь на https://huggingface.co"
echo "   - Создайте токен: https://huggingface.co/settings/tokens"
echo "   - Примите лицензии моделей (см. .env.example)"
echo ""
echo "3. Запустите первую транскрибацию:"
echo "   python main.py ваш_файл.mp3"
echo ""
echo "4. Для справки:"
echo "   python main.py --help"
echo ""
echo "Подробная документация в README.md"
echo "================================================================================"
