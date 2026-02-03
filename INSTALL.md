# Подробная инструкция по установке

Пошаговое руководство для пользователей без опыта работы с Python.

## Шаг 1: Установка Python

### Windows

1. Скачайте Python 3.10 или новее с [python.org](https://www.python.org/downloads/)
2. **ВАЖНО**: При установке поставьте галочку "Add Python to PATH"
3. Проверьте установку, открыв командную строку (Win+R → `cmd`) и введя:
   ```bash
   python --version
   ```
   Должно показать версию >= 3.10

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

### macOS

```bash
# Через Homebrew
brew install python@3.10
```

## Шаг 2: Установка FFmpeg

### Windows

**Способ 1: Chocolatey (рекомендуется)**
1. Установите [Chocolatey](https://chocolatey.org/install)
2. Откройте PowerShell от имени администратора
3. Выполните:
   ```powershell
   choco install ffmpeg
   ```

**Способ 2: Вручную**
1. Скачайте FFmpeg с [ffmpeg.org](https://ffmpeg.org/download.html)
2. Распакуйте архив (например, в `C:\ffmpeg`)
3. Добавьте `C:\ffmpeg\bin` в PATH:
   - Откройте "Система" → "Дополнительные параметры системы"
   - "Переменные среды"
   - В "Системные переменные" найдите Path
   - Добавьте путь к bin папке FFmpeg
4. Перезапустите терминал

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install ffmpeg
```

### macOS

```bash
brew install ffmpeg
```

**Проверка установки:**
```bash
ffmpeg -version
```

## Шаг 3: Загрузка проекта

### Вариант A: С помощью Git

```bash
git clone https://github.com/yourusername/transcribator.git
cd transcribator
```

### Вариант B: Скачать ZIP

1. Скачайте проект как ZIP архив
2. Распакуйте в удобное место
3. Откройте командную строку в папке проекта

## Шаг 4: Создание виртуального окружения

### Windows

```bash
# Создание окружения
python -m venv venv

# Активация
venv\Scripts\activate

# Если появляется ошибка политики выполнения:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux/macOS

```bash
# Создание окружения
python3 -m venv venv

# Активация
source venv/bin/activate
```

**Как понять что окружение активировано?**
В начале строки терминала появится `(venv)`.

## Шаг 5: Установка зависимостей

**⚠️ КРИТИЧЕСКИ ВАЖНО: Устанавливайте в указанном порядке!**

```bash
# Обновляем pip
pip install --upgrade pip

# Устанавливаем в правильном порядке (избегаем конфликтов)
pip install torch==2.10.0 torchaudio==2.10.0
pip install pyannote.audio==4.0.3
pip install faster-whisper==1.2.1
pip install onnxruntime
pip install pydub ffmpeg-python python-dotenv
```

**Проверка установки:**
```bash
pip list | findstr "whisper pyannote torch"
```

Должны быть установлены:
- faster-whisper 1.2.1
- pyannote.audio 4.0.3
- torch 2.10.0

## Шаг 6: Получение HuggingFace токена

1. **Регистрация на HuggingFace**
   - Перейдите на [huggingface.co](https://huggingface.co)
   - Нажмите "Sign Up" (Зарегистрироваться)
   - Подтвердите email

2. **Создание токена**
   - Войдите в аккаунт
   - Перейдите в [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - Нажмите "New token"
   - Имя: `transcribator` (любое)
   - Тип: `Read` (достаточно)
   - Создайте и скопируйте токен

3. **Принятие лицензий моделей (ОБЯЗАТЕЛЬНО!)**
   
   Откройте эти страницы и нажмите "Agree and access repository":
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   
   Без этого шага модели не будут работать!

## Шаг 7: Настройка конфигурации

1. **Создание .env файла**
   
   ### Windows (командная строка)
   ```bash
   copy .env.example .env
   ```
   
   ### Linux/macOS
   ```bash
   cp .env.example .env
   ```

2. **Редактирование .env файла**
   
   Откройте `.env` в любом текстовом редакторе (Блокнот, VS Code, Notepad++) и измените:
   
   ```env
   # Вставьте ваш реальный токен вместо your_huggingface_token_here
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   
   # Остальные настройки можно оставить как есть
   WHISPER_MODEL=small
   DEFAULT_LANGUAGE=ru
   OUTPUT_DIR=./output
   ```

3. **Сохраните файл**

## Шаг 8: Проверка установки

Запустите тестовую команду:

```bash
python main.py --help
```

Должна показаться справка по использованию программы.

## Шаг 9: Первый запуск

При первом запуске скачаются модели (~5 GB), это займет время:

```bash
# Подготовьте небольшой тестовый аудио файл (например, test.mp3)
python main.py test.mp3 --no-diarization
```

Параметр `--no-diarization` пропустит определение спикеров для быстрого теста.

## Решение проблем при установке

### Ошибка: "python" не является внутренней командой

**Решение:**
- Python не добавлен в PATH
- Переустановите Python с галочкой "Add to PATH"
- Или используйте полный путь: `C:\Python310\python.exe`

### Ошибка: pip не найден

**Решение:**
```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Ошибка: не удается активировать venv на Windows

**Решение:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Ошибка при установке pyannote.audio

**Решение:**
```bash
# Установите Visual C++ Build Tools для Windows
# Скачать с: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Медленная установка зависимостей

**Решение:**
```bash
# Используйте другое зеркало PyPI
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
```

### Ошибка: недостаточно места на диске

**Требования:**
- ~5 GB для моделей
- ~2 GB для зависимостей
- Освободите место или выберите другой диск

## Дополнительные инструменты (опционально)

### Visual Studio Code (рекомендуется)

Удобный редактор для работы с проектом:
1. Скачайте с [code.visualstudio.com](https://code.visualstudio.com/)
2. Установите расширение Python
3. Откройте папку проекта в VS Code

### Git (для обновлений)

```bash
# Windows
choco install git

# Linux
sudo apt install git

# macOS
brew install git
```

## Что дальше?

После успешной установки переходите к [README.md](README.md) для изучения использования программы.

## Нужна помощь?

Если что-то не работает:
1. Проверьте этот гайд еще раз
2. Посмотрите раздел Troubleshooting в README.md
3. Создайте Issue на GitHub с описанием проблемы
