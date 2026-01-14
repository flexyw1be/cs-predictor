#!/bin/bash
# Скрипт для скачивания готового датасета с Kaggle

echo "=========================================="
echo "ЗАГРУЗКА CS:GO ДАТАСЕТА С KAGGLE"
echo "=========================================="

# Проверка установки kaggle
if ! command -v kaggle &> /dev/null; then
    echo "Установка kaggle CLI..."
    pip install kaggle
fi

# Проверка настройки API
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "ОШИБКА: Не найден файл ~/.kaggle/kaggle.json"
    echo ""
    echo "Для настройки:"
    echo "1. Зайдите на https://www.kaggle.com/settings"
    echo "2. Прокрутите до 'API' и нажмите 'Create New Token'"
    echo "3. Скачается файл kaggle.json"
    echo "4. Выполните:"
    echo "   mkdir -p ~/.kaggle"
    echo "   mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# Создаём папку для данных
mkdir -p data/kaggle

echo ""
echo "Скачивание датасета CS:GO Professional Matches..."
kaggle datasets download -d mateusdmachado/csgo-professional-matches -p data/kaggle

echo ""
echo "Распаковка..."
cd data/kaggle
unzip -o *.zip

echo ""
echo "Готово! Файлы в папке data/kaggle:"
ls -la

echo ""
echo "Теперь запустите:"
echo "  python scraper/convert_kaggle_data.py"
