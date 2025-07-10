# 🤖 AI Category Matcher Service

**Автоматическая категоризация товаров с помощью искусственного интеллекта**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![Railway](https://img.shields.io/badge/Deployed%20on-Railway-0B0D0E.svg)](https://railway.app)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Описание

AI Category Matcher Service - это интеллектуальный сервис для автоматической категоризации товаров на основе их названий. Использует современные технологии машинного обучения и естественного языка для точного определения категорий товаров.

### ✨ Возможности

- 🧠 **AI-категоризация** - автоматическое определение категорий товаров
- 📊 **База данных** - 207,801 товар с готовыми эмбеддингами
- 🔍 **Семантический поиск** - поиск по смыслу, а не только по ключевым словам
- 🌐 **REST API** - удобное API для интеграции с любыми системами
- 🔗 **n8n интеграция** - готовые эндпоинты для workflow автоматизации
- ☁️ **Облачное развертывание** - готов к работе на Railway

## 🚀 Быстрый старт

### 1. Клонируйте репозиторий
```bash
git clone <your-repo-url>
cd AI_Category_Service
```

### 2. Установите зависимости
```bash
pip install -r requirements.txt
```

### 3. Настройте переменные окружения
```bash
export OPENROUTER_API_KEY="your-api-key"
export EMBEDDINGS_DOWNLOAD_URL="https://your-dropbox-url/products_with_embeddings.pkl"
```

### 4. Запустите сервис
```bash
python AISearchCat.py
```

Сервис будет доступен по адресу: `http://localhost:8000`

## 🚀 Развертывание на Railway

### 1. Создайте проект на Railway
- Подключите GitHub репозиторий
- Railway автоматически определит Dockerfile

### 2. Установите переменные окружения
```bash
OPENROUTER_API_KEY=your-openrouter-api-key
EMBEDDINGS_DOWNLOAD_URL=https://your-dropbox-url/products_with_embeddings.pkl
PORT=8000
HOST=0.0.0.0
```

### 3. Загрузите файл эмбеддингов
- Загрузите `products_with_embeddings.pkl` (396MB) на Dropbox
- Получите прямую ссылку для скачивания
- Установите как `EMBEDDINGS_DOWNLOAD_URL`

### 4. Получите домен
- В настройках проекта найдите "Domains"
- Нажмите "Generate Domain"
- Скопируйте полученный URL

## 📡 API Endpoints

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| `GET` | `/` | Главная страница API |
| `GET` | `/health` | Проверка здоровья сервиса |
| `POST` | `/n8n_process` | Обработка товаров для n8n |
| `GET` | `/debug` | Диагностическая информация |
| `GET` | `/test_search` | Тестирование поиска |

### Пример использования

```bash
# Проверка здоровья
curl https://your-domain.railway.app/health

# Категоризация товара
curl -X POST https://your-domain.railway.app/n8n_process \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Мотоблок дизельный 7 л.с.",
    "code": "534143"
  }'
```

**Ответ:**
```json
{
  "status": "success",
  "data": {
    "Артикул": "534143",
    "Наименование": "Мотоблок дизельный 7 л.с.",
    "Категория": "Мотоблоки",
    "Подкатегория": "Дизельные мотоблоки",
    "Комментарий": "95"
  },
  "processed_items": 1
}
```

## 🔗 Интеграция с n8n

### Настройка workflow

1. **Создайте новый workflow в n8n**
2. **Добавьте HTTP Request узел:**
   - **Method**: POST
   - **URL**: `https://your-domain.railway.app/n8n_process`
   - **Headers**: `Content-Type: application/json`
   - **Body**:
   ```json
   {
     "name": "{{ $json.name }}",
     "code": "{{ $json.code }}"
   }
   ```

3. **Импортируйте готовый workflow** из `examples/n8n_workflow.json`

## 🏗️ Архитектура

### Технологический стек
- **Backend**: FastAPI, Python 3.11
- **AI/ML**: SentenceTransformers, PyTorch
- **База данных**: Pandas (pickle файл)
- **API**: OpenRouter (DeepSeek Chat)
- **Развертывание**: Docker, Railway

### Алгоритм работы
1. Получение названия товара
2. Создание эмбеддинга с помощью SentenceTransformers
3. Поиск похожих товаров в базе данных
4. AI анализ с помощью OpenRouter
5. Определение категории и подкатегории
6. Возврат результата с процентом уверенности

## 📊 Производительность

- **База данных**: 207,801 товар
- **Время загрузки**: ~30 секунд
- **Время обработки**: 2-5 секунд на товар
- **Точность**: 85-95%

## 🔧 Конфигурация

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `OPENROUTER_API_KEY` | API ключ OpenRouter | - |
| `EMBEDDINGS_DOWNLOAD_URL` | URL для загрузки файла эмбеддингов | - |
| `BASE_PKL_PATH` | Путь к файлу эмбеддингов | `products_with_embeddings.pkl` |
| `PORT` | Порт сервиса | `8000` |
| `HOST` | Хост сервиса | `0.0.0.0` |

## 🐳 Docker

### Локальное развертывание
```bash
# Сборка образа
docker build -t ai-category-service .

# Запуск контейнера
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY=your-api-key \
  -e EMBEDDINGS_DOWNLOAD_URL=your-url \
  ai-category-service
```

## 📁 Структура проекта

```
AI_Category_Service/
├── README.md                    # Основная документация
├── LICENSE                      # MIT лицензия
├── .gitignore                   # Исключения для Git
├── Dockerfile                   # Docker образ
├── requirements.txt             # Зависимости Python
├── AISearchCat.py               # Основной сервис
├── download_embeddings.py       # Загрузка эмбеддингов
├── docs/                        # Документация
│   ├── API.md                   # Документация API
│   └── DEPLOYMENT.md            # Руководство по развертыванию
└── examples/                    # Примеры
    ├── test_requests.py         # Тестовые запросы
    └── n8n_workflow.json        # Пример n8n workflow
```

## 🆘 Поддержка

### Частые проблемы

**Q: Ошибка загрузки файла эмбеддингов**
A: Проверьте URL и убедитесь, что файл доступен для скачивания.

**Q: Сервис не находит категорию**
A: Проверьте `/debug` эндпоинт для анализа базы данных.

**Q: API ключ не работает**
A: Убедитесь, что ключ правильный и есть баланс на OpenRouter.

### Контакты
- **GitHub Issues**: для баг-репортов
- **Документация**: `docs/` папка
- **Примеры**: `examples/` папка

## 📄 Лицензия

MIT License - см. файл [LICENSE](LICENSE) для подробностей.

---

**⭐ Если этот проект вам помог, поставьте звездочку на GitHub!**

*Создано с ❤️ для автоматизации бизнес-процессов* 