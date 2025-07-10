FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .

# Обновление pip
RUN pip install --no-cache-dir --upgrade pip

# Установка пакетов в правильном порядке
RUN pip install --no-cache-dir fastapi==0.104.1
RUN pip install --no-cache-dir uvicorn[standard]==0.24.0
RUN pip install --no-cache-dir pandas==2.1.3
RUN pip install --no-cache-dir requests==2.31.0
RUN pip install --no-cache-dir tqdm==4.66.1
RUN pip install --no-cache-dir openpyxl==3.1.2
RUN pip install --no-cache-dir torch==2.1.1
RUN pip install --no-cache-dir transformers==4.35.2
RUN pip install --no-cache-dir huggingface-hub==0.19.4
RUN pip install --no-cache-dir sentence-transformers==2.2.2
RUN pip install --no-cache-dir python-multipart==0.0.6

# Копирование кода приложения
COPY . .

# Создание директории для логов
RUN mkdir -p /app/logs

# Открытие порта
EXPOSE 8000

# Запуск приложения
CMD ["python", "AISearchCat.py"] 