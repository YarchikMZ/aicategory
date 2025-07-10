#!/usr/bin/env python3
"""
Скрипт для загрузки файла эмбеддингов из внешнего источника
"""
import os
import requests
import sys
from tqdm import tqdm

def download_file(url, filename):
    """Загружает файл с прогресс-баром"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        
        return True
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return False

def main():
    """Основная функция"""
    filename = "products_with_embeddings.pkl"
    
    # URL для загрузки файла (нужно будет заменить на реальный)
    # Можно использовать Google Drive, Dropbox, или другой сервис
    download_url = os.getenv("EMBEDDINGS_DOWNLOAD_URL")
    
    if not download_url:
        print("❌ Не указан URL для загрузки файла эмбеддингов")
        print("Установите переменную окружения EMBEDDINGS_DOWNLOAD_URL")
        return False
    
    if os.path.exists(filename):
        print(f"✅ Файл {filename} уже существует")
        return True
    
    print(f"📥 Загружаем файл эмбеддингов из {download_url}")
    success = download_file(download_url, filename)
    
    if success:
        print(f"✅ Файл {filename} успешно загружен")
        print(f"📊 Размер: {os.path.getsize(filename) / (1024*1024):.1f} MB")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 