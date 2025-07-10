#!/usr/bin/env python3
"""
Тестовые запросы для AI Category Matcher Service
Примеры использования всех API эндпоинтов
"""

import requests
import json
import time
from typing import Dict, Any, List

class AICategoryServiceTester:
    def __init__(self, base_url: str):
        """
        Инициализация тестера
        
        Args:
            base_url: Базовый URL сервиса (например, https://your-domain.railway.app)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Category-Service-Tester/1.0'
        })
    
    def test_health(self) -> Dict[str, Any]:
        """Тест эндпоинта здоровья сервиса"""
        print("🏥 Тестирование /health...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Статус: {data.get('status')}")
            print(f"📊 Модель загружена: {data.get('model_loaded')}")
            print(f"📈 Товаров в базе: {data.get('base_loaded')}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка: {e}")
            return {"error": str(e)}
    
    def test_main_page(self) -> Dict[str, Any]:
        """Тест главной страницы API"""
        print("\n🏠 Тестирование главной страницы...")
        
        try:
            response = self.session.get(self.base_url)
            response.raise_for_status()
            
            data = response.json()
            print(f"✅ Сервис: {data.get('service')}")
            print(f"📋 Версия: {data.get('version')}")
            print(f"📝 Описание: {data.get('description')}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка: {e}")
            return {"error": str(e)}
    
    def test_debug(self) -> Dict[str, Any]:
        """Тест диагностического эндпоинта"""
        print("\n🔍 Тестирование /debug...")
        
        try:
            response = self.session.get(f"{self.base_url}/debug")
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'success':
                debug_info = data.get('data', {})
                print(f"🤖 Модель: {debug_info.get('model_info', {}).get('name')}")
                print(f"📊 Товаров: {debug_info.get('database_info', {}).get('total_products')}")
                print(f"📁 Категорий: {debug_info.get('database_info', {}).get('categories_count')}")
                print(f"🔑 API ключ: {'✅' if debug_info.get('api_info', {}).get('openrouter_key_set') else '❌'}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка: {e}")
            return {"error": str(e)}
    
    def test_search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Тест поиска товаров"""
        print(f"\n🔍 Тестирование поиска: '{query}'...")
        
        try:
            params = {'query': query, 'limit': limit}
            response = self.session.get(f"{self.base_url}/test_search", params=params)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'success':
                results = data.get('data', {}).get('results', [])
                print(f"📈 Найдено товаров: {len(results)}")
                
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.get('name')} ({result.get('category')}) - {result.get('similarity'):.2f}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка: {e}")
            return {"error": str(e)}
    
    def test_categorization(self, name: str, code: str) -> Dict[str, Any]:
        """Тест категоризации товара"""
        print(f"\n🤖 Тестирование категоризации: '{name}'...")
        
        try:
            payload = {
                "name": name,
                "code": code
            }
            
            response = self.session.post(f"{self.base_url}/n8n_process", json=payload)
            response.raise_for_status()
            
            data = response.json()
            if data.get('status') == 'success':
                result = data.get('data', {})
                print(f"✅ Категория: {result.get('Категория')}")
                print(f"📁 Подкатегория: {result.get('Подкатегория')}")
                print(f"🎯 Уверенность: {result.get('Комментарий')}%")
            else:
                print(f"❌ Ошибка: {data.get('message')}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка: {e}")
            return {"error": str(e)}
    
    def test_batch_categorization(self, products: List[Dict[str, str]]) -> Dict[str, Any]:
        """Тест пакетной категоризации"""
        print(f"\n📦 Тестирование пакетной категоризации ({len(products)} товаров)...")
        
        results = []
        for i, product in enumerate(products, 1):
            print(f"  Обработка {i}/{len(products)}: {product['name']}")
            result = self.test_categorization(product['name'], product['code'])
            results.append(result)
            
            # Небольшая пауза между запросами
            if i < len(products):
                time.sleep(0.5)
        
        return {"results": results}
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Запуск полного набора тестов"""
        print("🚀 Запуск полного набора тестов AI Category Service")
        print("=" * 60)
        
        test_results = {}
        
        # 1. Тест здоровья
        test_results['health'] = self.test_health()
        
        # 2. Главная страница
        test_results['main_page'] = self.test_main_page()
        
        # 3. Диагностика
        test_results['debug'] = self.test_debug()
        
        # 4. Тесты поиска
        search_queries = [
            "мотоблок",
            "генератор",
            "инструмент",
            "насос"
        ]
        
        test_results['search'] = {}
        for query in search_queries:
            test_results['search'][query] = self.test_search(query, limit=3)
        
        # 5. Тесты категоризации
        test_products = [
            {"name": "Мотоблок дизельный 7 л.с.", "code": "534143"},
            {"name": "Генератор бензиновый 2.5 кВт", "code": "123456"},
            {"name": "Насос центробежный 1.5 кВт", "code": "789012"},
            {"name": "Перфоратор SDS-plus 800 Вт", "code": "345678"}
        ]
        
        test_results['categorization'] = self.test_batch_categorization(test_products)
        
        print("\n" + "=" * 60)
        print("✅ Тестирование завершено!")
        
        return test_results
    
    def save_test_results(self, results: Dict[str, Any], filename: str = "test_results.json"):
        """Сохранение результатов тестов в файл"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Результаты сохранены в {filename}")
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")


def main():
    """Основная функция для запуска тестов"""
    
    # Настройки
    BASE_URL = "https://your-domain.railway.app"  # Замените на ваш URL
    
    print("🤖 AI Category Service - Тестовый скрипт")
    print("=" * 50)
    
    # Создание тестера
    tester = AICategoryServiceTester(BASE_URL)
    
    # Запуск тестов
    results = tester.run_full_test_suite()
    
    # Сохранение результатов
    tester.save_test_results(results)
    
    # Краткая сводка
    print("\n📊 Краткая сводка:")
    print(f"  🏥 Health: {'✅' if results.get('health', {}).get('status') == 'healthy' else '❌'}")
    print(f"  🏠 Main Page: {'✅' if 'service' in results.get('main_page', {}) else '❌'}")
    print(f"  🔍 Debug: {'✅' if results.get('debug', {}).get('status') == 'success' else '❌'}")
    
    # Подсчет успешных категоризаций
    categorization_results = results.get('categorization', {}).get('results', [])
    successful_categorizations = sum(
        1 for r in categorization_results 
        if r.get('status') == 'success'
    )
    print(f"  🤖 Категоризации: {successful_categorizations}/{len(categorization_results)} ✅")


if __name__ == "__main__":
    main() 