import os
import sys
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests
from tqdm import tqdm
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Any, Optional

# --- Логгер ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("log_output.txt")

# === НАСТРОЙКИ ===
MODEL_ID = "deepseek/deepseek-chat-v3-0324:free"
OPENROUTER_API_KEY = "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"
SIMILARITY_THRESHOLD = 0.85  # Повышенный порог для более точного отбора

print("🔄 Загружаем модель SentenceTransformer...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Используем переменные окружения для путей
BASE_PKL_PATH = os.getenv("BASE_PKL_PATH", "products_with_embeddings.pkl")

def download_embeddings_if_needed():
    """Загружает файл эмбеддингов из внешнего источника если нужно"""
    if os.path.exists(BASE_PKL_PATH):
        return True
    
    download_url = os.getenv("EMBEDDINGS_DOWNLOAD_URL")
    if not download_url:
        return False
    
    try:
        print(f"📥 Загружаем файл эмбеддингов из {download_url}")
        
        # Для Google Drive используем специальные заголовки
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(download_url, stream=True, headers=headers)
        response.raise_for_status()
        
        # Проверяем, что это не HTML страница
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type or response.text[:100].strip().startswith('<'):
            print("❌ Получена HTML страница вместо файла. Google Drive требует авторизации.")
            return False
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(BASE_PKL_PATH, 'wb') as file, tqdm(
            desc="Загрузка эмбеддингов",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        
        # Проверяем, что файл действительно загружен
        if os.path.getsize(BASE_PKL_PATH) < 1000:  # Меньше 1KB - скорее всего HTML
            print("❌ Загруженный файл слишком маленький, возможно это HTML страница")
            os.remove(BASE_PKL_PATH)
            return False
        
        print(f"✅ Файл эмбеддингов загружен: {os.path.getsize(BASE_PKL_PATH) / (1024*1024):.1f} MB")
        return True
    except Exception as e:
        print(f"❌ Ошибка загрузки файла эмбеддингов: {e}")
        if os.path.exists(BASE_PKL_PATH):
            os.remove(BASE_PKL_PATH)
        return False

print("📁 Загружаем базу категорий и эмбеддинги...")

# Пытаемся загрузить файл из внешнего источника если его нет
if not os.path.exists(BASE_PKL_PATH):
    download_embeddings_if_needed()

if os.path.exists(BASE_PKL_PATH):
    df_base: pd.DataFrame = pd.read_pickle(BASE_PKL_PATH)
    print(f"✅ Загружено {len(df_base)} строк с готовыми эмбеддингами.")
else:
    print(f"⚠️ Файл эмбеддингов {BASE_PKL_PATH} не найден. Создаем пустую базу данных.")
    # Создаем пустую базу данных с правильной структурой
    df_base = pd.DataFrame({
        'sku': [],
        'name': [],
        'category': [],
        'emb': []
    })
    print("✅ Создана пустая база данных. Сервис готов к работе, но категоризация будет ограничена.")

# === ЗАПРОС В OpenRouter ===
def query_openrouter(prompt: str, model: str = MODEL_ID) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yourdomain.com",
        "X-Title": "CategoryMatcher"
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Ты — помощник, который определяет категорию товара по его названию."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)

    try:
        data = response.json()
    except Exception as e:
        print("❌ Не удалось распарсить JSON:", str(e))
        print("🔁 Response content:", response.text)
        return None

    if response.status_code != 200:
        print("⚠️ Ошибка от OpenRouter:", data.get("error", "Неизвестная ошибка"))
        return None

    if "choices" not in data:
        print("⚠️ В ответе отсутствует ключ 'choices':", data)
        return None

    return data["choices"][0]["message"]["content"].strip()

# === API Функції ===
app = FastAPI(title="AI Category Matcher", description="API для автоматической категоризации товаров")

@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "AI Category Matcher API",
        "version": "1.0.0",
        "endpoints": {
            "/process": "POST - Обработка Excel файла",
            "/n8n_process": "POST - Специальный эндпоинт для n8n",
            "/health": "GET - Проверка здоровья сервиса"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "healthy", "model_loaded": True, "base_loaded": len(df_base) > 0}

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    """Обробляє Excel файл через API"""
    try:
        # Читаємо вхідний файл
        df_new = pd.read_excel(await file.read())
        results = process_products(df_new)
        
        # Повертаємо результат у JSON
        return JSONResponse({
            "status": "success",
            "data": results,
            "processed_items": len(results)
        })
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=400
        )

@app.post("/n8n_process")
async def n8n_process(request: Request):
    """Спеціальний ендпоінт для n8n"""
    try:
        # Получаем сырые данные и парсим вручную
        raw_data = await request.json()
        print("📦 Получены данные:", raw_data)
        
        # Конвертуємо вхідні дані у DataFrame
        processed_data = {
            "Наименование": raw_data.get("name") or raw_data.get("Наименование") or raw_data.get("title") or "",
            "Артикул": raw_data.get("code") or raw_data.get("Артикул") or raw_data.get("sku") or ""
        }
        
        print("🔍 Обработанные данные:", processed_data)
        
        if not processed_data["Наименование"]:
            return JSONResponse(
                {"status": "error", "message": "Поле 'Наименование' обязательно"},
                status_code=400
            )
            
        df_new = pd.DataFrame([processed_data])
        results = process_products(df_new)
        
        return JSONResponse({
            "status": "success",
            "data": results[0] if results else {},
            "processed_items": len(results)
        })
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=400
        )

def process_products(df_new: pd.DataFrame) -> List[Dict[str, Any]]:
    """Основна логіка обробки товарів"""
    global df_base, model
    results = []
    try:
        records = df_new.to_dict(orient="records")
        if not records:
            raise ValueError("Пустой DataFrame")
        for idx, row in enumerate(tqdm(records, desc="Обработка")):
            name = str(row.get("Наименование", "")).strip()
            art = str(row.get("Артикул", "")).strip()
            if not name:
                results.append({
                    "status": "error",
                    "message": "Поле 'Наименование' обязательно",
                    "data": None
                })
                continue
            print(f"\n🆕 [{idx+1}/{len(records)}] Обрабатываем товар: {name}")
            # Фільтрація по першим 2-3 словам
            product_words = name.lower().split()[:3]
            product_type = " ".join(product_words)
            escaped_words = [re.escape(word) for word in product_words]
            # Приведение колонки name к строке для избежания ошибок Pandas
            df_base["name"] = df_base["name"].astype(str)
            mask = df_base["name"].str.lower().str.contains('|'.join(escaped_words), regex=True)
            df_filtered: pd.DataFrame = df_base[mask].copy()
            if df_filtered.empty or len(df_filtered) < 5:
                print(f"⚠️ Слишком мало совпадений по типу товара '{product_type}', используем всю базу.")
                df_filtered = df_base.copy()
            else:
                print(f"🔍 Найдено {len(df_filtered)} потенциальных совпадений по типу '{product_type}'")
            # Обчислення схожості
            emb = model.encode(name, convert_to_tensor=True)
            df_filtered["similarity"] = df_filtered["emb"].apply(lambda x: util.cos_sim(emb, x).item())
            df_filtered = df_filtered.sort_values("similarity", ascending=False)
            # Логика выбора категории
            top_matches = df_filtered[df_filtered["similarity"] >= SIMILARITY_THRESHOLD]
            # --- Категоризация по эмбеддингам (как было)
            cat_ai, confidence, sku_ai = "", "", ""
            if top_matches.empty:
                lower_threshold = 0.75
                top_matches = df_filtered[df_filtered["similarity"] >= lower_threshold]
                print(f"⚠️ Нет совпадений выше порога {SIMILARITY_THRESHOLD:.2f}, пробуем порог {lower_threshold:.2f}")
                if top_matches.empty:
                    lower_threshold = 0.65
                    top_matches = df_filtered[df_filtered["similarity"] >= lower_threshold]
                    print(f"⚠️ Нет совпадений, пробуем порог {lower_threshold:.2f}")
                    if top_matches.empty:
                        top_matches = df_filtered.head(1)
                        print(f"⚠️ Нет совпадений даже при пороге {lower_threshold:.2f}, берем лучший вариант")
                        if df_filtered.iloc[0]["similarity"] < 0.5:
                            cat_ai = "Пропустить"
                            confidence = "низкое качество совпадения"
                            sku_ai = ""
                            print("Категория: Пропустить (качество совпадения < 0.5)")
            else:
                top_variants = "\n".join([
                    f"{i+1}. {match['category']} ({match['name']}) — {match['similarity']*100:.1f}% [SKU: {match['sku']}]"
                    for i, match in enumerate(top_matches.to_dict(orient="records"))
                ])
                print("🔍 Топ совпадений:")
                print(top_variants)
                # Берём первую наиболее похожую категорию
                best = top_matches.iloc[0]
                cat_ai = best["category"]
                confidence = f"{best['similarity']*100:.1f}"
                sku_ai = best["sku"]

            # --- Всегда делаем запрос к AI ---
            print("\n🤖 Запрашиваем категорию у AI...")
            # Формируем структурированный блок категорий для prompt
            category_chain = cat_ai if cat_ai else ""
            category_structured = ""
            if category_chain:
                parts = [c.strip() for c in category_chain.split('>')]
                for i, part in enumerate(parts):
                    exclam = '!' * i
                    category_structured += f"category{exclam}: {part}\n"
                category_chain_block = f"\nСтруктурированная цепочка категорий для товара:\n{category_structured}\n"
            else:
                category_chain_block = ""
            # Формируем топ вариантов для prompt (устойчиво к отсутствию top_variants)
            top_variants_str = top_variants if 'top_variants' in locals() else ''
            # Формируем prompt
            ai_prompt = f'''Ты — эксперт по технике и запчастям с 20-летним опытом. Твоя задача — максимально точно определить категорию товара по его названию.\n\nПравила анализа:\n1. Внимательно изучи название товара: \"{name}\"\n2. Определи основной тип техники (например: мотоблок, генератор, скутер и т.д.)\n3. Проанализируй все варианты ниже и выбери наиболее точное соответствие\n4. Если сомневаешься (уверенность < 95%) — укажи \"Пропустить\"\n5. Для точных совпадений (уверенность ≥ 99.9%) укажи точный процент\n\nДополнительные указания:\n- Учитывай специфику товара (запчасть, аксессуар, комплект и т.д.)\n- Проверь соответствие бренда и модели\n- Игнорируй общие категории в пользу более специфичных\n{category_chain_block}\nТоп вариантов:\n{top_variants_str}\n\nФормат ответа (строго соблюдай):\ncategory: [уровень 1]\ncategory!: [уровень 2]\ncategory!!: [уровень 3]\n...\nКомментарий: [только число 50-100, % уверенности]\n\nПример правильного ответа:\ncategory: Мотозапчасти\ncategory!: Амортизаторы и подвеска\ncategory!!: Амортизаторы задние\nКомментарий: 99\n'''
            ai_category = query_openrouter(ai_prompt)
            print(f"AI ответ: {ai_category}")

            results.append({
                "Артикул": art,
                "Наименование": name,
                "Категория": cat_ai,
                "SKU": sku_ai,
                "Комментарий": confidence,
                "AI_Категория": ai_category
            })
    except Exception as e:
        print(f"Ошибка в process_products: {str(e)}")
        raise
    return results

if __name__ == "__main__":
    # Получаем порт из переменных окружения (для облачных платформ)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")  # 0.0.0.0 для облачных платформ
    
    print("🚀 Запускаем AI Category Matcher API...")
    print(f"📡 API будет доступен по адресу: http://{host}:{port}")
    print(f"🔗 Эндпоинт для n8n: http://{host}:{port}/n8n_process")
    print(f"🏥 Проверка здоровья: http://{host}:{port}/health")
    
    # Отключаем стандартное логгирование Uvicorn
    uvicorn.run(app, host=host, port=port, log_config=None)
