#!/usr/bin/env python3
"""
Оптимизированный пайплайн обработки изображений с:
- Кэшированием моделей
- Управлением памятью GPU
- Пакетной обработкой
- Подробным логированием
- Интеграцией Llama 3.1 для улучшенного извлечения текста
"""

import os
import torch
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from PIL import Image
from functools import lru_cache
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from gensim.models import KeyedVectors
import gc
from tqdm import tqdm
from llama_cpp import Llama


# ====================== КОНФИГУРАЦИЯ ======================
MODEL_CACHE_SIZE = 3            # Кол-во моделей в кэше LRU
MEMORY_CLEAR_THRESHOLD = 2      # Порог очистки памяти (GB)
IMAGE_MAX_SIZE = 384            # Макс. размер изображения
BATCH_PROCESS_SIZE = 1          # Уменьшенный размер батча (1 изображение за раз)
LOG_FILE = "image_processing.log"  # Файл логов
LLAMA_MODEL_PATH = "Meta-Llama-3.1-8B-Claude-IQ2_M.gguf"  # Путь к модели Llama
# ==========================================================

# Отключаем ограничение на размер изображения
Image.MAX_IMAGE_PIXELS = None

# Настройка окружения для уменьшения потребления памяти
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Инициализация логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """Очистка памяти GPU с подробным логированием"""
    before = torch.cuda.mem_get_info()[0]/1e9
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    after = torch.cuda.mem_get_info()[0]/1e9
    logger.info(f"GPU memory cleared. Before: {before:.2f}GB, After: {after:.2f}GB")

def adaptive_memory_management(min_free_mem=MEMORY_CLEAR_THRESHOLD):
    """
    Адаптивное управление памятью GPU
    Возвращает True если была выполнена очистка
    """
    free_mem = torch.cuda.mem_get_info()[0]/1e9
    if free_mem < min_free_mem:
        logger.warning(f"Low GPU memory ({free_mem:.2f}GB), clearing cache")
        clear_gpu_memory()
        return True
    return False

@lru_cache(maxsize=MODEL_CACHE_SIZE)
def load_models():
    """
    Загрузка и кэширование всех моделей
    Использует LRU-кэш для избежания повторной загрузки
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        logger.info("Loading models...")
        
        # Конфиг квантизации для Qwen2-VL
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        # Загрузка Qwen2-VL модели с оптимизациями
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True
        )
        
        # Загрузка процессора
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            use_fast=True
        )
        
        # Загрузка текстового эмбеддера
        text_embedder = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            device=device
        )
        
        # Загрузка CLIP модели (с fallback логикой)
        clip_model, clip_preprocess = load_clip_model(device)
        
        # Загрузка FastText модели
        fasttext_model = KeyedVectors.load("/mldata/model/model.model")
        
        # Загрузка Llama модели (на CPU)
        llama_model = Llama.from_pretrained(
            repo_id="bartowski/Meta-Llama-3.1-8B-Claude-GGUF",
            filename=LLAMA_MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0  # Полностью на CPU для экономии GPU памяти
        )
        
        logger.info("All models loaded and cached")
        return model, processor, text_embedder, clip_model, clip_preprocess, fasttext_model, llama_model, device
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def load_clip_model(device):
    """Загрузка CLIP модели с fallback-механизмом"""
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        logger.info("Using original CLIP model")
        return model, preprocess
    except (ImportError, AttributeError) as e:
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            model = model.to(device)
            logger.info("Using open_clip as fallback")
            return model, preprocess
        except ImportError:
            logger.error("CLIP libraries not available")
            raise ImportError("Install either clip or open_clip package")

def preprocess_image(image_path, clip_preprocess, device):
    """Оптимизированная обработка изображений"""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            
            # Масштабирование
            width, height = img.size
            if max(width, height) > IMAGE_MAX_SIZE:
                ratio = IMAGE_MAX_SIZE / max(width, height)
                new_size = (int(width*ratio), int(height*ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            image_tensor = clip_preprocess(img).unsqueeze(0).to(device).float()
            return image_tensor
    except Exception as e:
        logger.error(f"Image processing failed for {image_path}: {str(e)}")
        raise

def text_to_embedding(text, embedder, model_type='sentence'):
    """Универсальная функция векторизации текста"""
    try:
        if model_type == 'sentence':
            return embedder.encode(text, convert_to_tensor=True).cpu().numpy()
        elif model_type == 'fasttext':
            words = text.split()
            vectors = [embedder[word] for word in words if word in embedder]
            return np.mean(vectors, axis=0) if vectors else np.zeros(embedder.vector_size)
    except Exception as e:
        logger.error(f"Text embedding failed: {str(e)}")
        return np.zeros(embedder.vector_size if model_type == 'fasttext' else 384)

def extract_text_with_llama(initial_description, llama_model):
    """
    Улучшение извлечения текста с помощью Llama 3.1
    Возвращает структурированный JSON с извлеченной информацией
    """
    try:
        prompt = f"""
        Проанализируй следующее описание изображения и извлеки информацию в структурированном виде:
        
        Описание: {initial_description}
        
        Извлеки:
        1. Основное описание изображения (полное предложение)
        2. Категорию изображения (1-3 слова)
        3. Ключевые слова и фразы (5-10 штук, через запятую)
        4. Текст, присутствующий на изображении (если есть)
        
        Верни ответ в формате JSON:
        {{
            "description": "...",
            "category": "...",
            "keywords": ["...", "..."],
            "image_text": "..."
        }}
        """
        
        response = llama_model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512
        )
        
        # Извлекаем JSON из ответа
        result = response['choices'][0]['message']['content']
        
        # Очистка ответа (иногда Llama добавляет лишний текст)
        try:
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            cleaned_result = result[json_start:json_end]
            return eval(cleaned_result)
        except:
            logger.warning("Failed to parse Llama response as JSON, returning raw text")
            return {"raw_response": result}
            
    except Exception as e:
        logger.error(f"Error in Llama text extraction: {str(e)}")
        return {"error": str(e)}

def process_single_image(image_path, models):
    """
    Обработка одного изображения с оптимизациями памяти
    Теперь с улучшенным извлечением текста через Llama 3.1
    """
    model, processor, text_embedder, clip_model, clip_preprocess, word2vec_model, llama_model, device = models
    
    try:
        logger.info(f"Processing {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            logger.error(f"File not found: {image_path}")
            return None

        adaptive_memory_management()
        
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                
                width, height = img.size
                if max(width, height) > IMAGE_MAX_SIZE:
                    ratio = IMAGE_MAX_SIZE / max(width, height)
                    new_size = (int(width*ratio), int(height*ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                
                temp_path = f"/tmp/{os.path.basename(image_path)}"
                img.save(temp_path, quality=85, optimize=True)
    
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {str(e)}", exc_info=True)
            return {
                "image_path": image_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": temp_path},
                {"type": "text", "text": "Опиши, что изображено на картинке, на русском языке. Присвой категорию. Определи ключевые слова и фразы. Выведи полностью текст с изображения в виде 'На картинке изображено: . Категория: . Ключевые слова и фразы: . Текст с изображения: '"},
            ],
        }]

        with torch.no_grad():
            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                inputs = processor(
                    text=[text],
                    images=[Image.open(temp_path)],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)
                
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=1
                )
                
                output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Улучшенное извлечение текста с помощью Llama
                extracted_data = extract_text_with_llama(output_text, llama_model)
                
                # Векторизация текста
                text_vector = text_to_embedding(output_text, text_embedder, 'sentence')
                word2vec_vector = text_to_embedding(output_text, word2vec_model, 'fasttext')
                
                # Обработка изображения CLIP
                image_tensor = preprocess_image(temp_path, clip_preprocess, device)
                
                if hasattr(clip_model, 'encode_image'):
                    if image_tensor.dtype != torch.float32:
                        image_tensor = image_tensor.float()
                    image_vector = clip_model.encode_image(image_tensor).cpu().numpy().flatten()
                else:
                    if image_tensor.dtype != torch.float32:
                        image_tensor = image_tensor.float()
                    image_vector = clip_model(image_tensor).cpu().numpy().flatten()
                
                # Формируем результат с новыми полями
                result = {
                    "image_path": image_path,
                    "initial_text": output_text.strip(),
                    "text_vector": text_vector.tolist(),
                    "word2vec_vector": word2vec_vector.tolist(),
                    "image_vector": image_vector.tolist(),
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Добавляем извлеченные данные
                result.update({
                    "llama_extracted": extracted_data,
                    "description": extracted_data.get("description", ""),
                    "category": extracted_data.get("category", ""),
                    "keywords": ", ".join(extracted_data.get("keywords", [])),
                    "image_text": extracted_data.get("image_text", "")
                })
                
                return result
                
            except torch.cuda.OutOfMemoryError:
                logger.error(f"Out of memory processing {image_path}, clearing cache")
                clear_gpu_memory()
                return None
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}", exc_info=True)
                return None
            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass
                torch.cuda.empty_cache()
                
    except Exception as e:
        logger.error(f"Failed to process {image_path}: {str(e)}", exc_info=True)
        return None
    
def process_image_batch(image_paths, output_file, skip_processed=True):
    """
    Пакетная обработка изображений с новыми полями
    """
    try:
        # Проверка/создание выходного файла
        if not os.path.exists(output_file):
            columns = [
                "image_path", "initial_text", "description", "category", 
                "keywords", "image_text", "text_vector", 
                "word2vec_vector", "image_vector", "timestamp", 
                "error", "llama_extracted"
            ]
            pd.DataFrame(columns=columns).to_csv(output_file, index=False)
        else:
            try:
                processed = set(pd.read_csv(output_file, usecols=["image_path"])["image_path"].values)
                image_paths = [p for p in image_paths if p not in processed]
                logger.info(f"Found {len(processed)} already processed images, {len(image_paths)} remaining")
            except Exception as e:
                logger.warning(f"Could not read processed files: {str(e)}")
                if skip_processed:
                    logger.warning("Continuing without skipping processed images due to read error")

        models = load_models()
        
        logger.info(f"Starting processing of {len(image_paths)} images")
        start_time = datetime.now()
        
        results = []
        processed_count = 0
        success_count = 0
        error_count = 0
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            result = process_single_image(img_path, models)
            if result:
                results.append(result)
                success_count += 1
            else:
                error_count += 1
                
            processed_count += 1
            
            if processed_count % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Processed {processed_count}/{len(image_paths)} images "
                    f"({processed_count/elapsed:.2f} img/s)"
                )
            
            if len(results) >= 10:
                pd.DataFrame(results).to_csv(
                    output_file,
                    mode='a',
                    header=not os.path.exists(output_file) or os.stat(output_file).st_size == 0,
                    index=False
                )
                results = []
            
            adaptive_memory_management()

        if results:
            pd.DataFrame(results).to_csv(
                output_file,
                mode='a',
                header=not os.path.exists(output_file) or os.stat(output_file).st_size == 0,
                index=False
            )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Processing complete. Saved to {output_file}\n"
            f"Total images: {len(image_paths)}\n"
            f"Successfully processed: {success_count}\n"
            f"Failed: {error_count}\n"
            f"Skipped: {len(image_paths) - processed_count}\n"
            f"Total time: {elapsed:.2f} seconds\n"
            f"Processing speed: {processed_count/elapsed:.2f} images/second"
        )
        
        return True
    
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}", exc_info=True)
        return False
    finally:
        clear_gpu_memory()

def main():
    """Точка входа с обработкой ошибок"""
    try:
        input_folder = "/mldata/russ_dataset/russ2024y+russ2500_24_03_2025/data"
        output_file = "/mldata/data/processed/new_results_optimized_final.csv"
        
        image_paths = [
            os.path.join(input_folder, f) 
            for f in sorted(os.listdir(input_folder))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        
        logger.info(f"Found {len(image_paths)} images to process in {input_folder}")
        
        start_time = datetime.now()
        success = process_image_batch(image_paths, output_file)
        
        if success:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Processing completed successfully in {elapsed:.2f} seconds "
                f"({len(image_paths)/elapsed:.2f} images per second)"
            )
        else:
            logger.error("Processing failed for some images")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1
    finally:
        clear_gpu_memory()

if __name__ == "__main__":
    exit(main())
