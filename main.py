import json
import time
import requests
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Dict
import os
import sys

# Settings for translation
TRANSLATION_SETTINGS = {
    "model": "gemma2:9b",  # LLM model to use
    "target_language": "zh-Hant-TW",  # Target language: pl,se,de,fr,es,zh-Hant-TW
    "domain": "APP Mesh Network Wireless Communication",  # Professional domain
    "style": "easy to understand",  # Translation style
    "retry_count": 3,  # Number of retries
    "min_score": 60,  # Minimum acceptable score
    "prompt_version": "en",  # Prompt version (zh or en)
}

# Load prompts from JSON file
def load_prompts():
    try:
        with open('prompts.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading prompts: {e}")
        sys.exit(1)

PROMPTS = load_prompts()

# 終端機色彩常數
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class TranslationMemory:
    def __init__(self, max_examples: int = 5):
        self.memory: Dict[str, str] = {}
        self.memory_file = "translation_memory.json"
        self.max_examples = max_examples
        self.load_memory()
    
    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
        except Exception as e:
            print(f"載入翻譯記憶時發生錯誤: {e}")
            self.memory = {}
    
    def save_memory(self):
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"儲存翻譯記憶時發生錯誤: {e}")
    
    def get_translation(self, text: str) -> Optional[str]:
        return self.memory.get(text)
    
    def add_translation(self, source: str, target: str):
        self.memory[source] = target
        self.save_memory()
    
    def get_similar_translations(self, text: str) -> list:
        """獲取與當前文本相似的翻譯記錄"""
        return find_similar_entries(text, self.memory, self.max_examples)

def log_translation(original, old_translation, new_translation, score=None):
    """Log translation results to log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""[{timestamp}]
Source: {original}
{f'Previous translation: {old_translation}' if old_translation else ''}
New translation: {new_translation}
{f'Score: {score}' if score else ''}
{'=' * 50}
"""
    with open('translation_log.txt', 'a', encoding='utf-8') as f:
        f.write(log_entry)

def call_ollama(prompt, model=TRANSLATION_SETTINGS["model"], max_retries=TRANSLATION_SETTINGS["retry_count"]):
    for attempt in range(max_retries):
        try:
            response = requests.post('http://localhost:11434/api/generate',
                json={
                    "model": model,
                    "prompt": f"{prompt}",
                    "stream": False
                }, timeout=30)  # 添加超時設定
            
            if response.status_code == 200:
                data = response.json()
                if 'response' in data:
                    return data['response']
            
            # 如果沒有得到有效回應，等待後重試
            time.sleep(2 * (attempt + 1))
            continue
            
        except Exception as e:
            print(f"嘗試 {attempt + 1}/{max_retries} 失敗: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return "翻譯服務暫時無法使用，請稍後再試"
    
    return "無法獲取翻譯結果"

def validate_special_chars(original: str, translation: str) -> bool:
    """驗證特殊符號是否正確保留"""
    # 需要檢查的特殊格式
    patterns = [
        r'%@', r'%d', r'%s', r'%lld', r'%i', r'%f', r'%u', r'%x',  # 基本格式
        r'%\d+\$@', r'%\d+\$d', r'%\d+\$s',  # 帶數字的格式
        r'\${.*?}',  # 變數名稱格式
        r'%[\d.]*lld%%',  # 百分比格式
        r'[:\-\.,!\?]'  # 標點符號
    ]
    
    def count_pattern(text: str, pattern: str) -> int:
        import re
        return len(re.findall(pattern, text))
    
    # 檢查每個模式在原文和翻譯中的出現次數是否相同
    for pattern in patterns:
        if count_pattern(original, pattern) != count_pattern(translation, pattern):
            return False
    
    # 檢查空格保留
    original_spaces = [i for i, char in enumerate(original) if char == ' ']
    translation_spaces = [i for i, char in enumerate(translation) if char == ' ']
    
    # 如果原文只包含特殊格式，翻譯必須完全相同
    if all(c in '%$@{}[]()' for c in original.replace(' ', '')):
        return original == translation
        
    return True

def evaluate_translation(original, translation, max_retries=TRANSLATION_SETTINGS["retry_count"], is_double_check=False):
    """Evaluate translation quality with strict scoring"""
    # First check special characters
    if not validate_special_chars(original, translation):
        print(f"{Colors.WARNING}>> Format validation failed{Colors.ENDC}")
        return 0
        
    prompt_template = PROMPTS[TRANSLATION_SETTINGS["prompt_version"]]["evaluation"]
    prompt = prompt_template.format(original=original, translation=translation)
    
    for attempt in range(max_retries):
        response = call_ollama(prompt)
        if response and response != "Translation service unavailable" and response != "Failed to get translation":
            try:
                score = ''.join(filter(str.isdigit, response))
                score = int(score) if score else 50
                
                # Add score validation only if not already double checking
                if score >= 95 and not is_double_check:
                    print(f"{Colors.WARNING}>> High score detected ({score}), verifying...{Colors.ENDC}")
                    # Do a second check with only one retry
                    second_score = evaluate_translation(
                        original, 
                        translation, 
                        max_retries=1, 
                        is_double_check=True
                    )
                    print(f"{Colors.BLUE}>> Verification score: {second_score}{Colors.ENDC}")
                    # Take the average of both scores
                    final_score = (score + second_score) // 2
                    print(f"{Colors.BLUE}>> Final score: {final_score}{Colors.ENDC}")
                    return final_score
                    
                return score
            except:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
    return 50

def evaluate_both_translations(key, en_text, translation):
    """評估 key 和 en_text 的翻譯結果"""
    key_score = evaluate_translation(key, translation)
    en_score = evaluate_translation(en_text, translation) if en_text and en_text != key else 0
    
    # 如果都有分數，返回較高的分數
    if en_score > 0:
        return max(key_score, en_score)
    return key_score

class TranslationContext:
    def __init__(self, domain: str = "一般", style: str = "正式"):
        self.domain = domain
        self.style = style
    
    def get_context(self) -> str:
        return f"""
領域：{self.domain}
風格：{self.style}
"""

def translate_with_ollama(text: str, context: TranslationContext, memory: TranslationMemory):
    """直接使用記憶翻譯或處理特殊格式變數"""
    # 檢查記憶和純變數文本
    cached_translation = memory.get_translation(text)
    if cached_translation:
        return cached_translation, True
    
    if all(c in '%$@{}[]() ' for c in text):
        memory.add_translation(text, text)
        return text, True
        
    # 使用選擇的提示詞版本，添加更強的指令
    prompt = f"""INSTRUCTION: Translate this text from English to {TRANSLATION_SETTINGS["target_language"]}.
IMPORTANT: Output ONLY the translation, no explanations or additional text.

Source text: {text}

RULES:
1. Return ONLY the translated text
2. Maintain all format variables (%@, %d, etc.) exactly as-is
3. Keep all spaces and punctuation
4. Do not add any explanations or notes
5. Do not repeat the source text

Translation:"""

    translation = call_ollama(prompt).strip()
    
    # 清理回應，只保留實際翻譯內容
    if translation:
        # 移除可能的前綴説明
        if ":" in translation:
            translation = translation.split(":")[-1].strip()
        # 移除引號如果存在
        translation = translation.strip('"').strip("'")
        
        if translation not in ["翻譯服務暫時無法使用，請稍後再試", "無法獲取翻譯結果"]:
            memory.add_translation(text, translation)
            
    return translation, False

def format_memory_context(text: str, memory: TranslationMemory) -> str:
    """格式化相關的翻譯記憶作為上下文"""
    similar_entries = memory.get_similar_translations(text)
    if not similar_entries:
        return "無相關翻譯記錄"
    
    context = "相關翻譯記錄：\n"
    for source, target in similar_entries:
        context += f"原文：{source}\n翻譯：{target}\n"
    return context

def find_similar_entries(text: str, memory: Dict[str, str], max_entries: int) -> list:
    """找出相似的翻譯記錄"""
    from difflib import SequenceMatcher
    
    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    # 計算所有記憶項目與當前文本的相似度
    similar_items = [
        (source, target, similarity(text, source))
        for source, target in memory.items()
    ]
    
    # 排序並返回最相似的幾個
    similar_items.sort(key=lambda x: x[2], reverse=True)
    return [(source, target) for source, target, _ in similar_items[:max_entries]]

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def save_translations(data, original_file, model="gemma2:27b"):
    # 從模型名稱中移除非法字元
    safe_model_name = "".join(c for c in model if c.isalnum() or c in ":-_")
    
    # 生成帶時間戳和模型名稱的新檔名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.splitext(original_file)[0]
    new_file = f"{file_name}_translated_{safe_model_name}_{timestamp}.xcstrings"
    
    # 保存翻譯後的完整數據
    with open(new_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return new_file

def create_translation_unit(translation: str) -> dict:
    """創建標準的翻譯單元結構"""
    return {
        "stringUnit": {
            "state": "translated",  # 必須包含狀態
            "value": translation    # 翻譯內容
        }
    }

def process_json_value(value: dict, memory: TranslationMemory, context: TranslationContext) -> dict:
    """Process JSON value recursively to translate all string values while maintaining structure"""
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k == "variations":
                # Handle variations structure
                print(f"{Colors.BLUE}>> Processing variations structure{Colors.ENDC}")
                result[k] = {
                    variation_type: {
                        form: process_json_value(form_data, memory, context)
                        for form, form_data in variation_data.items()
                    }
                    for variation_type, variation_data in v.items()
                }
            elif isinstance(v, dict):
                result[k] = process_json_value(v, memory, context)
            elif k == "value" and isinstance(v, str):
                # Only translate the "value" field
                print(f"{Colors.BLUE}>> Source text:{Colors.ENDC} {v}")
                translation, is_cached = translate_with_ollama(v, context, memory)
                
                if not is_cached:
                    # Only evaluate if not from cache
                    score = evaluate_translation(v, translation)
                    print(f"{Colors.GREEN}>> Translation:{Colors.ENDC} {translation}")
                    print(f"{Colors.BLUE}>> Quality score:{Colors.ENDC} {score}")
                else:
                    print(f"{Colors.GREEN}>> Using cached translation:{Colors.ENDC} {translation}")
                    
                result[k] = translation
            else:
                result[k] = v
        return result
    return value

def translate_variations(variations: dict, memory: TranslationMemory, context: TranslationContext) -> dict:
    """Specifically handle plural variations"""
    result = {}
    for variation_type, forms in variations.items():
        if variation_type == "plural":
            print(f"{Colors.BLUE}>> Processing plural variation{Colors.ENDC}")
            result[variation_type] = {}
            for form, content in forms.items():
                if "stringUnit" in content:
                    string_unit = content["stringUnit"]
                    if "value" in string_unit:
                        source_text = string_unit["value"]
                        print(f"{Colors.BLUE}>> Source text ({form}):{Colors.ENDC} {source_text}")
                        
                        translation, is_cached = translate_with_ollama(source_text, context, memory)
                        
                        if not is_cached:
                            score = evaluate_translation(source_text, translation)
                            print(f"{Colors.GREEN}>> Translation ({form}):{Colors.ENDC} {translation}")
                            print(f"{Colors.BLUE}>> Quality score ({form}):{Colors.ENDC} {score}")
                        else:
                            print(f"{Colors.GREEN}>> Using cached translation ({form}):{Colors.ENDC} {translation}")
                            
                        result[variation_type][form] = {
                            "stringUnit": {
                                "state": "translated",
                                "value": translation
                            }
                        }
                    else:
                        result[variation_type][form] = content
                else:
                    result[variation_type][form] = content
        else:
            result[variation_type] = forms
    return result

def print_strings(file_path, context: Optional[TranslationContext] = None):
    """Process localization strings and add target language if missing"""
    if context is None:
        context = TranslationContext(
            domain=TRANSLATION_SETTINGS["domain"],
            style=TRANSLATION_SETTINGS["style"]
        )
    
    memory = TranslationMemory()
    target_lang = TRANSLATION_SETTINGS["target_language"]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    strings = data.get('strings', {})
    total = len(strings)
    processed = 0
    
    for key, value in strings.items():
        try:
            processed += 1
            print(f"\nProcessing: {processed}/{total}")
            
            # Find source text to translate
            source_text = key  # Default to key if no English text found
            
            if isinstance(value, dict) and "localizations" in value:
                localizations = value["localizations"]
                
                # Try to get English text as source
                if "en" in localizations and "stringUnit" in localizations["en"]:
                    source_text = localizations["en"]["stringUnit"].get("value", key)
                
                # Check if target language exists
                if target_lang not in localizations:
                    # Create new translation
                    print(f"{Colors.BLUE}>> Adding new translation for {key}{Colors.ENDC}")
                    print(f"{Colors.BLUE}Source text:{Colors.ENDC} {source_text}")
                    
                    translation, is_cached = translate_with_ollama(source_text, context, memory)
                    if not is_cached:
                        score = evaluate_translation(source_text, translation)
                        print(f"{Colors.GREEN}>> Translation:{Colors.ENDC} {translation}")
                        print(f"{Colors.BLUE}>> Quality score:{Colors.ENDC} {score}")
                    else:
                        print(f"{Colors.GREEN}>> Using cached translation:{Colors.ENDC} {translation}")
                    
                    # Add new translation with same structure
                    localizations[target_lang] = {
                        "stringUnit": {
                            "state": "translated",
                            "value": translation
                        }
                    }
                    
                elif "variations" in localizations[target_lang]:
                    # Handle variations structure
                    localizations[target_lang]["variations"] = translate_variations(
                        localizations[target_lang]["variations"],
                        memory,
                        context
                    )
                else:
                    # Update existing translation if needed
                    current = localizations[target_lang].get("stringUnit", {}).get("value")
                    if current:
                        score = evaluate_translation(source_text, current)
                        if score < TRANSLATION_SETTINGS["min_score"]:
                            print(f"{Colors.WARNING}>> Low score ({score}), retranslating{Colors.ENDC}")
                            translation, _ = translate_with_ollama(source_text, context, memory)
                            localizations[target_lang]["stringUnit"]["value"] = translation
            
            print("-" * 50)
            
        except Exception as e:
            print(f"{Colors.FAIL}Error processing item: {str(e)}{Colors.ENDC}")
            continue
    
    # Save translation results
    new_file = save_translations(data, file_path, model=TRANSLATION_SETTINGS["model"])
    print(f"\nTranslation results saved to: {new_file}")

def translate_with_improve_loop(key, source_text, current_translation, context, memory, source_type):
    """Handle translation improvement loop"""
    max_attempts = TRANSLATION_SETTINGS["retry_count"]
    current_attempt = 0
    best_translation = current_translation
    best_score = 0

    while current_attempt < max_attempts:
        translation, is_cached = translate_with_ollama(source_text if source_text else key, context, memory)
        
        if is_cached:
            print(f"{Colors.GREEN}>> Using cached translation{Colors.ENDC}")
            return translation
            
        print(f"{Colors.GREEN}>> Translation result:{Colors.ENDC}", translation)
        
        score = evaluate_both_translations(key, source_text, translation) if source_type == "String key" else evaluate_translation(source_text, translation)
        print(f"{Colors.BLUE}>> Quality score (#{current_attempt + 1}):{Colors.ENDC}", score)
        
        if score > best_score:
            best_translation = translation
            best_score = score
            
        if score >= TRANSLATION_SETTINGS["min_score"]:
            break
            
        current_attempt += 1
        print(f"{Colors.WARNING}>> Retrying for better quality...{Colors.ENDC}")
        
    return best_translation

if __name__ == "__main__":
    # 選擇提示詞版本
    version = input("Select prompt version (zh/en, default: en): ").strip().lower()
    if version in ["zh", "en"]:
        TRANSLATION_SETTINGS["prompt_version"] = version
    
    context = TranslationContext(
        domain=TRANSLATION_SETTINGS["domain"],
        style=TRANSLATION_SETTINGS["style"]
    )
    file_path = "Localizable.xcstrings"
    print_strings(file_path, context)
