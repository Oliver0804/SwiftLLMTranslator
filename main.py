import json
import time
import requests
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Dict
import os
import sys

# 翻譯設定
TRANSLATION_SETTINGS = {
    "model": "llama3.2:3b",  # 使用的 LLM 模型
    "target_language": "zh-Hant-TW",  # 目標語言
    "domain": "APP Mesh網路 無線通訊 領域",  # 專業領域
    "style": "容易了解",  # 翻譯風格
    "retry_count": 3,  # 重試次數
    "min_score": 60,  # 最低接受分數
    "prompt_version": "zh",  # 提示詞版本 (zh 或 en)
}

# 提示詞模板
PROMPTS = {
    "zh": {
        "translation": """您是一個專業的翻譯助手，請將英文翻譯成繁體中文。

⚠️ 核心規則：變數格式處理規則（違反直接取消輸出）
1. 格式完全保留：
   - 基本: %@、%d、%s、%lld、%i、%f、%u、%x 
   - 數字: %1$@、%2$@、%3$@ (必須保持數字)
   - 百分比: %%、%lld%% (雙%)
   - 變數名: ${name}

2. 變數完整性：
   - 所有變數【完全】照原樣複製
   - 空格數量和位置必須一致
   - 標點符號保持原樣
   - 禁止改變順序
   - 禁止拆分或插入任何字元

3. 特殊情況：
   - 純變數文本需完全相同
   - 連接符號(如"-")保持原樣
   - 阿拉伯數字不轉中文數字

相關記憶：
{memory_context}

{translation_context}

原文：{text}

請直接輸出翻譯（禁止任何說明）：""",

        "evaluation": """請嚴格按照規則為以下翻譯評分(0-100)：
原文：{original}
翻譯：{translation}

⚠️ 格式驗證規則（違反任一項直接給0分）：
1. 變數格式必須完全一致：
   - 基本格式：%@、%d、%s、%lld、%i、%f、%u、%x 等
   - 帶數字格式：%1$@、%2$@、%3$@ 等
   - 百分比格式：%lld%% (保持雙%)
2. 變數完整性：
   - 變數內部不得插入任何字元
   - 變數順序不得調換
   - 變數前後空格必須保持原樣
3. 純變數文本：
   - 若原文僅包含變數（如 "%@" 或 "%1$@ - %2$@"），翻譯必須完全相同

評分標準（總分100分）：
1. 專業術語翻譯正確性 (25分)
2. 繁體中文用字正確 (25分)
3. 語意表達清晰度 (25分)
4. 格式符號處理正確性 (25分)

請只輸出分數："""
    },
    
    "en": {
        "translation": """You are a professional translation assistant. Please translate English to Traditional Chinese.

⚠️ CRITICAL: Variable Format Rules (Violation results in rejection)
1. Preserve formats exactly:
   - Basic: %@, %d, %s, %lld, %i, %f, %u, %x 
   - Numbered: %1$@, %2$@, %3$@ (must keep numbers)
   - Percentage: %%, %lld%% (double %)
   - Variables: ${name}

2. Variable Integrity:
   - Copy ALL variables EXACTLY as-is
   - Maintain exact spacing
   - Keep original punctuation
   - DO NOT change order
   - NO splitting or inserting within variables

3. Special Cases:
   - Pure variable text must be identical
   - Keep connectors (e.g. "-") unchanged
   - Keep Arabic numerals, no conversion

Translation Memory:
{memory_context}

{translation_context}

Source: {text}

Output translation only (no explanations):""",

        "evaluation": """Score this translation strictly (0-100):
Source: {original}
Translation: {translation}

⚠️ Format Rules (Any violation = 0 points):
1. Variable formats must match exactly:
   - Basic: %@, %d, %s, %lld, %i, %f, %u, %x etc.
   - Numbered: %1$@, %2$@, %3$@ etc.
   - Percentage: %lld%% (keep double %)
2. Variable Integrity:
   - No insertions within variables
   - No reordering of variables
   - Maintain exact spacing around variables
3. Pure Variable Text:
   - Must be identical if source is variables only (e.g. "%@" or "%1$@ - %2$@")

Scoring Criteria (Total 100):
1. Technical Term Accuracy (25)
2. Traditional Chinese Usage (25)
3. Clarity of Expression (25)
4. Format Symbol Handling (25)

Output score only:"""
    }
}

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
    """記錄翻譯結果到日誌文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""[{timestamp}]
原文: {original}
{f'原翻譯: {old_translation}' if old_translation else ''}
新翻譯: {new_translation}
{f'評分: {score}' if score else ''}
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

def evaluate_translation(original, translation, max_retries=TRANSLATION_SETTINGS["retry_count"]):
    """使用選擇的提示詞版本進行評分"""
    if not validate_special_chars(original, translation):
        return 0
        
    prompt_template = PROMPTS[TRANSLATION_SETTINGS["prompt_version"]]["evaluation"]
    prompt = prompt_template.format(original=original, translation=translation)
    
    for attempt in range(max_retries):
        response = call_ollama(prompt)
        if response and response != "翻譯服務暫時無法使用，請稍後再試" and response != "無法獲取翻譯結果":
            try:
                score = ''.join(filter(str.isdigit, response))
                return int(score) if score else 50
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
        
    # 使用選擇的提示詞版本
    prompt_template = PROMPTS[TRANSLATION_SETTINGS["prompt_version"]]["translation"]
    
    # 構建 prompt
    prompt = prompt_template.format(
        memory_context=format_memory_context(text, memory),
        translation_context=context.get_context(),
        text=text
    )
    
    translation = call_ollama(prompt).strip()
    if translation and translation not in ["翻譯服務暫時無法使用，請稍後再試", "無法獲取翻譯結果"]:
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

def print_strings(file_path, context: Optional[TranslationContext] = None):
    if context is None:
        context = TranslationContext(
            domain=TRANSLATION_SETTINGS["domain"],
            style=TRANSLATION_SETTINGS["style"]
        )
    
    memory = TranslationMemory()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    strings = data.get('strings', {})
    total = len(strings)
    start_time = time.time()
    processed = 0
    
    for key, value in strings.items():
        try:
            processed += 1
            # 簡單打印進度
            print(f"\n處理進度: {processed}/{total}")
            
            en_localization = value.get('localizations', {}).get('en', {})
            en_text = en_localization.get('stringUnit', {}).get('value') if en_localization else None
            source_type = "en本地化" if en_text else "字串鍵值"
            
            if not en_text:
                en_text = key
            
            zh_tw = value.get('localizations', {}).get('zh-Hant-TW', {})
            zh_tw_value = zh_tw.get('stringUnit', {}).get('value')
            
            if zh_tw_value:
                # 如果有現有翻譯，只打印評分
                if source_type == "字串鍵值":
                    score = evaluate_both_translations(key, en_text, zh_tw_value)
                    log_translation(en_text if en_text else key, zh_tw_value, zh_tw_value, score)
                    print(f"{Colors.BLUE}>> 評分 {key}: {score}{Colors.ENDC}")
                else:
                    score = evaluate_translation(en_text, zh_tw_value)
                    log_translation(en_text, zh_tw_value, zh_tw_value, score)
                    print(f"{Colors.BLUE}>> 評分 {key}: {score}{Colors.ENDC}")
                
                if score < TRANSLATION_SETTINGS["min_score"]:
                    # 需要重新翻譯
                    print(f"{Colors.HEADER}需要重新翻譯 [{source_type}]:{Colors.ENDC}", key)
                    print(f"{Colors.BLUE}原文:{Colors.ENDC}", en_text)
                    print(f"{Colors.BLUE}現有翻譯:{Colors.ENDC}", zh_tw_value)
                    new_translation = translate_with_improve_loop(key, en_text, zh_tw_value, context, memory, source_type)
                    value['localizations']['zh-Hant-TW'] = create_translation_unit(new_translation)
            else:
                # 需要新翻譯
                print(f"{Colors.HEADER}新翻譯 [{source_type}]:{Colors.ENDC}", key)
                print(f"{Colors.BLUE}原文:{Colors.ENDC}", en_text)
                new_translation = translate_with_improve_loop(key, en_text, None, context, memory, source_type)
                if 'localizations' not in value:
                    value['localizations'] = {}
                value['localizations']['zh-Hant-TW'] = create_translation_unit(new_translation)
            
            print("-" * 50)
            
        except Exception as e:
            print(f"{Colors.FAIL}處理項目時發生錯誤: {str(e)}{Colors.ENDC}")
            continue
    
    # 保存翻譯結果
    new_file = save_translations(data, file_path, model=TRANSLATION_SETTINGS["model"])
    print(f"\n翻譯結果已保存至: {new_file}")

def translate_with_improve_loop(key, en_text, current_translation, context, memory, source_type):
    """處理翻譯改進循環"""
    max_attempts = TRANSLATION_SETTINGS["retry_count"]
    current_attempt = 0
    best_translation = current_translation
    best_score = 0

    while current_attempt < max_attempts:
        translation, is_cached = translate_with_ollama(en_text if en_text else key, context, memory)
        
        if is_cached:
            print(f"{Colors.GREEN}>> 使用記憶翻譯，無需評分{Colors.ENDC}")
            return translation
            
        print(f"{Colors.GREEN}>> 翻譯結果:{Colors.ENDC}", translation)
        
        score = evaluate_both_translations(key, en_text, translation) if source_type == "字串鍵值" else evaluate_translation(en_text, translation)
        print(f"{Colors.BLUE}>> 自評分數 (#{current_attempt + 1}):{Colors.ENDC}", score)
        
        if score > best_score:
            best_translation = translation
            best_score = score
            
        if score >= TRANSLATION_SETTINGS["min_score"]:
            break
            
        current_attempt += 1
        
    return best_translation

if __name__ == "__main__":
    # 選擇提示詞版本
    version = input("選擇提示詞版本 (zh/en，預設zh): ").strip().lower()
    if version in ["zh", "en"]:
        TRANSLATION_SETTINGS["prompt_version"] = version
    
    context = TranslationContext(
        domain=TRANSLATION_SETTINGS["domain"],
        style=TRANSLATION_SETTINGS["style"]
    )
    file_path = "Localizable.xcstrings"
    print_strings(file_path, context)
