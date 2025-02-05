# 智能翻譯工具

這是一個專門用於處理iOS本地化檔案（.xcstrings）的智能翻譯工具，使用LLM（Large Language Model）進行翻譯，並具有翻譯記憶和品質控制功能。

## 功能展示

[![功能展示](https://img.youtube.com/vi/eYZaniR_ZIA/0.jpg)](https://www.youtube.com/watch?v=eYZaniR_ZIA)

點擊上方圖
## 功能特點

- 支援 .xcstrings 檔案的批次翻譯
- 使用 LLM (推薦 Gemma 2:27b) 進行智能翻譯
- 翻譯記憶功能，避免重複翻譯
- 自動保留所有變數格式（如 %@、%d、%lld 等）
- 翻譯品質自動評分
- 低分自動重試機制
- 完整的翻譯日誌記錄
- 支援專業領域術語一致性

## 系統要求

- Python 3.8+
- Ollama (需要安裝 Gemma2 模型)

## 使用說明

```bash
git clone [repository-url]
cd SwiftLLMTranslator
python main.py
```