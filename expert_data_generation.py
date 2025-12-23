import os
import google.generativeai as genai
import json
import time
from tqdm import tqdm  # 用于显示进度条

# ================= 配置区域 =================
# 请替换你的 API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAttTS3ZPtU4_wE-3wnOUcIquatiyhFlx4"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 初始化模型 (推荐使用 Pro 或 Flash)
model = genai.GenerativeModel('gemini-3-flash-preview')

# 定义 10 个高难度领域
FIELDS = [
    "Physics",
    "Mathematics",
    "Computer Science",
    "Philosophy",
    "Geography",
    "Chemistry",
    "Biology",
    "Economics",
    "Statistics",
    "Law"
]

# 每个领域生成多少条
TARGET_PER_FIELD = 50
# 每次 API 调用生成多少条 (建议 5 条，保证每条的深度和长度)
BATCH_SIZE = 5

# ================= 核心生成逻辑 =================

def generate_expert_dataset():
    all_data = []
    
    print(f"🚀 开始生成数据任务")
    print(f"📚 领域数量: {len(FIELDS)}")
    print(f"🎯 目标总数: {len(FIELDS) * TARGET_PER_FIELD}")
    
    # 遍历每个领域
    for field in FIELDS:
        print(f"\nProcessing Field: {field}...")
        field_data = []
        
        # 计算需要多少个批次 (例如 50 / 5 = 10 次)
        num_batches = TARGET_PER_FIELD // BATCH_SIZE
        
        # 使用 tqdm 显示当前领域的进度
        for i in tqdm(range(num_batches), desc=f"Generating {field}"):
            
            # 精心设计的 Prompt，强调 "Graduate-level" 和 "English"
            prompt = f"""
            Role: You are a distinguished professor specializing in {field}.
            
            Task: Generate {BATCH_SIZE} distinct Q&A pairs focusing on the **most important, foundational, and cornerstone concepts** of this field.
            
            Guidelines:
            1.  **Selection Criteria:** Do NOT ask obscure trivia or insanely difficult calculations. Instead, ask about the "Big Ideas", "Central Dogmas", or "Fundamental Theorems" that define the field.
            2.  **Question Style:** The questions should ask for conceptual depth, mechanisms, or the underlying logic (e.g., "Why does...", "Explain the principle of...", "What is the significance of...").
            3.  **Answer Quality:** The answer must be rigorous, academic, and comprehensive (Graduate-level understanding), NOT a simplified summary.
            4.  **Language:** Strictly English.
            5.  **Format:** Return a RAW JSON list.
            
            Output Example (if field was Biology):
            [
                {{
                    "field": "Biology",
                    "question": "Explain the central dogma of molecular biology and its significance in genetic expression.",
                    "answer": "The central dogma describes the flow of genetic information within a biological system..."
                }}
            ]
            """
            
            try:
                # 调用 Gemini API
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json", # 强制 JSON
                        "temperature": 0.8 #稍微高一点，增加多样性
                    }
                )
                
                # 解析数据
                batch_items = json.loads(response.text)
                
                # 简单验证一下数量
                if isinstance(batch_items, list):
                    field_data.extend(batch_items)
                
                # 避免触发 API 速率限制 (Rate Limit)
                time.sleep(2)
                
            except Exception as e:
                print(f"⚠️ Error in batch {i} for {field}: {e}")
                time.sleep(5) # 出错多歇会儿

        # 将当前领域的数据加入总表
        all_data.extend(field_data)
        
        # 为了防止程序半途崩溃，每完成一个领域就存一次盘
        save_to_file(all_data, filename="dataset_capability_english_partial.json")

    # 最终保存
    save_to_file(all_data, filename="dataset_capability_english_final.json")
    print(f"\n✅ 全部完成！共生成 {len(all_data)} 条数据。")

def save_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_expert_dataset()