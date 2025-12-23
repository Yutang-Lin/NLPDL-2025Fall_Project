import os
import google.generativeai as genai
import json
import time
from tqdm import tqdm

# ================= 配置区域 =================
os.environ["GOOGLE_API_KEY"] = "AIzaSyAttTS3ZPtU4_wE-3wnOUcIquatiyhFlx4"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel('gemini-3-flash-preview')

# 定义 10 个生活化/通用话题，确保覆盖面广
TOPICS = [
    "Daily Routine & Chores",
    "Weather & Climate",
    "Food & Dining",
    "Shopping & Commerce",
    "Technology & Gadgets",
    "Health & Fitness",
    "Work & Office Life",
    "Travel & Transport",
    "Emotions & Relationships",
    "Entertainment"
]

TARGET_PER_TOPIC = 1
BATCH_SIZE = 1

# ================= 核心生成逻辑 =================

def generate_style_pairs_dataset():
    all_data = []
    
    print(f"🚀 开始生成风格差异数据 (Style Pairs)")
    print(f"🌍 话题数量: {len(TOPICS)}")
    print(f"🎯 目标总数: {len(TOPICS) * TARGET_PER_TOPIC}")
    
    for topic in TOPICS:
        print(f"\nProcessing Topic: {topic}...")
        topic_data = []
        num_batches = TARGET_PER_TOPIC // BATCH_SIZE
        
        for i in tqdm(range(num_batches), desc=f"Generating {topic}"):
            
            # ====================================================
            # 风格差异专用 PROMPT
            # ====================================================
            prompt = f"""
            Role: You are a linguistic expert specializing in register variation (Style Transfer).
            
            Task: Generate {BATCH_SIZE} triples of sentences related to the topic: "{topic}".
            
            Structure for each triple:
            1.  **Neutral:** A simple, factual, standard English sentence (The core meaning).
            2.  **Casual:** The same meaning, but rewritten in extremely informal, slang-heavy, spoken English (Gen-Z style, text message style, simple vocab).
            3.  **Formal:** The same meaning, but rewritten in extremely formal, academic, bureaucratic, or archaic English (Complex syntax, GRE vocabulary, passive voice).
            
            Guidelines:
            - The **meaning** must remain exactly the same across all three versions.
            - **Casual** should use contractions, slang (like "gonna", "legit", "vibes"), and simple structure.
            - **Formal** should sound like a legal contract, a scientific paper, or a 19th-century novel.
            - Strictly English.
            - Return a RAW JSON list.
            
            Output Example:
            [
                {{
                    "topic": "{topic}",
                    "neutral": "I am very hungry.",
                    "casual": "Yo, I'm literally starving rn, need food ASAP.",
                    "formal": "The subject is currently experiencing an acute physiological requirement for nutritional sustenance."
                }}
            ]
            """
            
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.8 # 稍微高一点，让风格差异更夸张
                    }
                )
                
                batch_items = json.loads(response.text)
                if isinstance(batch_items, list):
                    topic_data.extend(batch_items)
                
                time.sleep(1.0)
                
            except Exception as e:
                print(f"⚠️ Error in batch {i}: {e}")
                time.sleep(2)

        all_data.extend(topic_data)
        # 断点保存
        save_to_file(all_data, filename="dataset_style_pairs_partial.json")

    # 最终保存
    save_to_file(all_data, filename="dataset_style_pairs_final.json")
    print(f"\n✅ 全部完成！共生成 {len(all_data)} 条风格数据。")

def save_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    generate_style_pairs_dataset()