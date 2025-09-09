import json
import os
import time
import gdown
from gpt4all import GPT4All
from pymongo import MongoClient
 
# --- Step 1: Download model from Google Drive if not exists ---
drive_url = "https://drive.google.com/uc?id=1c2XOp78-KgIECyMWpvhKyDVj74KiFv5L"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "Meta-Llama-3-8B-Instruct.Q4_0.gguf")

if not os.path.exists(model_path):
    print("📥 Downloading Meta-Llama-3-8B-Instruct.Q4_0.gguf from Google Drive...")
    gdown.download(drive_url, model_path, quiet=False)
else:
    print("✅ Model already exists locally")

# --- Step 2: Load GPT4All model ---
model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", model_path=model_dir, device="cpu")
print("✅ Model loaded successfully")

# --- Step 3: Load input.json ---
with open("input.json", "r", encoding="utf-8") as f:
    questions = json.load(f)
questions=questions[:1]
# --- Step 4: Connect to MongoDB Atlas using environment variable ---
MONGO_URL = os.environ.get("MONGO_URL")
if not MONGO_URL:
    raise ValueError("❌ MONGO_URL environment variable not set!")

client = MongoClient(MONGO_URL)
db = client["java_codes"]
collection = db["code_ans"]

# --- Step 5: Generate Java code for each question and save ---
for obj in questions:
    prompt = f"""
You are a Java programmer. Write full Java code for the following question. 
Use the given base code to complete it. Only return the code inside the main class.
Do not add explanations and comments.

Question:
{obj['question']}

Basecode:
{obj['basecode']}
"""
    print(f"🟢 Generating code for: {obj['title']}...")

    code_output = []
    with model.chat_session():
        for token in model.generate(prompt, max_tokens=1000, streaming=True):
            code_output.append(token)

    obj["output"] = "".join(code_output).strip()

    # Save to MongoDB
    collection.insert_one(obj)
    print(f"✅ Code for '{obj['title']}' saved to MongoDB\n")
    time.sleep(1)

print("✅ All Java codes generated and saved successfully!")
