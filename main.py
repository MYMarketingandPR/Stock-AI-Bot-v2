import os
import requests
from google import genai

# 拿出我們藏好的鑰匙
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(text):
    # 呼叫小郵差寄信
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def analyze_stock():
    # 叫超級大腦開始工作
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # 這裡放我們最簡單的指令（之後可以換成我們超強的10分提示詞）
    prompt = "你是一個華爾街大師，請用一句話告訴我今天 TSLA 的股票狀態，並給出一個買入或觀望的建議。"
    
    response = client.models.generate_content(
        model='gemini-1.5-flash', contents=prompt,
    )
    
    # 把大腦寫好的報告，加上標題
    report = f"📊 **每日美股 AI 報告** 📊\n\n{response.text}"
    send_telegram_message(report)

if __name__ == "__main__":
    analyze_stock()
