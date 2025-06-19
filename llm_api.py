from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("请先设置环境变量 OPENAI_API_KEY")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def llm_chat(messages, system_prompt=None):
    # messages: list of {"role": "user"/"assistant", "content": ...}
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content 