# model.py

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# ✅ Load environment variables (make sure GOOGLE_API_KEY is in your .env file)
load_dotenv()

# ✅ Initialize Gemini 2.0 Flash model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",   # use gemini-1.5-pro if you want a bigger one
)
