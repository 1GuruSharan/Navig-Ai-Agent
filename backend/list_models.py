# list_models.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("Gemini Key Loaded:", bool("GEMINI_API_KEY"))
