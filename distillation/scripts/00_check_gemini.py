import os
import google.generativeai as genai

# Configure Gemini 2.5 Pro using your API key from environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-pro")

# Quick test query
response = model.generate_content("Say 'Gemini test OK'")
print(response.text)

