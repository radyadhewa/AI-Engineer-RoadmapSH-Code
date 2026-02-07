"""
Inferencing via OpenRouter SDK, a simple way.
"""

from openrouter import OpenRouter
from dotenv import load_dotenv
import os

load_dotenv()

print ("Please wait a several moments...")

# LLM inferencing via SDK
with OpenRouter(
    api_key=os.getenv("OPENROUTER_API_KEY")) as client:
    response = client.chat.send(
        model="z-ai/glm-4.5-air:free",
        messages=[
            {"role": "user",
                "content": "What is the best food in Indonesia"
            }
        ],
    )
    print (response.choices[0].message.content)
