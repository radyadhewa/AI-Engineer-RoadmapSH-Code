'''
Prompt engineering is the art of making AI follow instructions and understand you.
The goal is to elicit accurate, relevant, and useful responses from the AI
by crafting prompts that are clear, specific, and contextually appropriate.

We will continue the inferencing-via-sdk but improvised with prompt engineering techniques.
'''

from openrouter import OpenRouter
from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_URL = os.getenv("OPENROUTER_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# define the 'system prompt' to set the behavior of the AI
SYSTEM_PROMPT = """
You are a world-class Culinary Travel Guide specializing in Indonesian cuisine.
Your goal is to provide detailed, mouth-watering descriptions.

### CONSTRAINTS ###
- Mention at least one regional specialty (e.g., Padang, Sundanese).
- Explain the taste of each dish by describing its key ingredients.
- Suggest the best time of day to enjoy each dish.
- Format the output as a bulleted list.
"""

print ("Please wait a several moments...")

# call the inference sdk
with OpenRouter(
    api_key=os.getenv("OPENROUTER_API_KEY")) as client:
    response = client.chat.send(
        model="z-ai/glm-4.5-air:free",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}, # we add system role here with our system prompt above
            {"role": "user",
                "content": "What are the top 3 must-try traditional foods in Indonesia?"
            }
        ],
    )
    print (response.choices[0].message.content)


'''
there  are several prompt engineering techniques applied here:
1. System Prompting: We define a system prompt to set the context and behavior of the AI.
2. Role Specification: We use roles (system and user) to clarify the source of each message.
3. Detailed Instructions: The system prompt includes specific constraints and formatting instructions to guide the AI's response.
These techniques help ensure that the AI provides a more accurate and contextually relevant answer.

Next is, learn more about prompt engineering techniques and experiments.
'''