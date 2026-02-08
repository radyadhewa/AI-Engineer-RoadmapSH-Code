"""
Via API call to OpenRouter to interact with LLMs.
"""

# main library
import json
import os

import requests
from dotenv import load_dotenv

# other libraries (to create loading screen)
import itertools
import sys
import threading
import time

load_dotenv()

OPENROUTER_URL = os.getenv("OPENROUTER_URL")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

# validation
if not OPENROUTER_URL or not OPENROUTER_KEY:
    raise ValueError(
        "OPENROUTER_URL and OPENROUTER_KEY must be set in environment variables."
    )

# headers (the label) you are sending to the API
headers = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
}

# payload (the content) you are sending to the API
payload = {
    "model": "z-ai/glm-4.5-air:free",
    "messages": [{"role": "user", "content": "What is the best food in Indonesia"}],
}

# Function to display a loading spinner while waiting for LLM response
def show_loading_spinner():
    for char in itertools.cycle("|/-\\"):
        sys.stdout.write(f"\rLoading for answer {char}")
        sys.stdout.flush()
        time.sleep(0.1)

spinner_thread = threading.Thread(target=show_loading_spinner, daemon=True)
spinner_thread.start()

# Make the POST request to the OpenRouter API
response = requests.post(url=OPENROUTER_URL, headers=headers, data=json.dumps(payload))

# Stop the spinner after the response is received
spinner_thread.join(0)
sys.stdout.write("\r")  # Clear the spinner line

# Check the response status and print the result
if response.status_code == 200:
    print(response.json()["choices"][0]["message"]["content"])
else:
    print(f"Error: {response.status_code} - {response.text}")
