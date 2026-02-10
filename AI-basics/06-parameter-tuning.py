'''
Make your AI thinks based on your wants. Like settings in camera

Temperature: How "creative" or "random" the AI is.
Top-P: Limits the AI to a certain set of words.
Max Tokens: How much the AI can write.
'''

from openrouter import OpenRouter
from dotenv import load_dotenv
import os

load_dotenv()

with OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")) as client:
    # low temperature (easy predictable answer)
    response = client.chat.send(
        model="z-ai/glm-4.5-air:free",
        messages=[
            {"role": "user",
             "content": "Write a short story about robot love to cook indonesian food"
            }
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=1500,
    )
    print("Low Temperature Response:")
    print(response.choices[0].message.content)
    print('\n')

    # high temperature (more creative answer)
    response = client.chat.send(
        model="z-ai/glm-4.5-air:free",
        messages=[
            {"role": "user",
             "content": "Write a short story about robot love to cook indonesian food"
            }
        ],
        temperature=0.9,
        top_p=0.9,
        max_tokens=1500,
    )
    print("High Temperature Response:")
    print (response.choices[0].message.content)
    print('\n')

    # low top_p (more focused answer)
    response = client.chat.send(
        model="z-ai/glm-4.5-air:free",
        messages=[
            {"role": "user",
             "content": "Write a short story about robot love to cook indonesian food"
            }
        ],
        temperature=0.7,
        top_p=0.3,
        max_tokens=1500,
    )
    print("Low Top-P Response:")
    print(response.choices[0].message.content)
    print('\n')

    # high top_p (more diverse answer)
    response = client.chat.send(
        model="z-ai/glm-4.5-air:free",
        messages=[
            {"role": "user",
             "content": "Write a short story about robot love to cook indonesian food"
            }
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=1500,
    )
    print("High Top-P Response:")
    print (response.choices[0].message.content)
    print('\n')

    # limited max tokens (may be cut in the middle of the answer)
    response = client.chat.send(
        model="z-ai/glm-4.5-air:free",
        messages=[
            {"role": "user",
             "content": "Write a short story about robot love to cook indonesian food"
            }
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=500,
    )
    print("Limited Max Tokens Response:")
    print(response.choices[0].message.content)
    print()
