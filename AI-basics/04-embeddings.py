'''
embeddings The process of translating human-scale concepts (words, images, or audio)
into a high-dimensional mathematical "map" where proximity equals similarity in meaning.

we will save the embedding to vector database for later retrieval and use.

because there is no free openrouter embedding model, we will use huggingface inference API
'''

import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# ensure .env has HUGGINGFACE_TOKEN
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN must be set in your .env file.")

# initialize Hugging Face inference client
client = InferenceClient(api_key=HF_TOKEN)

# function to get embeeding
def get_embedding(text):
    model_id = "sentence-transformers/all-mpnet-base-v2"

    # This generates a 768-dimensional vector
    return client.feature_extraction(text, model=model_id)

# usage
text = "Indonesian rendang is a slow-cooked beef dish."
vector = get_embedding(text)

print("Model: all-mpnet-base-v2")
print(f"Vector Length: {len(vector)}") # Should be 768
print(f"Embedding Vector: {vector[:5]}...") # Show first 5 values for preview
