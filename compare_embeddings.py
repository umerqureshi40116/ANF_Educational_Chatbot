"""
This script demonstrates how to:
1. Generate embeddings for a word using OpenAI's embedding model.
2. Compare the similarity (distance) between two words using LangChain's evaluator.
"""

from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import openai
import os

# Load environment variables from .env file (for API key)
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

def main():
    """
    Main function to:
    - Generate embedding for the word "apple".
    - Compare embeddings of two words ("apple" and "iphone").
    """

    # Create embedding function using OpenAI's "text-embedding-3-large" model
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

    # Get embedding vector for the word "apple"
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")  # Print the full embedding vector
    print(f"Vector length: {len(vector)}")  # Print the length of the vector

    # Load evaluator to compare embeddings between two words
    evaluator = load_evaluator("pairwise_embedding_distance")

    # Words to compare
    words = ("apple", "iphone")

    # Compare embeddings of the two words and calculate distance
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])

    # Print comparison result (lower distance means more similarity)
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


# Entry point of the script
if __name__ == "__main__":
    main()
