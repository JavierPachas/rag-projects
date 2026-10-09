# How to use cross-encoder reranking models (BGE Reranker) to improve the relevance ordering of retrieved documents.

"""
- Load and use the BGE Reranker v2 cross-encoder model
- Score query-document pairs for relevance
- Rerank retrieved documents by relevance score
"""

from sentence_transformers.cross_encoder import CrossEncoder

# Prepare query and documents

query = "What is the main benefit of using a transformer model in NLP?"

documents = [
    "Recurrent Neural Networks (RNNs) were previously popular for sequence tasks.", # Less relevant
    "Transformers allow for parallel processing of input tokens, leading to faster training times compared to RNNs.", # Highly relevant (speed/parallelism)
    "BERT, a popular transformer model, achieves state-of-the-art results on many NLP benchmarks.", # Relevant context, but not the *main benefit* of the architecture itself
    "The attention mechanism in transformers enables the model to weigh the importance of different words in the input sequence.", # Highly relevant (attention mechanism)
    "Convolutional Neural Networks (CNNs) are primarily used in computer vision.", # Irrelevant
    "A key advantage of the transformer architecture is its ability to handle long-range dependencies more effectively than RNNs.", # Highly relevant (long-range dependencies)
    "You can fine-tune pre-trained transformer models for specific downstream tasks." # Related benefit, but maybe not the *main* architectural one
]

model_name = 'BAAI/bge-reranker-v2-m3'

try:
    # Let sentence-transformers handle device placement automatically
    # It usually defaults to GPU if available and compatible, otherwise CPU.
    model = CrossEncoder(model_name)
    print(f"Successfully loaded BGE Reranker model: {model_name}")
    print("-" * 30)
except Exception as e:
    print(f"Error loading model '{model_name}': {e}")
    print("Please ensure 'sentence-transformers' is installed (`pip install -U sentence-transformers`)")
    print("Also ensure you have PyTorch or TensorFlow installed.")
    exit()

# Score and rerank
"""
1. Prepare the query-document pairs for scoring.
2. Use the model to score each pair.
3. Sort the documents based on their scores in descending order.
4. Display the results.
"""

sentence_pairs = [(query, doc) for doc in documents]

print(f"Scoring {len(sentence_pairs)} query-document pairs...")
scores = model.predict(sentence_pairs, show_progress_bar=True)

print("-" * 30)
print("Raw scores:", scores)
print("-" * 30)

docs_with_scores = list(zip(documents, scores))
reranked_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)

print("Query:", query)
print("\n-- Original Document Order --")
for i, doc in enumerate(documents):
    print(f"{i + 1}. {doc}")

print("\n-- Reranked Document Order --")
print("Rank | Score | Document")
for rank, (doc, score) in enumerate(reranked_docs):
    print(f"{rank + 1:>4} | {score:.4f} | {doc}")

