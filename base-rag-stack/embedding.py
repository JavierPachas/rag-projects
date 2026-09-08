
from sentence_transformers import SentenceTransformer
import random
import matplotlib.pyplot as plt

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# List of sentences to embed
sentences = [
    "Surely, you're joking, Mr. Feynman! The Adventures of a Curious Character is a book by Richard Feynman, a renowned physicist and Nobel laureate.",
    "The book is a collection of anecdotes and stories from Feynman's life, showcasing his curiosity, wit, and unique perspective on the world.",
    "It covers various aspects of his career, including his work on the Manhattan Project, his experiences as a teacher, and his adventures in science and beyond.",
    "The book provides insight into Feynman's personality, his approach to problem-solving, and his love for learning and discovery."
]

# Generate embeddings for the sentences
embeddings = model.encode(sentences)

print(embeddings.shape)  # Print the shape of the embeddings array
print(embeddings[:,:5])  # Print the first 5 dimensions of each embedding

similarity_matrix = model.similarity(embeddings, embeddings)

print(similarity_matrix)  # Print the similarity matrix

plt.figure(figsize=(5, 5))
plt.imshow(
    similarity_matrix, cmap = 'RdYlGn_r', interpolation = 'nearest', vmin = 0, vmax = 1
)
plt.title("Cosine similarity between sentence embeddings")
plt.xlabel("Sentence ID")
plt.ylabel("Sentence ID")
plt.yticks([0,1,2,3])

# Adding text annotations to the heatmap
for i in range(len(similarity_matrix)):
    for j in range(len(similarity_matrix)):
        plt.text(j, i, f"{similarity_matrix[i][j]:.2f}", ha='center', va='center', color='black')

plt.show()

# generate two random integers between 0 and 384
rand1 = random.randint(0, 384)
rand2 = random.randint(0, 384)

if rand1 > rand2:
    rand1, rand2 = rand2, rand1  # Swap to ensure rand1 is less than or equal to rand2

print(f"Using semantic dimensiones from {rand1} to {rand2}")

similarity_matrix = model.similarity(embeddings[:, rand1:rand2], embeddings[:, rand1:rand2])

print(similarity_matrix)  # Print the similarity matrix for the selected dimensions
