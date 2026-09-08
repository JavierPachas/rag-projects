import spacy

# Load a pre-trained English model
nlp = spacy.load("en_core_web_sm")

text = """Surely, you're joking, Mr. Feynman! The Adventures of a Curious Character is a book by Richard Feynman, a renowned physicist and Nobel laureate. The book is a collection of anecdotes and stories from Feynman's life, showcasing his curiosity, wit, and unique perspective on the world. It covers various aspects of his career, including his work on the Manhattan Project, his experiences as a teacher, and his adventures in science and beyond. The book provides insight into Feynman's personality, his approach to problem-solving, and his love for learning and discovery."""
doc = nlp(text)

# Iterate over sentences
for sent in doc.sents:
    print(sent.text)    

# Fixed-length chunking with overlap
chunk_size = 30  # Number of characters in each chunk
chunk_overlap = 5  # Number of characters to overlap between chunks
step_size = chunk_size - chunk_overlap
for i in range(0, len(text), step_size):
    chunk = text[i:i + chunk_size]
    print(chunk)