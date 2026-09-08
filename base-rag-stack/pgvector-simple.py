"""
Vector Search with pgvector

- How to create a pgvector table and HNSW index in PostgreSQL
- How to encode text into embeddings with Sentence Transformers and store them
- How to perform vector similarity search using SQL

Prerequisites:

A running PostgreSQL instance with the pgvector extension installed
pip install -r requirements.txt
Set PGVECTOR_PASSWORD environment variable (export PGVECTOR_PASSWORD=yourpassword)
"""

from typing import List
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

sample_sentences = [
    "Surely, you're joking, Mr. Feynman! The Adventures of a Curious Character is a book by Richard Feynman, a renowned physicist and Nobel laureate.",
    "The book is a collection of anecdotes and stories from Feynman's life, showcasing his curiosity, wit, and unique perspective on the world.",
    "It covers various aspects of his career, including his work on the Manhattan Project, his experiences as a teacher, and his adventures in science and beyond.",
    "The book provides insight into Feynman's personality, his approach to problem-solving, and his love for learning and discovery."
]

embeddings = model.encode(sample_sentences)

# Database connection parameters
pg_password = os.getenv("PGVECTOR_PASSWORD", "RAGBOOK")
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': pg_password,
    'database': 'postgres'
}

# Connect to a PostgreSQL database
conn = psycopg2.connect(
    **DB_CONFIG
)
conn.autocommit = True
cursor = conn.cursor()

DB_FOR_VECTORS = "ragbook"

try:
    cursor.execute(f"CREATE DATABASE {DB_FOR_VECTORS};")
except psycopg2.errors.DuplicateDatabase:
    print(f"Database {DB_FOR_VECTORS} already exists.")

# Switch to the new database
DB_CONFIG['database'] = DB_FOR_VECTORS
conn = psycopg2.connect(
    **DB_CONFIG
)
conn.autocommit = True
cursor = conn.cursor()

# Enable pgvector extension
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
register_vector(conn)  # return VECTOR columns as pgvector Vector objects instead of strings

# Drop the table if it already exists
cursor.execute("DROP TABLE IF EXISTS sentence_embeddings;")

# Create a table to store sentences and their embeddings
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sentence_embeddings (
        text TEXT NOT NULL,
        embedding VECTOR(384)
    );
    """)
# Create HNSW index in the table "sentence_embeddings" for efficient similarity search
cursor.execute("""
    CREATE INDEX sentence_embeddings_hnsw_idx
    ON sentence_embeddings
    USING hnsw (embedding vector_l2_ops)
    WITH (m = 16, ef_construction = 64);
    """)

# Insert sample sentences and their embeddings into the table "sentence_embeddings" 
for sentence, embedding in zip(sample_sentences, embeddings):
    embedding_as_list: List[float] = embedding.tolist()
    cursor.execute(
        "INSERT INTO sentence_embeddings (text, embedding) "
        "VALUES (%s, %s::vector)",
        (sentence, embedding_as_list)
    )

cursor.execute("SELECT text, embedding FROM sentence_embeddings;")
rows = cursor.fetchall()
for row in rows:
    print(row[0], row[1].to_numpy()[:3])

# Embed a query and find the most similar sentence in the database
def vector_search(query, model, top_k):
    query_embedding = model.encode([query])[0]
    query_embedding_as_list: List[float] = query_embedding.tolist()
    
    cursor.execute(
        """
        SELECT text,
        1 - (embedding <=> %s::vector) as similarity
        FROM sentence_embeddings
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_embedding_as_list, query_embedding_as_list, top_k)
    )
    
    results = cursor.fetchall()

    for row in results:
        print(row)


query = "What is the book about?"
vector_search(query, model, top_k=3)

query = "Who is the main character in the book?"
vector_search(query, model, top_k=3)