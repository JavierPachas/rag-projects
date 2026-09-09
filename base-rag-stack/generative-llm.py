import dotenv
dotenv.load_dotenv()

from openai import OpenAI

client = OpenAI()

query = "What is the main theme of the book 'Surely, You're Joking, Mr. Feynman!'?"

context = """Surely, you're joking, Mr. Feynman! The Adventures of a Curious Character is a book by Richard Feynman, a renowned physicist and Nobel laureate. The book is a collection of anecdotes and stories from Feynman's life, showcasing his curiosity, wit, and unique perspective on the world. It covers various aspects of his career, including his work on the Manhattan Project, his experiences as a teacher, and his adventures in science and beyond. The book provides insight into Feynman's personality, his approach to problem-solving, and his love for learning and discovery."""  

prompt_template =  """
You are a helpful assistant. Use the following context to briefly answer the question.
Query: {query}
Context: {context}
"""
model = "gpt-5.5"
response = client.chat.completions.create(
    model=model,
    max_completion_tokens=200,
    messages=[
        {"role": "user", "content": prompt_template.format(query=query, context=context)}
    ]
)

print(response.choices[0].message.content)