"""
Load: Download and parse a PDF document
Split: Chunk the document into smaller pieces
Embed & Store: Create embeddings and store in a vector database
Query: Retrieve relevant chunks and generate responses
"""

from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY from sample/.env into the environment

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import LanceDB
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load the PDF from URL
pdf_url = "https://www.adobe.com/be_en/active-use/pdf/Alice_in_Wonderland.pdf"
loader = PyPDFLoader(pdf_url)
pages = loader.load()

print(f"Loaded {len(pages)} pages from the PDF")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = text_splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = LanceDB.from_documents(chunks, embeddings)

print(f"Created vector store with {len(chunks)} documents")

# Create the RAG chain using LCEL (LangChain Expression Language)
# Using LCEL (LangChain Expression Language), we create a chain that:
"""
Takes a question
Retrieves relevant documents
Formats them as context
Passes to the LLM for answering
"""
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}"""
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask questions about Alice in Wonderland
questions = [
    "Who is the main character and what happens at the beginning of the story?",
    "What did the Caterpillar ask Alice?",
    "Describe the Mad Hatter's tea party.",
    "What happened at the trial at the end of the story?"
]

for q in questions:
    answer = rag_chain.invoke(q)
    print(f"Q: {q}")
    print(f"A: {answer}")
    print("-" * 60)