# base-rag-stack

Small, self-contained scripts covering each stage of a RAG pipeline:
document parsing, chunking, embedding, and vector search.

## Setup

All scripts run from inside this folder with the local virtual environment.

```bash
cd base-rag-stack
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The requirements file also pulls the spaCy `en_core_web_sm` language model
from GitHub, so no separate `spacy download` step is needed.

Run any script with:

```bash
.venv/bin/python <script>.py
```

## Scripts

| Script | What it does | Needs |
|---|---|---|
| `parse-pdf.py` | Extract text, blocks, tables, and images from a PDF with PyMuPDF | nothing extra |
| `parse-docx.py` | Extract paragraphs, tables, and images from a DOCX | nothing extra |
| `parse-llm.py` | Send the PDF and an extracted image to an OpenAI model for text, table, and image description | OpenAI API key, and `parse-pdf.py` run first |
| `chunking.py` | Sentence splitting with spaCy and fixed-length chunking with overlap | nothing extra |
| `embedding.py` | Embed sentences with Sentence Transformers and plot a cosine similarity heatmap | nothing extra |
| `pgvector-simple.py` | Store embeddings in PostgreSQL with pgvector and run similarity search in SQL | running PostgreSQL with pgvector |

Sample inputs live in `sample_data/`. Scripts that extract images write them
to this folder as `extracted_image_*` files, which are gitignored.

## OpenAI API key (`parse-llm.py`)

Create a `.env` file in this folder. It is gitignored and never committed.

```
OPENAI_API_KEY=sk-...
```

The script loads it with `python-dotenv`. Run `parse-pdf.py` first so that
`extracted_image_0_0.png` exists, since the last step of `parse-llm.py`
sends that image to the model.

## PostgreSQL with pgvector (`pgvector-simple.py`)

The script expects a PostgreSQL server on `localhost:5432` with user
`postgres`. It creates a database named `ragbook`, enables the `vector`
extension, and builds an HNSW index. The password is read from the
`PGVECTOR_PASSWORD` environment variable and defaults to `RAGBOOK`.

The simplest way to get a matching server is Docker with the official
pgvector image, which ships with the extension preinstalled.

1. Install and start Docker Desktop.

2. Start a container. This matches the script defaults:

   ```bash
   docker run -d \
     --name pgvector \
     -e POSTGRES_PASSWORD=RAGBOOK \
     -p 5432:5432 \
     pgvector/pgvector:pg17
   ```

3. Confirm it is accepting connections:

   ```bash
   docker exec pgvector pg_isready -U postgres
   ```

4. Run the script:

   ```bash
   .venv/bin/python pgvector-simple.py
   ```

To use a different password, set the variable before running:

```bash
export PGVECTOR_PASSWORD=yourpassword
```

Useful container commands:

```bash
docker stop pgvector      # stop, keep data
docker start pgvector     # start again
docker rm -f pgvector     # remove container and its data
```

If you already have PostgreSQL installed locally instead of Docker, install
the pgvector extension for your version and make sure the connection
details at the top of the script match your server.
