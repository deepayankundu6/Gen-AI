# Generate_embedings

This small script generates embeddings using an OpenAI-compatible API running at http://localhost:5000/v1/embeddings and stores them in a local MongoDB instance.

Files
- `Generate_embedings.py` - main script
- `requirements.txt` - Python deps

Usage (Windows cmd)

1. Install dependencies in your Python environment:

```cmd
python -m pip install -r requirements.txt
```

2. Run with a single text:

```cmd
python Generate_embedings.py --text "Hello world"
```

3. Run with a file (one line per text):

```cmd
python Generate_embedings.py --file input.txt
```

4. Read from console / piped stdin:

```cmd
# Interactive console mode (enter lines, finish with an empty line)
python Generate_embedings.py --console

# Pipe from another command or file
type input.txt | python Generate_embedings.py --console
```

Configuration
All configuration can be set via environment variables or command line arguments. Command line arguments take precedence over environment variables.

Environment Variables:
```
# API Configuration
OPENAI_API_BASE          - OpenAI-compatible base URL (default: http://localhost:5000/v1)
OPENAI_API_KEY           - API key for authentication

# MongoDB Configuration
MONGO_URI               - MongoDB connection string (default: mongodb://localhost:27017)
DB_NAME                 - Database name (default: Gen_AI)
EMBEDDING_COLLECTION    - Collection for embeddings (default: Embedings)
RESULT_COLLECTION       - Collection for responses (default: Result)

# Model Configuration
EMBEDDING_MODEL         - Model for generating embeddings (default: text-embedding-3-small)
RESPONSE_MODEL         - Model for generating responses (default: deepseek/deepseek-r1-0528-qwen3-8b)

# Processing Configuration
BATCH_SIZE             - Number of texts to process per batch (default: 16)
```

Command Line Arguments (override environment variables):
- `--api-base`: Override OPENAI_API_BASE
- `--api-key`: Override OPENAI_API_KEY
- `--mongo`: Override MONGO_URI
- `--embedding-model`: Override EMBEDDING_MODEL
- `--response-model`: Override RESPONSE_MODEL
- `--batch-size`: Override BATCH_SIZE

MongoDB
- Database: `Gen_AI`
- Collection: `Embedings`

Notes
- The script uses the `openai` Python package and configures `openai.api_base` so it can work with OpenAI-compatible servers (like a local proxy).
- Keep the local embedding server and MongoDB running before executing the script.
