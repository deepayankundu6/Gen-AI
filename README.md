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
- OPENAI_API_BASE environment variable or `--api-base` to change the OpenAI-compatible base URL (defaults to http://localhost:5000/v1)
- OPENAI_API_KEY or `--api-key` to provide an API key if your local server requires one.
- MONGO_URI or `--mongo` to change the MongoDB connection string (defaults to mongodb://localhost:27017).

MongoDB
- Database: `Gen_AI`
- Collection: `Embedings`

Notes
- The script uses the `openai` Python package and configures `openai.api_base` so it can work with OpenAI-compatible servers (like a local proxy).
- Keep the local embedding server and MongoDB running before executing the script.
