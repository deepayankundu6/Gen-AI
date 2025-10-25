"""
processor.py

Main script for generating embeddings and responses, storing them in MongoDB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from requests.exceptions import RequestException

from embedding_generator import EmbeddingGenerator
from response_generator import ResponseGenerator, ResponseAPIError

# Constants
DEFAULT_OPENAI_API_BASE = "http://localhost:5000/v1"
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "Gen_AI"
EMBEDDING_COLLECTION = "Embedings"
RESULT_COLLECTION = "Results"
API_KEY = "Dummy api key"


def save_documents(mongo_uri: str, embedding_docs: List[Dict[str, Any]], response_docs: List[Dict[str, Any]]) -> None:
    """Save embeddings and responses to their respective collections."""
    client = MongoClient(mongo_uri)
    db = client[DB_NAME]
    emb_coll = db[EMBEDDING_COLLECTION]
    res_coll = db[RESULT_COLLECTION]
    
    try:
        if embedding_docs:
            # Insert embeddings
            result = emb_coll.insert_many(embedding_docs)
            print(f"Inserted {len(embedding_docs)} documents into {DB_NAME}.{EMBEDDING_COLLECTION}")
            
            # Update response documents with matching _ids
            for doc_id, resp_doc in enumerate(response_docs):
                resp_doc["_id"] = result.inserted_ids[doc_id]
            
            res_coll.insert_many(response_docs)
            print(f"Inserted {len(response_docs)} documents into {DB_NAME}.{RESULT_COLLECTION}")
        else:
            print("No documents to insert.")
    except PyMongoError as e:
        print("Failed to insert documents into MongoDB:", e)
        raise
    finally:
        client.close()


def build_embedding_docs(texts: List[str], embeddings: List[List[float]]) -> List[Dict[str, Any]]:
    """Build documents for the embeddings collection."""
    docs = []
    ts = int(time.time())
    for i, (t, emb) in enumerate(zip(texts, embeddings)):
        docs.append({
            "text": t,
            "embedding": emb,
            "created_at": ts,
            "source": "local_script",
            "index": i,
        })
    return docs


def build_response_docs(texts: List[str], responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build documents for the responses collection."""
    docs = []
    ts = int(time.time())
    for i, (t, resp) in enumerate(zip(texts, responses)):
        doc = {
            "text": t,
            "response": resp["text"],
            "created_at": resp.get("created_at", ts),
            "response_id": resp.get("id"),
            "model": resp.get("model"),
            "usage": resp.get("usage", {}),
            "source": "local_script",
            "index": i,
        }
        docs.append(doc)
    return docs


def read_texts_from_file(path: str) -> List[str]:
    """Read texts from a newline-delimited file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    return [l for l in lines if l]


def chunked(iterable: List[Any], size: int):
    """Split iterable into chunks of specified size."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Generate embeddings and responses, store in MongoDB")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t", help="Text to process")
    group.add_argument("--file", "-f", help="Path to newline-delimited file of texts")
    group.add_argument("--console", "-c", action="store_true", help="Read text lines from console/STDIN")
    p.add_argument("--embedding-model", default="text-embedding-3-small", help="Embedding model to use")
    p.add_argument("--response-model", default="deepseek/deepseek-r1-0528-qwen3-8b", help="Response model to use")
    p.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", DEFAULT_OPENAI_API_BASE), help="OpenAI API base URL")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", API_KEY), help="OpenAI API key")
    p.add_argument("--mongo", default=os.environ.get("MONGO_URI", DEFAULT_MONGO_URI), help="MongoDB URI")
    p.add_argument("--batch-size", type=int, default=16, help="Number of texts per request")
    return p.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Initialize generators
    embedding_gen = EmbeddingGenerator(api_base=args.api_base, api_key=args.api_key)
    response_gen = ResponseGenerator(api_base=args.api_base, api_key=args.api_key)

    # Get input texts
    if args.text:
        texts = [args.text]
    elif args.file:
        try:
            texts = read_texts_from_file(args.file)
        except Exception as e:
            print(f"Failed to read file {args.file}: {e}")
            return 2
    elif args.console:
        # Read from stdin (pipe or interactive)
        if not sys.stdin.isatty():
            data = sys.stdin.read()
            texts = [l for l in (data.splitlines()) if l.strip()]
        else:
            print("Enter text lines. Submit an empty line to finish:")
            lines: List[str] = []
            try:
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
            except EOFError:
                pass
            texts = lines
    else:
        # Fallback to piped input if available
        if not sys.stdin.isatty():
            data = sys.stdin.read()
            texts = [l for l in (data.splitlines()) if l.strip()]
        else:
            print("No input provided. Use --text, --file, or --console. Exiting.")
            return 1

    if not texts:
        print("No input texts found.")
        return 1

    all_embedding_docs: List[Dict[str, Any]] = []
    all_response_docs: List[Dict[str, Any]] = []

    for batch in chunked(texts, args.batch_size):
        # Retry logic for transient failures
        last_exc = None
        for attempt in range(3):
            try:
                # Get embeddings
                print(f"Getting embeddings.....")
                embeddings = embedding_gen.get_embeddings("".join(batch), model=args.embedding_model)
                embedding_docs = build_embedding_docs(batch, embeddings)
                all_embedding_docs.extend(embedding_docs)
                print(f"Embeddings received.")
                # Get responses
                try:
                    print(f"Getting responses....")
                    responses = response_gen.get_responses("".join(batch), model=args.response_model)
                    if responses and isinstance(responses[0], dict):
                        print(f"Response received (id: {responses[0].get('id')})")
                        print(f"Model: {responses[0].get('model')}")
                        if 'usage' in responses[0]:
                            usage = responses[0]['usage']
                            print(f"Tokens used: {usage.get('total_tokens', 0)} "
                                  f"(input: {usage.get('input_tokens', 0)}, "
                                  f"output: {usage.get('output_tokens', 0)})")
                    response_docs = build_response_docs(batch, responses)
                    all_response_docs.extend(response_docs)
                    print(f"Responses received successfully")
                except ResponseAPIError as e:
                    print(f"Response API error (attempt {attempt+1}/3):")
                    print(f"  Status: {e.status_code}")
                    print(f"  Message: {str(e)}")
                    if e.response_body:
                        print(f"  Details: {json.dumps(e.response_body, indent=2)}")
                    raise
                except RequestException as e:
                    print(f"Network error (attempt {attempt+1}/3): {e}")
                    raise
                
                # Small delay between batches
                time.sleep(0.05)
                break
            except Exception as e:
                last_exc = e
                print(f"API request failed (attempt {attempt+1}/3): {e}")
                time.sleep(0.5 * (attempt + 1))
        else:
            print("Failed after retries:", last_exc)
            return 3

    try:
        save_documents(args.mongo, all_embedding_docs, all_response_docs)
    except Exception:
        return 4

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())