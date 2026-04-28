import argparse
import logging
import os

import chromadb
import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from embedder import EmbedderConfig, TritonEmbedder

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRITON_URL = os.environ.get("TRITON_EMBEDDER_URL", "http://localhost:7000")
CHROMA_HOST = os.environ.get("CHROMA_DB_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_DB_PORT", 8443))


def load_config(path: str) -> dict:
    logger.info("Loading config from %s", path)
    with open(path) as f:
        return yaml.safe_load(f)


def build_embedder_config(data_cfg: dict) -> EmbedderConfig:
    embedder_section = data_cfg.get("embedder", {})
    return EmbedderConfig(
        triton_url=embedder_section.get("triton_url", TRITON_URL),
        model_name=embedder_section.get("model_name", "gemma_embedding"),
        tokenizer_name=embedder_section.get("tokenizer_name", "onnx-community/embeddinggemma-300m-ONNX"),
        triton_output_name=embedder_section.get("triton_output_name", "sentence_embedding"),
        batch_size=embedder_section.get("batch_size", 8),
        triton_request_timeout=embedder_section.get("triton_request_timeout", 480),
    )


def main(config_path: str):
    data_cfg = load_config(config_path)
    collection_name = data_cfg["collection_name"]

    # 1. Embedder
    embedder_cfg = build_embedder_config(data_cfg)
    embedder = TritonEmbedder(embedder_cfg)

    try:
        # 2. ChromaDB
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        heartbeat = client.heartbeat()
        logger.info("ChromaDB connected. Heartbeat: %s", heartbeat)

        # 3. Drop existing collection (if any)
        try:
            client.delete_collection(name=collection_name)
            logger.info("Dropped existing collection '%s'.", collection_name)
        except Exception:
            logger.info("Collection '%s' does not exist yet.", collection_name)

        # 4. Create collection — no embedding function, we supply embeddings manually
        collection = client.get_or_create_collection(name=collection_name)
        logger.info("Collection '%s' ready.", collection_name)

        # 5. Read CSV
        df = pd.read_csv(data_cfg["csv_file_path"])
        content_col = data_cfg["content_column"]
        metadata_cols: list = data_cfg["metadata_columns"]  # type: ignore[assignment]
        total = len(df)

        # 6. Embed and ingest one doc at a time
        for idx in tqdm(range(total), desc="Ingesting"):
            doc = df.iloc[idx]
            text = doc[content_col]
            metadatas = {col: doc[col] for col in metadata_cols} if metadata_cols else None  # type: ignore[index]
            doc_id = f"row_{idx}"

            embeddings = embedder.embed_passages([str(text)])
            collection.add(documents=[str(text)], metadatas=[metadatas] if metadatas else None, ids=[doc_id], embeddings=embeddings)  # type: ignore[arg-type]

        logger.info("Done! Collection contains %d documents.", collection.count())

    except FileNotFoundError:
        logger.error("CSV not found: %s", data_cfg.get("csv_file_path"))
    except Exception:
        logger.error("Ingestion failed.", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CSV data into ChromaDB.")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to YAML config file.")
    args = parser.parse_args()
    main(args.config)
