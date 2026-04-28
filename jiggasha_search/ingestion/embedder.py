import json
import logging
from typing import Any, Dict, List

import numpy as np
import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

QUERY_PREFIX = "task: search result | query: "
PASSAGE_PREFIX = "title: none | text: "


class EmbedderConfig(BaseModel):
    """Configuration for the Triton embedder."""

    triton_url: str = Field(description="Base URL for the Triton Inference Server")
    triton_request_timeout: int = Field(default=480, description="Request timeout in seconds.")
    model_name: str = Field(default="gemma_embedding", description="Name of the model in Triton.")
    tokenizer_name: str = Field(default="onnx-community/embeddinggemma-300m-ONNX", description="HF tokenizer name.")
    triton_output_name: str = Field(default="sentence_embedding", description="Name of the output tensor.")
    batch_size: int = Field(default=8, description="Batch size for embedding requests sent to Triton.")


class TritonEmbedder:
    """Embeds text via a Triton Inference Server. Supports query and passage prefixes for ChromaDB."""

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        logger.info("Embedder initialized for Triton at %s (batch_size=%d)", config.triton_url, config.batch_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, texts: List[str]) -> Dict[str, Any]:
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=2048, return_tensors="np")
        return {
            "input_ids": tokens["input_ids"].astype(np.int64),
            "attention_mask": tokens["attention_mask"].astype(np.int64),
        }

    def _build_payload(self, texts: List[str]) -> Dict[str, Any]:
        tokens = self._tokenize(texts)
        return {
            "inputs": [
                {"name": "input_ids", "shape": list(tokens["input_ids"].shape), "datatype": "INT64", "data": tokens["input_ids"].flatten().tolist()},
                {"name": "attention_mask", "shape": list(tokens["attention_mask"].shape), "datatype": "INT64", "data": tokens["attention_mask"].flatten().tolist()},
            ],
            "outputs": [{"name": self.config.triton_output_name}],
        }

    @staticmethod
    def _post_process(data: Dict[str, Any], output_name: str) -> List[List[float]]:
        output = next((o for o in data["outputs"] if o["name"] == output_name), None)
        if output is None:
            raise ValueError(f"Output '{output_name}' not found in Triton response.")
        return np.array(output["data"], dtype=np.float32).reshape(output["shape"]).tolist()

    def _embed_raw(self, texts: List[str], model_name: str) -> List[List[float]]:
        if not texts:
            return []
        payload = self._build_payload(texts)
        url = f"{self.config.triton_url.rstrip('/')}/v2/models/{model_name}/infer"
        resp = requests.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=self.config.triton_request_timeout)
        resp.raise_for_status()
        return self._post_process(resp.json(), self.config.triton_output_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """Embed queries with the query prefix."""
        if not texts:
            return []
        prefixed = [QUERY_PREFIX + t for t in texts]
        return [
            emb
            for i in range(0, len(prefixed), self.config.batch_size)
            for emb in self._embed_raw(prefixed[i : i + self.config.batch_size], self.config.model_name)
        ]

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """Embed passages with the passage prefix."""
        if not texts:
            return []
        prefixed = [PASSAGE_PREFIX + t for t in texts]
        return [
            emb
            for i in range(0, len(prefixed), self.config.batch_size)
            for emb in self._embed_raw(prefixed[i : i + self.config.batch_size], self.config.model_name)
        ]

    def as_chroma_embedder(self) -> EmbeddingFunction:
        """Return an EmbeddingFunction compatible with ChromaDB (passages only)."""

        class _ChromaWrapper(EmbeddingFunction):
            def __init__(self, embedder: TritonEmbedder):
                self._embedder = embedder

            def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
                return self._embedder.embed_passages(input)  # type: ignore[arg-type]

        return _ChromaWrapper(self)
