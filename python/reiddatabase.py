#!/usr/bin/env python3

import os
import json
import numpy as np
import faiss
from typing import Dict, List, Optional


class ReIDDatabase:
    def __init__(
        self,
        dim: int = 512,
        similarity_threshold: float = 0.75,
        db_path: str = "reid_db"
    ):
        """
        FAISS-backed ReID database.

        Args:
            dim: embedding dimension (512 for OSNet)
            similarity_threshold: cosine similarity threshold for matching
            db_path: folder to persist DB
        """
        self.dim = dim
        self.threshold = similarity_threshold
        self.db_path = db_path

        os.makedirs(db_path, exist_ok=True)

        # FAISS index (cosine similarity via inner product)
        self.index = faiss.IndexFlatIP(dim)

        # Metadata
        self.id_map: List[int] = []  # maps FAISS idx -> global_id
        self.embeddings: Dict[int, List[np.ndarray]] = {}  # raw embeddings
        self.mean_embeddings: Dict[int, np.ndarray] = {}   # averaged embeddings

        self.next_id = 0

    # ---------------------------
    # Utilities
    # ---------------------------

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        return emb / np.linalg.norm(emb)

    def _update_index(self):
        """Rebuild FAISS index from mean embeddings."""
        self.index.reset()
        self.id_map = []

        if not self.mean_embeddings:
            return

        vectors = []
        for gid, emb in self.mean_embeddings.items():
            vectors.append(emb.astype(np.float32))
            self.id_map.append(gid)

        vectors = np.vstack(vectors)
        self.index.add(vectors)

    # ---------------------------
    # Core API
    # ---------------------------

    def add_identity(self, embedding: np.ndarray) -> int:
        """Create a new identity."""
        embedding = self._normalize(embedding)

        gid = self.next_id
        self.next_id += 1

        self.embeddings[gid] = [embedding]
        self.mean_embeddings[gid] = embedding

        self._update_index()
        return gid

    def update_identity(self, gid: int, embedding: np.ndarray):
        """Update an existing identity with new embedding."""
        embedding = self._normalize(embedding)

        self.embeddings[gid].append(embedding)

        # Update mean embedding
        self.mean_embeddings[gid] = np.mean(
            self.embeddings[gid], axis=0
        )
        self.mean_embeddings[gid] = self._normalize(self.mean_embeddings[gid])

        self._update_index()

    def query(self, embedding: np.ndarray) -> Optional[int]:
        """
        Query the database.

        Returns:
            matched global_id or None
        """
        if self.index.ntotal == 0:
            return None

        embedding = self._normalize(embedding).astype(np.float32)
        embedding = np.expand_dims(embedding, axis=0)

        sims, indices = self.index.search(embedding, k=1)

        sim = sims[0][0]
        idx = indices[0][0]

        if sim >= self.threshold:
            return self.id_map[idx]

        return None

    def get_or_create(self, embedding: np.ndarray) -> int:
        """
        Main entry point:
        - match existing identity OR
        - create new one
        """
        gid = self.query(embedding)

        if gid is not None:
            self.update_identity(gid, embedding)
            return gid

        return self.add_identity(embedding)

    # ---------------------------
    # Track-level aggregation
    # ---------------------------

    def aggregate_track(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Aggregate multiple embeddings into one."""
        if len(embeddings) == 0:
            raise ValueError("Empty embedding list")

        emb = np.mean(embeddings, axis=0)
        return self._normalize(emb)

    # ---------------------------
    # Persistence
    # ---------------------------

    def save(self):
        """Save database to disk."""
        data = {
            "next_id": self.next_id,
            "embeddings": {
                str(k): [e.tolist() for e in v]
                for k, v in self.embeddings.items()
            }
        }

        with open(os.path.join(self.db_path, "db.json"), "w") as f:
            json.dump(data, f)

    def load(self):
        """Load database from disk."""
        path = os.path.join(self.db_path, "db.json")
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            data = json.load(f)

        self.next_id = data["next_id"]

        self.embeddings = {
            int(k): [np.array(e) for e in v]
            for k, v in data["embeddings"].items()
        }

        # rebuild means
        self.mean_embeddings = {
            gid: self._normalize(np.mean(v, axis=0))
            for gid, v in self.embeddings.items()
        }

        self._update_index()

    # ---------------------------
    # Debugging
    # ---------------------------

    def __len__(self):
        return len(self.mean_embeddings)

    def stats(self):
        return {
            "num_identities": len(self),
            "total_embeddings": sum(len(v) for v in self.embeddings.values())
        }
