#!/usr/bin/env python3
import numpy as np
import faiss
from collections import defaultdict, deque

class ReIDDatabase:
    def __init__(self, dim=768, alpha=0.1):
        self.dim = dim
        self.alpha = alpha
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.prototypes = {}
        self.next_id = 0


    def add(self, emb):
        emb = self._normalize(emb)

        player_id = self.next_id
        self.next_id += 1

        self.prototypes[player_id] = emb

        # FAISS requires IDs to be 64-bit ints
        self.index.add_with_ids(emb.reshape(1, -1), np.array([player_id], dtype=np.int64))

        return player_id

    def match(self, emb, threshold=0.9):
        if self.index.ntotal == 0:
            return None

        emb = self._normalize(emb)

        # Search for the top 1 closest prototype
        D, I = self.index.search(emb.reshape(1, -1), k=1)

        similarity = D[0][0]
        matched_id = I[0][0]

        if similarity > threshold:
            return matched_id
        return None

    def update(self, player_id, emb):
        """
        Updates the player's prototype using Exponential Moving Average (EMA).
        This prevents 'drift' where a single blurry frame ruins the database ID.
        """
        if player_id not in self.prototypes:
            return

        emb = self._normalize(emb)

        # 1. Calculate the new EMA prototype
        old_proto = self.prototypes[player_id]
        new_proto = (1.0 - self.alpha) * old_proto + self.alpha * emb
        new_proto = self._normalize(new_proto)

        # 2. Update dictionary storage
        self.prototypes[player_id] = new_proto

        # 3. Update FAISS (Remove old vector, add new vector with same ID)
        self.index.remove_ids(np.array([player_id], dtype=np.int64))
        self.index.add_with_ids(new_proto.reshape(1, -1), np.array([player_id], dtype=np.int64))

    def _normalize(self, emb):
        """Force L2 normalization. FAISS Inner Product only acts as Cosine Similarity if vectors are normalized."""
        norm = np.linalg.norm(emb)
        if norm > 0:
            return (emb / norm).astype("float32")
        return emb.astype("float32")


class TrackMemory:
    def __init__(self, history=10):
        self.buffer = defaultdict(lambda: deque(maxlen=history))

    def add(self, track_id, emb):
        self.buffer[track_id].append(emb)

    def get(self, track_id, min_frames=5):
        # we still want to try and ID them.
        if len(self.buffer[track_id]) < min_frames:
            return None

        # Average the embeddings in the buffer to create a robust initial shot
        avg = np.mean(self.buffer[track_id], axis=0)

        norm = np.linalg.norm(avg)
        if norm > 0:
            avg /= norm

        return avg.astype("float32")
