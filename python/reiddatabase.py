#!/usr/bin/env python3

class ReIDDatabase:
    def __init__(self, dim=768):  # ViT embeddings often 768
        self.index = faiss.IndexFlatIP(dim)
        self.player_ids = []
        self.embeddings = []

    def add(self, emb):
        self.index.add(emb.reshape(1, -1))
        new_id = len(self.player_ids)
        self.player_ids.append(new_id)
        self.embeddings.append([emb])
        return new_id

    def match(self, emb, threshold=0.75):
        if len(self.player_ids) == 0:
            return None

        D, I = self.index.search(emb.reshape(1, -1), k=1)

        if D[0][0] > threshold:
            return self.player_ids[I[0][0]]
        return None

    def update(self, player_id, emb):
        self.embeddings[player_id].append(emb)

        # optional: keep last N embeddings
        if len(self.embeddings[player_id]) > 20:
            self.embeddings[player_id].pop(0)


class TrackMemory:
    def __init__(self):
        self.buffer = defaultdict(lambda: deque(maxlen=10))

    def add(self, track_id, emb):
        self.buffer[track_id].append(emb)

    def get(self, track_id):
        if len(self.buffer[track_id]) < 5:
            return None
        avg = np.mean(self.buffer[track_id], axis=0)
        avg /= np.linalg.norm(avg)
        return avg
