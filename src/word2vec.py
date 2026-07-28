"""Minimal skip-gram Word2Vec with negative sampling, implemented in plain NumPy.

`gensim` (the usual source of a Word2Vec baseline) does not yet ship a
Python 3.14 wheel and fails to build from source on this machine (its
Cython extensions use CPython internals removed in 3.14). This module
implements the same skip-gram-negative-sampling objective directly so the
milestone-1 "baseline Word2Vec embeddings" requirement is still met, using
only numpy.
"""

from collections import Counter

import numpy as np


class SimpleWord2Vec:
    def __init__(self, embedding_dim=50, window=2, negative=5, seed=42):
        self.embedding_dim = embedding_dim
        self.window = window
        self.negative = negative
        self.rng = np.random.default_rng(seed)
        self.word2idx = {}
        self.idx2word = []
        self.W_in = None
        self.W_out = None

    def build_vocab(self, tokenized_sentences, min_count=2):
        counts = Counter(tok for sent in tokenized_sentences for tok in sent)
        self.idx2word = [w for w, c in counts.items() if c >= min_count]
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        vocab_size = len(self.idx2word)

        freqs = np.array([counts[w] for w in self.idx2word], dtype=np.float64)
        self._neg_probs = freqs**0.75
        self._neg_probs /= self._neg_probs.sum()

        limit = 0.5 / self.embedding_dim
        self.W_in = self.rng.uniform(-limit, limit, size=(vocab_size, self.embedding_dim))
        self.W_out = np.zeros((vocab_size, self.embedding_dim))
        return vocab_size

    def _make_pairs(self, tokenized_sentences):
        centers, contexts = [], []
        for sent in tokenized_sentences:
            idxs = [self.word2idx[w] for w in sent if w in self.word2idx]
            n = len(idxs)
            for i, center in enumerate(idxs):
                lo, hi = max(0, i - self.window), min(n, i + self.window + 1)
                for j in range(lo, hi):
                    if j != i:
                        centers.append(center)
                        contexts.append(idxs[j])
        return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def train(self, tokenized_sentences, epochs=5, batch_size=512, lr=0.01, min_count=2, verbose=True):
        self.build_vocab(tokenized_sentences, min_count=min_count)
        centers, contexts = self._make_pairs(tokenized_sentences)
        n_pairs = len(centers)
        if verbose:
            print(f"vocab={len(self.idx2word)} pairs={n_pairs}")

        for epoch in range(epochs):
            order = self.rng.permutation(n_pairs)
            total_loss = 0.0
            for start in range(0, n_pairs, batch_size):
                batch_idx = order[start : start + batch_size]
                c = centers[batch_idx]
                o = contexts[batch_idx]
                bsz = len(c)
                neg = self.rng.choice(len(self.idx2word), size=(bsz, self.negative), p=self._neg_probs)

                v_c = self.W_in[c]  # (bsz, D)
                u_o = self.W_out[o]  # (bsz, D)
                u_neg = self.W_out[neg]  # (bsz, neg, D)

                pos_score = self._sigmoid(np.sum(v_c * u_o, axis=1))  # (bsz,)
                neg_score = self._sigmoid(np.sum(v_c[:, None, :] * u_neg, axis=2))  # (bsz, neg)

                total_loss += float(
                    -np.sum(np.log(pos_score + 1e-10)) - np.sum(np.log(1 - neg_score + 1e-10))
                )

                grad_u_o = (pos_score - 1.0)[:, None] * v_c  # (bsz, D)
                grad_v_from_pos = (pos_score - 1.0)[:, None] * u_o  # (bsz, D)
                grad_u_neg = neg_score[:, :, None] * v_c[:, None, :]  # (bsz, neg, D)
                grad_v_from_neg = np.sum(neg_score[:, :, None] * u_neg, axis=1)  # (bsz, D)
                grad_v_c = grad_v_from_pos + grad_v_from_neg

                # clip so that repeated updates to a frequent word within one
                # vectorized batch (an approximation of sequential SGD) can't blow up
                clip = 5.0
                grad_v_c = np.clip(grad_v_c, -clip, clip)
                grad_u_o = np.clip(grad_u_o, -clip, clip)
                grad_u_neg = np.clip(grad_u_neg, -clip, clip)

                np.add.at(self.W_in, c, -lr * grad_v_c)
                np.add.at(self.W_out, o, -lr * grad_u_o)
                np.add.at(self.W_out, neg.reshape(-1), (-lr * grad_u_neg).reshape(-1, self.embedding_dim))

            if verbose:
                print(f"epoch {epoch + 1}/{epochs} loss={total_loss / max(n_pairs, 1):.4f}")

    def sentence_vector(self, tokens):
        idxs = [self.word2idx[w] for w in tokens if w in self.word2idx]
        if not idxs:
            return np.zeros(self.embedding_dim)
        return self.W_in[idxs].mean(axis=0)
