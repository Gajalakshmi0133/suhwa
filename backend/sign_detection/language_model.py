import math
from collections import defaultdict


class NGramLM:
    """A tiny character-level bigram language model with add-one smoothing.
    Train with a list of lowercased sentences (space-separated words allowed).
    """
    def __init__(self):
        self.unigrams = defaultdict(int)
        self.bigrams = defaultdict(lambda: defaultdict(int))
        self.total_unigrams = 0

    def train_from_sentences(self, sentences):
        for s in sentences:
            s = s.strip().lower()
            if not s:
                continue
            prev = None
            for ch in s:
                self.unigrams[ch] += 1
                self.total_unigrams += 1
                if prev is not None:
                    self.bigrams[prev][ch] += 1
                prev = ch

    def score(self, sequence):
        """Return log-probability of a sequence (sum of log bigram probs).
        Uses add-one smoothing over observed characters.
        """
        seq = sequence.lower()
        if not seq:
            return 0.0
        vocab_size = max(1, len(self.unigrams))
        logp = 0.0
        prev = None
        for ch in seq:
            if prev is None:
                # unigram probability
                count = self.unigrams.get(ch, 0)
                prob = (count + 1) / (self.total_unigrams + vocab_size)
            else:
                prev_counts = self.bigrams.get(prev, {})
                count = prev_counts.get(ch, 0)
                denom = sum(prev_counts.values()) + vocab_size
                prob = (count + 1) / denom
            logp += math.log(prob)
            prev = ch
        return logp


def build_lm_from_file(path):
    lm = NGramLM()
    with open(path, 'r', encoding='utf8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    lm.train_from_sentences(lines)
    return lm


class WordNGramLM:
    """Simple word-level bigram language model with add-one smoothing.
    Train with a list of sentences; scoring accepts a sentence string and
    returns log-probability (sum of log bigram probs over words).
    """
    def __init__(self):
        self.unigrams = defaultdict(int)
        self.bigrams = defaultdict(lambda: defaultdict(int))
        self.total_unigrams = 0

    def train_from_sentences(self, sentences):
        for s in sentences:
            s = s.strip().lower()
            if not s:
                continue
            words = s.split()
            prev = None
            for w in words:
                self.unigrams[w] += 1
                self.total_unigrams += 1
                if prev is not None:
                    self.bigrams[prev][w] += 1
                prev = w

    def score(self, sentence):
        """Score a sentence (string). Returns log-probability (float)."""
        s = (sentence or '').strip().lower()
        if not s:
            return 0.0
        words = s.split()
        vocab_size = max(1, len(self.unigrams))
        logp = 0.0
        prev = None
        for w in words:
            if prev is None:
                count = self.unigrams.get(w, 0)
                prob = (count + 1) / (self.total_unigrams + vocab_size)
            else:
                prev_counts = self.bigrams.get(prev, {})
                count = prev_counts.get(w, 0)
                denom = sum(prev_counts.values()) + vocab_size
                prob = (count + 1) / denom
            logp += math.log(prob)
            prev = w
        return logp


def build_word_lm_from_file(path):
    lm = WordNGramLM()
    with open(path, 'r', encoding='utf8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    lm.train_from_sentences(lines)
    return lm
