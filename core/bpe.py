"""
Byte Pair Encoding (BPE) Tokenizer implementation.
Trains vocabulary merges from corpus and encodes/decodes text.
"""

from collections import Counter
import re


class BPETokenizer:
    def __init__(self, special_tokens=None):
        self.special_tokens = special_tokens or ["<|endoftext|>", "<|pad|>", "<|user|>", "<|assistant|>"]
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = []
        self._init_base_vocab()

    def _init_base_vocab(self):
        # 1. Special tokens
        idx = 0
        for st in self.special_tokens:
            self.vocab[st] = idx
            self.inverse_vocab[idx] = st
            idx += 1
        
        # 2. All standard ASCII / single byte characters
        for b in range(256):
            char = chr(b)
            if char not in self.vocab:
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
                idx += 1

    def train(self, corpus: list[str], max_merges: int = 50):
        """
        Train BPE vocabulary on a list of input text strings.
        """
        # Split each word in corpus into space-separated characters
        word_counts = Counter()
        for text in corpus:
            for word in text.strip().split():
                if word:
                    # Represent word with space between characters and end-of-word tag
                    spaced = " ".join(list(word)) + " </w>"
                    word_counts[spaced] += 1

        for _ in range(max_merges):
            # Count pair frequencies
            pair_counts = Counter()
            for word, freq in word_counts.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            self.merges.append(best_pair)
            new_token = "".join(best_pair)
            if new_token not in self.vocab:
                new_idx = len(self.vocab)
                self.vocab[new_token] = new_idx
                self.inverse_vocab[new_idx] = new_token

            # Apply merge to word counts
            pattern = re.compile(r'(?<!\S)' + re.escape(' '.join(best_pair)) + r'(?!\S)')
            new_word_counts = Counter()
            for word, freq in word_counts.items():
                new_word = pattern.sub(new_token, word)
                new_word_counts[new_word] = freq
            word_counts = new_word_counts

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.
        """
        tokens = []
        words = text.split()
        for idx, word in enumerate(words):
            chars = list(word) + ["</w>"]
            for pair in self.merges:
                i = 0
                while i < len(chars) - 1:
                    if chars[i] == pair[0] and chars[i + 1] == pair[1]:
                        chars = chars[:i] + ["".join(pair)] + chars[i + 2:]
                    else:
                        i += 1
            for token_str in chars:
                if token_str in self.vocab:
                    tokens.append(self.vocab[token_str])
                else:
                    # Fallback to individual characters
                    for c in token_str:
                        if c in self.vocab:
                            tokens.append(self.vocab[c])
            # Add space token between words if not last
            if idx < len(words) - 1 and " " in self.vocab:
                tokens.append(self.vocab[" "])
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs back to text.
        """
        pieces = []
        for tid in token_ids:
            if tid in self.inverse_vocab:
                piece = self.inverse_vocab[tid]
                piece = piece.replace("</w>", " ")
                pieces.append(piece)
        result = "".join(pieces)
        return result.rstrip()
