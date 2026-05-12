from collections.abc import Iterable, Iterator
import ast
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPE_Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.special_tokens = list(special_tokens or [])
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_rank = {(a, b): i for i, (a, b) in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = {}
        # Parse the whole file so raw control-byte tokens (e.g. '\n', '\t')
        # are handled correctly.
        with open(vocab_filepath, "r", encoding="utf-8", newline="") as f:
            content = f.read()
        for m in re.finditer(r"(?s)(\d+)\t(.*?)(?=\n\d+\t|\Z)", content):
            idx = int(m.group(1))
            token = m.group(2).encode("utf-8", errors="replace")
            vocab[idx] = token
        
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            next(f)  # skip header
            next(f)  # skip separator
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                a_str, b_str = parts[1].strip(), parts[2].strip()
                a = ast.literal_eval(a_str).encode("utf-8", errors="replace")
                b = ast.literal_eval(b_str).encode("utf-8", errors="replace")
                merges.append((a, b))
        
        return cls(vocab, merges, special_tokens)

    def pre_tokenize(self, text: str) -> list[bytes]:
        return [m.group(0).encode("utf-8", errors="replace") for m in re.finditer(PAT, text)]

    def _split_with_special_tokens(self, text: str) -> list[tuple[bool, bytes]]:
        if not self.special_tokens:
            return [(False, text.encode("utf-8", errors="replace"))]

        escaped = [re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)]
        split_pat = re.compile("|".join(escaped))

        out: list[tuple[bool, bytes]] = []
        last = 0
        for m in split_pat.finditer(text):
            if m.start() > last:
                out.append((False, text[last:m.start()].encode("utf-8", errors="replace")))
            out.append((True, m.group(0).encode("utf-8", errors="replace")))
            last = m.end()
        if last < len(text):
            out.append((False, text[last:].encode("utf-8", errors="replace")))
        return out
    
    def bpe_merge(self, pieces: list[bytes]) -> list[bytes]:
        while len(pieces) >= 2:
            best_i = -1
            best_rank = None
            for i in range(len(pieces) - 1):
                rank = self.merge_rank.get((pieces[i], pieces[i + 1]))
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i == -1:
                break
            pieces = pieces[:best_i] + [pieces[best_i] + pieces[best_i + 1]] + pieces[best_i + 2:]
        return pieces
    
    def get_ids_from_bytes(self, byte_list: list[bytes]) -> list[int]:
        token_ids = []
        for byte_seq in byte_list:
            if byte_seq in self.byte_to_id:
                token_ids.append(self.byte_to_id[byte_seq])
            else:
                raise ValueError(f"Byte sequence {byte_seq} not found in vocabulary")
        
        return token_ids
    
    def encode(self, text: str) -> list[int]:
        token_ids = []
        for is_special, chunk in self._split_with_special_tokens(text):
            if is_special:
                token_ids.append(self.byte_to_id[chunk])
                continue
            if not chunk:
                continue
            text_chunk = chunk.decode("utf-8", errors="replace")
            for token in self.pre_tokenize(text_chunk):
                pieces = [bytes([b]) for b in token]
                merged = self.bpe_merge(pieces)
                token_ids.extend(self.get_ids_from_bytes(merged))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs.
        This is required for memory-eﬀicient tokenization
        of large files that we cannot directly load into memory.
        '''
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_list: list[bytes] = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_list.append(self.vocab[token_id])
            else:
                raise ValueError(f"Token ID {token_id} not found in vocabulary")
        
        # decode the byte list back into a string
        return b"".join(byte_list).decode("utf-8", errors="replace")
