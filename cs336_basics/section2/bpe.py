import regex as re
import heapq
from collections import defaultdict
import tqdm


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# def pretokenize(text: str) -> dict[str, int]:
#     freq = {}
#     # token_list = re.findall(PAT, text)
#     token_list = [m.group(0) for m in re.finditer(PAT, text)]   
#     for i in token_list:
#         freq[i] = freq.get(i, 0) + 1
#     return freq

def build_token_freq(data: str, special_tokens: list[str]) -> dict[str, int]:
    '''
    Build a frequency dictionary for tokens in the given data.
    {} -> {'hello': 5, 'world': 3, ...}
    '''
    token_freq = {}

    if special_tokens:
        split_pat = "|".join(re.escape(t) for t in special_tokens)
        chunks = re.split(split_pat, data)
    else:
        chunks = [data]

    for chunk in chunks:
        if not chunk:
            continue
        for m in re.finditer(PAT, chunk):
            tok = m.group(0)
            token_freq[tok] = token_freq.get(tok, 0) + 1

    return token_freq

def get_byte_freq(token_freq: dict[str, int]) -> dict[tuple, int]:
    '''
    Convert token frequencies to byte frequencies.
    {} -> {(b'h', b'e', b'l', b'l', b'o'): 5, ...}
    '''
    byte_freq = {}
    for k, v in token_freq.items():
        # byte_freq[tuple(k)] = v
        byte_freq[tuple(bytes([b]) for b in k.encode("utf-8"))] = v
    return byte_freq

def calculate_pair_freq(byte_freq: dict[tuple, int]) -> dict[tuple[str, str], int]:
    '''
    Calculate the frequency of adjacent byte pairs in the byte frequency dictionary.
    {(b'h', b'e', b'l', b'l', b'o'): 5
    '''
    pair_freq = {}
    for k, v in byte_freq.items():
        if len(k) < 2:
            continue
        for i in range(1, len(k)):
            pair = (k[i-1], k[i])
            pair_freq[pair] = pair_freq.get(pair, 0) + v
    return pair_freq

def calculate_pair_idx(byte_freq: dict[tuple, int]) -> dict[tuple, set[tuple]]:
    pair_idx = {}
    for seq in byte_freq:
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i+1])
            pair_idx.setdefault(p, set()).add(seq)
    return pair_idx

def get_most_freq_pair(pair_freq: dict[tuple[str, str], int]) -> tuple[str, str]:
    # return max(pair_freq.items(), key=lambda x: x[1])[0]
    return max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]

def merge_pair(byte_freq: dict[tuple[str, str], int], pair_to_merge: tuple[str, str]) -> dict[tuple[str, str], int]:
    merged_freq = {}
    for k, v in byte_freq.items():
        i = 0
        k_new = []
        while i < len(k):
            if i + 1 < len(k) and k[i] == pair_to_merge[0] and k[i+1] == pair_to_merge[1]:
                # k_new.append(''.join(pair_to_merge))
                k_new.append(k[i] + k[i+1])
                i += 2
            else:
                k_new.append(k[i])
                i += 1

        new_k = tuple(k_new)
        merged_freq[new_k] = merged_freq.get(new_k, 0) + v

    return merged_freq

def update_pair_freq(pair_freq: dict[tuple[str, str], int], byte_freq: dict[tuple[str, str], int], pair_to_merge: tuple[str, str]) -> dict[tuple[str, str], int]:
    updated_pair_freq = {}
    for k, v in pair_freq.items():
        if pair_to_merge in k:
            continue
        updated_pair_freq[k] = v

    for k, v in byte_freq.items():
        if len(k) < 2:
            continue
        for i in range(1, len(k)):
            pair = (k[i-1], k[i])
            if pair == pair_to_merge:
                new_pair = (k[i-1] + k[i],)
                updated_pair_freq[new_pair] = updated_pair_freq.get(new_pair, 0) + v
            else:
                updated_pair_freq[pair] = updated_pair_freq.get(pair, 0) + v

    return updated_pair_freq

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    print('yes')

    data = open(input_path, "r", encoding="utf-8").read()

    print('data loaded')

    # # separate data into chunks based on <|endoftext|>
    # chunks = data.split("<|endoftext|>")

    # token_freq = {}
    # # pretokenize each chunk and accumulate token frequencies
    # for chunk in chunks:
    #     chunk_freq = pretokenize(chunk)
    #     for k, v in chunk_freq.items():
    #         token_freq[k] = token_freq.get(k, 0) + v

    token_freq = build_token_freq(data, special_tokens)

    print('token frequencies built')

    # initialize token list with single byte tokens
    # token_list = {int(k.encode('utf-8')[0]) : k.encode('utf-8') for k in vocab}
    token_list = {i: bytes([i]) for i in range(256)}
    idx = 256

    # remove special tokens from data
    for token in special_tokens:
        token_list[idx] = token.encode('utf-8')
        # data = data.replace(token, "")
        idx += 1

    byte_freq = get_byte_freq(token_freq)
    pair_freq = calculate_pair_freq(byte_freq)
    print('pair frequencies built')

    # add a reverse index for byte pairs to bytes
    pair_idx = calculate_pair_idx(byte_freq)

    # store the merged pair and tokens that include the pair,
    merges = []

    print('starting merges')

    # total number of merges we expect to perform
    total_merges = vocab_size - idx

    with tqdm(total=total_merges, desc="Training BPE", unit="merge") as pbar:
        while idx < vocab_size and pair_freq:
            pair_to_merge = get_most_freq_pair(pair_freq)
            merges.append(pair_to_merge)

            A, B = pair_to_merge
            AB = A + B

            # show current merge info in progress bar
            pbar.set_postfix({
                "vocab": idx,
                "pair": repr(AB.decode("utf-8", errors="replace"))[:30],
                "freq": pair_freq.get(pair_to_merge, 0),
            })

            # only look at affected sequences via pair_idx
            affected_seqs = list(pair_idx.pop(pair_to_merge, []))
            del pair_freq[pair_to_merge]

            for seq in affected_seqs:
                freq = byte_freq.pop(seq)

                # Step 1: undo this seq's contributions to pair_freq / pair_idx
                for i in range(len(seq) - 1):
                    p = (seq[i], seq[i + 1])

                    if p == pair_to_merge:
                        continue

                    pair_freq[p] -= freq
                    if pair_freq[p] <= 0:
                        del pair_freq[p]

                    if p in pair_idx:
                        pair_idx[p].discard(seq)
                        if not pair_idx[p]:
                            del pair_idx[p]

                # Step 2: build the merged sequence
                new_seq = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == A and seq[i + 1] == B:
                        new_seq.append(AB)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1

                new_seq = tuple(new_seq)

                # Step 3: add new_seq's contributions back
                byte_freq[new_seq] = byte_freq.get(new_seq, 0) + freq

                for i in range(len(new_seq) - 1):
                    p = (new_seq[i], new_seq[i + 1])
                    pair_freq[p] = pair_freq.get(p, 0) + freq
                    pair_idx.setdefault(p, set()).add(new_seq)
        
            # byte_freq = merge_pair(byte_freq, pair_to_merge)
            # pair_freq = calculate_pair_freq(byte_freq)
            
            # value as bytes, key as int
            # token_list[idx] = ''.join(pair_to_merge)
            # token_list[idx] = pair_to_merge[0] + pair_to_merge[1]
            token_list[idx] = AB
            idx += 1

            pbar.update(1)

    return token_list, merges