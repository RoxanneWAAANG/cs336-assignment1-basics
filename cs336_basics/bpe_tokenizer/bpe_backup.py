import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# def pretokenize(text: str) -> dict[str, int]:
#     freq = {}
#     # token_list = re.findall(PAT, text)
#     token_list = [m.group(0) for m in re.finditer(PAT, text)]   
#     for i in token_list:
#         freq[i] = freq.get(i, 0) + 1
#     return freq

def build_token_freq(data: str, special_tokens: list[str]) -> dict[str, int]:
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
    byte_freq = {}
    for k, v in token_freq.items():
        # byte_freq[tuple(k)] = v
        byte_freq[tuple(bytes([b]) for b in k.encode("utf-8"))] = v
    return byte_freq

def calculate_pair_freq(byte_freq: dict[tuple, int]) -> dict[tuple[str, str], int]:
    pair_freq = {}
    for k, v in byte_freq.items():
        if len(k) < 2:
            continue
        for i in range(1, len(k)):
            pair = (k[i-1], k[i])
            pair_freq[pair] = pair_freq.get(pair, 0) + v
    return pair_freq

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
        # byte_freq[tuple(k_new)] = v
        # del byte_freq[k]
        new_k = tuple(k_new)
        merged_freq[new_k] = merged_freq.get(new_k, 0) + v
        # merged_freq[tuple(k_new)] = merged_freq.get(tuple(k_new), 0) + v
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

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. text → token_freq
    data = open(input_path, "r").read() # text

    # # separate data into chunks based on <|endoftext|>
    # chunks = data.split("<|endoftext|>")

    # token_freq = {}
    # # pretokenize each chunk and accumulate token frequencies
    # for chunk in chunks:
    #     chunk_freq = pretokenize(chunk)
    #     for k, v in chunk_freq.items():
    #         token_freq[k] = token_freq.get(k, 0) + v

    token_freq = build_token_freq(data, special_tokens)

    # initialize token list with single byte tokens
    # token_list = {int(k.encode('utf-8')[0]) : k.encode('utf-8') for k in vocab}
    token_list = {i: bytes([i]) for i in range(256)}
    idx = 256

    # remove special tokens from data
    for token in special_tokens:
        token_list[idx] = token.encode('utf-8')
        # data = data.replace(token, "")
        idx += 1

    # byte_freq = get_byte_freq(token_freq)
    # pair_freq = calculate_pair_freq(byte_freq)

    # 2: build seqs
    seqs = {}
    seq_id = 0

    for token_str, freq in token_freq.items():
        byte_seq = tuple(bytes([b]) for b in token_str.encode("utf-8"))
        seqs[seq_id] = [list(byte_seq), freq]  # use list for mutability
        seq_id += 1

    # 3: build indices
    pair_to_occurrences = {}
    pair_freq = {}

    for sid, (seq, freq) in seqs.items():
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])

            # update freq
            pair_freq[pair] = pair_freq.get(pair, 0) + freq

            # update occurrences
            if pair not in pair_to_occurrences:
                pair_to_occurrences[pair] = set()
            pair_to_occurrences[pair].add(sid)

    merges = []

    while idx < vocab_size:
        if not pair_freq:
            break

        # pair_to_merge = get_most_freq_pair(pair_freq)
        # merges.append(pair_to_merge)  # already bytes!

        # 4. pick best pair
        pair_to_merge = get_most_freq_pair(pair_freq)
        merges.append(pair_to_merge)

        # 5. find affected sequences
        affected_seqs = pair_to_occurrences.get(pair_to_merge, set())

        for sid in affected_seqs:
            seq, freq = seqs[sid]

            # 6. update only those sequences
            i = 0
            new_seq = []
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair_to_merge:
                    merged = seq[i] + seq[i+1]
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            seqs[sid][0] = new_seq

            # 7. update pair_freq locally
            for j in range(len(seq) - 1):
                pair = (seq[j], seq[j+1])
                pair_freq[pair] = pair_freq.get(pair, 0) + freq

        # 8. update pair_to_occurrences
        pair_to_occurrences[pair].add(sid)

        # byte_freq = merge_pair(byte_freq, pair_to_merge)
        # pair_freq = calculate_pair_freq(byte_freq) # this is inefficient

        # 9. update vocab
        # value as bytes, key as int
        # token_list[idx] = ''.join(pair_to_merge)
        token_list[idx] = pair_to_merge[0] + pair_to_merge[1]
        idx += 1

    return token_list, merges