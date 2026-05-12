import time
import tracemalloc
# from bpe import train_bpe
from cs336_basics.section2.bpe import train_bpe

if __name__ == "__main__":
    input_path = "data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    tracemalloc.start()
    t0 = time.time()

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    elapsed = time.time() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Time:   {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"Memory: {peak_bytes / 1024**3:.2f} GB (peak)")

    # serialize
    with open("cs336_basics/section2/results/vocab.txt", "w") as f:
        for idx, token in vocab.items():
            f.write(f"{idx}\t{token.decode('utf-8', errors='replace')}\n")
    
    with open("cs336_basics/section2/results/merges.txt", "w") as f:
        f.write(f"{'#':<6}\t{'A':<20}\t{'B':<20}\t{'A+B':<20}\n")
        f.write("-" * 70 + "\n")
        for i, (a, b) in enumerate(merges):
            a_str = repr(a.decode('utf-8', errors='replace'))
            b_str = repr(b.decode('utf-8', errors='replace'))
            ab_str = repr((a + b).decode('utf-8', errors='replace'))
            f.write(f"{i+1:<6}\t{a_str:<20}\t{b_str:<20}\t{ab_str:<20}\n")

    # find longest token
    longest_idx, longest_tok = max(vocab.items(), key=lambda x: len(x[1]))
    print(f"Longest token: id={longest_idx}, '{longest_tok.decode('utf-8', errors='replace')}' ({len(longest_tok)} bytes)")
    