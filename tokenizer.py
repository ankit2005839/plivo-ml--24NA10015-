"""Custom Byte-Level BPE (BBPE) Tokenizer.
Fits the interface required by evaluate.py and train.py.
"""
import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(CURRENT_DIR, "vocab.json")


class TrainedBBPE:
    def __init__(self, vocab_file=VOCAB_PATH):
        if os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
            self.merges = {}
            for k, v in data["merges"].items():
                p1, p2 = map(int, k.split(","))
                self.merges[(p1, p2)] = v
        else:
            self.vocab = {i: bytes([i]) for i in range(256)}
            self.merges = {}
        
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pairs = list(zip(ids[:-1], ids[1:]))
            best_pair = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
            if best_pair not in self.merges:
                break
            
            new_id = self.merges[best_pair]
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def decode(self, ids: list[int]) -> str:
        raw_bytes = bytearray()
        for idx in ids:
            if idx in self.vocab:
                raw_bytes.extend(self.vocab[idx])
            else:
                raw_bytes.extend(bytes([idx % 256]))
        return raw_bytes.decode("utf-8", errors="replace")


def train_bpe(text_path, vocab_size=2048, out_path=VOCAB_PATH):
    print(f"Training BBPE tokenizer on {text_path} with target vocab size {vocab_size}...")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    ids = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}
    
    num_merges = vocab_size - 256
    for i in range(num_merges):
        counts = {}
        for pair in zip(ids[:-1], ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        best_pair = max(counts, key=counts.get)
        if counts[best_pair] < 2:
            break
            
        new_id = 256 + i
        merges[best_pair] = new_id
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        new_ids = []
        idx = 0
        while idx < len(ids):
            if idx < len(ids) - 1 and (ids[idx], ids[idx+1]) == best_pair:
                new_ids.append(new_id)
                idx += 2
            else:
                new_ids.append(ids[idx])
                idx += 1
        ids = new_ids
        
    serializable_merges = {f"{k[0]},{k[1]}": v for k, v in merges.items()}
    serializable_vocab = {str(k): list(v) for k, v in vocab.items()}
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"vocab": serializable_vocab, "merges": serializable_merges}, f)
    print(f"Successfully saved trained BBPE metadata to: {out_path}")


def load():
    return TrainedBBPE()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training corpus")
    parser.add_argument("--vocab_size", type=int, default=2048, help="Target vocabulary size")
    args = parser.parse_args()
    train_bpe(args.data, vocab_size=args.vocab_size)
