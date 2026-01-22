import argparse
import torch
import os
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Build CTC sub-vocabulary from multiple label files")
    # Allow multiple file paths via nargs='+'
    parser.add_argument("--label-paths", nargs='+', required=True, help="List of label files (e.g. train.ltr valid.ltr)")
    parser.add_argument("--llm-path", type=str, required=True, help="Path to the LLM model/tokenizer")
    parser.add_argument("--output", type=str, default="ctc_vocab.pt", help="Output path for the .pt file")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum frequency to keep a token")

    args = parser.parse_args()

    # 1. Load Tokenizer
    print(f"[1/4] Loading tokenizer from: {args.llm_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    token_counter = Counter()
    total_lines = 0

    # 2. Iterate through all input files
    for label_path in args.label_paths:
        print(f"[2/4] Processing file: {label_path}")
        if not os.path.exists(label_path):
            print(f"  Warning: File not found, skipping: {label_path}")
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc=f"Reading {os.path.basename(label_path)}"):
            line = line.strip()
            if not line:
                continue
            
            # Critical: Split line to extract text content
            # line.split(maxsplit=1) splits into [ID, Remainder]
            parts = line.split(maxsplit=1)
            
            if len(parts) < 2:
                # Skip lines with only ID or empty lines
                continue
            
            # Extract the last column as text content (handling tab separation in the remainder)
            text_content = parts[-1].strip().split('\t')[-1]

            # Tokenize: add_special_tokens=False to count content tokens only
            ids = tokenizer.encode(text_content, add_special_tokens=False)
            token_counter.update(ids)
            total_lines += 1

    print(f"\nProcessed {total_lines} lines across {len(args.label_paths)} files.")
    print(f"Total unique tokens found: {len(token_counter)}")

    # 3. Filter and Sort
    print(f"[3/4] Filtering tokens (min_count={args.min_count})...")
    
    # Filter by frequency
    valid_ids = [token_id for token_id, count in token_counter.items() if count >= args.min_count]
    
    # Sort (CTC IDs must be deterministic)
    valid_ids.sort()
    
    print(f"Tokens kept: {len(valid_ids)}")
    
    # 4. Build Map
    # index 0: -1 (CTC Blank)
    # index 1..K: LLM Token IDs
    ctc_map = [-1] + valid_ids
    
    print(f"[4/4] Saving map to: {args.output}")
    torch.save(ctc_map, args.output)
    
    print("-" * 30)
    print(f"CTC Vocab Size: {len(ctc_map)} (including blank)")
    print(f"Path: {os.path.abspath(args.output)}")
    print("-" * 30)

if __name__ == "__main__":
    main()