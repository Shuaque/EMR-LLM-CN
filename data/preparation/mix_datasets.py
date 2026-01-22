# -*- coding: utf-8 -*-
import os
import random

def get_b_valid_name(path_b):
    for name in ["test", "dev", "valid"]:
        if os.path.exists(os.path.join(path_b, f"{name}.tsv")):
            return name
    return None

def load_data(tsv_path, ltr_path):
    if not os.path.exists(tsv_path) or not os.path.exists(ltr_path):
        print(f"[WARN] Skipping missing file: {tsv_path}")
        return None, [], []

    with open(tsv_path, 'r', encoding='utf-8') as f:
        root = f.readline().strip()
        tsv_lines = [line.strip() for line in f if line.strip()]
    
    with open(ltr_path, 'r', encoding='utf-8') as f:
        ltr_lines = [line.strip() for line in f if line.strip()]

    # Simple alignment check
    if len(tsv_lines) != len(ltr_lines):
        print(f"[WARN] Line count mismatch: {tsv_path} ({len(tsv_lines)}) vs LTR ({len(ltr_lines)})! Truncating.")
    
    min_len = min(len(tsv_lines), len(ltr_lines))
    return root, tsv_lines[:min_len], ltr_lines[:min_len]

def make_absolute(root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)

def merge_and_align(split_type, output_dir):
    # --- Basic Configuration ---
    path_a = "/path/to/general_dataset"   # Replace with actual path for Dataset A
    path_b = "/path/to/medical_dataset"   # Replace with actual path for Dataset B
    
    # Determine corresponding split for B
    if split_type == "train":
        b_split = "train"
        sample_rate_a = 0.2 
    else:
        b_split = get_b_valid_name(path_b)
        sample_rate_a = 1.0 
    
    # 1. Load data
    root_a, tsv_a, ltr_a = load_data(f"{path_a}/{split_type}.tsv", f"{path_a}/{split_type}.ltr")
    root_b, tsv_b, ltr_b = load_data(f"{path_b}/{b_split}.tsv", f"{path_b}/{b_split}.ltr")

    final_data = []

    # 3. Process A (General Data)
    # A format: ID | Video | Audio | Frames | Samples
    # Target format: ID | NA | Video | Audio | Frames | Samples
    if tsv_a:
        combined_a = list(zip(tsv_a, ltr_a))
        sampled_a = random.sample(combined_a, int(len(combined_a) * sample_rate_a))
        
        for t_line, l_line in sampled_a:
            t_parts = t_line.split('\t')
            if len(t_parts) < 5: continue # Skip bad lines

            # Construct absolute paths
            v_abs = make_absolute(root_a, t_parts[1])
            a_abs = make_absolute(root_a, t_parts[2])
            
            # Construct new TSV line
            new_tsv = f"{t_parts[0]}\tNA\t{v_abs}\t{a_abs}\t{t_parts[3]}\t{t_parts[4]}"
            
            # Process LTR (A: id, text) -> (B: id, unknown, text)
            l_parts = l_line.split('\t', 1) 
            if len(l_parts) < 2: continue
            new_ltr = f"{l_parts[0]}\tunknown\t{l_parts[1]}"
            
            final_data.append((new_tsv, new_ltr))

    # 4. Process B (Medical Data)
    # B format: ID | Role | Video | Audio | Frames | Samples
    if tsv_b:
        for t_line, l_line in zip(tsv_b, ltr_b):
            t_parts = t_line.split('\t')
            if len(t_parts) < 6: continue # Skip bad lines

            # Construct absolute paths (Assuming Video is col 2, Audio is col 3)
            v_abs = make_absolute(root_b, t_parts[2])
            a_abs = make_absolute(root_b, t_parts[3])
            
            # Reassemble, ensuring paths are absolute
            new_tsv = f"{t_parts[0]}\t{t_parts[1]}\t{v_abs}\t{a_abs}\t{t_parts[4]}\t{t_parts[5]}"
            
            # B's LTR is already 3 columns (id, role, text), use directly
            final_data.append((new_tsv, l_line))

    # 5. Shuffle and save
    random.shuffle(final_data)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/{split_type}.tsv", 'w', encoding='utf-8') as f_tsv, \
         open(f"{output_dir}/{split_type}.ltr", 'w', encoding='utf-8') as f_ltr:
        
        f_tsv.write("/\n") # Root fixed to root directory
        for t, l in final_data:
            f_tsv.write(t + "\n")
            f_ltr.write(l + "\n")

    print(f"Done [{split_type}]: A({len(sampled_a) if tsv_a else 0}) + B({len(tsv_b) if tsv_b else 0}) -> Total {len(final_data)}")

if __name__ == "__main__":
    random.seed(42)
    OUT_DIR = "./data/Merged_Data_A_B" # Replace with your output path
    merge_and_align("train", OUT_DIR)
    merge_and_align("valid", OUT_DIR)
    merge_and_align("test", OUT_DIR)