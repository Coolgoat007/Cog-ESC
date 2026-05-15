"""TASK 0: Build deterministic 200-example subset with seed=42."""
import random
import os

SEED = 42
N_TOTAL = 682
N_SAMPLE = 200
OUT_PATH = os.path.join(os.path.dirname(__file__), "subset_200_seed42.txt")

def build_subset():
    rng = random.Random(SEED)
    indices = list(range(N_TOTAL))
    selected = sorted(rng.sample(indices, N_SAMPLE))
    with open(OUT_PATH, "w") as f:
        for idx in selected:
            f.write(f"{idx}\n")
    print(f"Saved {len(selected)} indices to {OUT_PATH}")
    print(f"First 10: {selected[:10]}")
    print(f"Last 10:  {selected[-10:]}")
    return selected

if __name__ == "__main__":
    build_subset()
