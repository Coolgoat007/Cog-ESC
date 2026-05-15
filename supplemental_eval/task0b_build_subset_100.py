"""Build 100-example subset for Task 3 (reduced scope)."""
import random
import os

SEED = 42
SUBSET_200_PATH = os.path.join(os.path.dirname(__file__), "subset_200_seed42.txt")
OUT_PATH = os.path.join(os.path.dirname(__file__), "subset_100_seed42.txt")

def build_subset_100():
    # Load the 200 subset
    with open(SUBSET_200_PATH) as f:
        indices_200 = [int(line.strip()) for line in f if line.strip()]

    # Sample 100 from the 200
    rng = random.Random(SEED)
    selected = sorted(rng.sample(indices_200, 100))

    with open(OUT_PATH, "w") as f:
        for idx in selected:
            f.write(f"{idx}\n")

    print(f"Saved {len(selected)} indices to {OUT_PATH}")
    print(f"First 10: {selected[:10]}")
    print(f"Last 10:  {selected[-10:]}")
    return selected

if __name__ == "__main__":
    build_subset_100()
