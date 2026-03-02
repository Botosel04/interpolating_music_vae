import os
import struct
import gzip
import numpy as np
import glob

# CONFIGURATION
INPUT_DIR = "synthetic_songs" 
OUTPUT_DIR = "data_synthetic" 
NUM_VARIATIONS = 200          
NOISE_LEVEL = 0.15            

def write_mnist_format(data, name):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save Image
    header_img = struct.pack('>IIII', 0x00000803, len(data), 28, 28)
    path_img = os.path.join(OUTPUT_DIR, f"{name}-images-idx3-ubyte.gz")
    print(f"Saving {len(data)} items to {path_img}...")
    
    with gzip.open(path_img, 'wb') as f:
        f.write(header_img)
        for vec in data:
            vec_byte = (vec * 255).astype(np.uint8)
            f.write(vec_byte.tobytes())

    # Save Labels
    labels = [0] * len(data)
    header_lbl = struct.pack('>II', 0x00000801, len(labels))
    path_lbl = os.path.join(OUTPUT_DIR, f"{name}-labels-idx1-ubyte.gz")
    with gzip.open(path_lbl, 'wb') as f:
        f.write(header_lbl)
        f.write(bytes(labels))

def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    originals = []
    
    if not files:
        print("Error: No .txt files found!")
        return

    print(f"Found {len(files)} base songs.")
    for f in files:
        vec = np.loadtxt(f)
        originals.append(vec)

    dataset = []

    print("Adding clean originals...")
    for _ in range(25):
        for song in originals:
            dataset.append(song)

    # Add the Noisy Variations (as before)
    print("Adding noisy variations...")
    for _ in range(NUM_VARIATIONS):
        for song in originals:
            noise = np.random.normal(0, NOISE_LEVEL, song.shape)
            remix = song + noise
            remix = np.clip(remix, 0.0, 1.0)
            dataset.append(remix)

    write_mnist_format(dataset, "train")
    write_mnist_format(dataset[:100], "t10k")

    print("Done! Dataset updated.")

if __name__ == "__main__":
    main()