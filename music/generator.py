import numpy as np
import os
import math

# CONFIGURATION
OUTPUT_DIR = "synthetic_songs"
SIZE = 28

def save_song(filename, grid):
    # Flatten 28x28 grid to a single list of 784 numbers
    data = grid.flatten()
    path = os.path.join(OUTPUT_DIR, filename)
    np.savetxt(path, data, fmt='%.4f')
    print(f"Generated: {filename}")

def generate_songs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # A simple diagonal line from bottom-left to top-right
    grid = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        grid[i][i] = 1.0
    save_song("song_0.txt", grid)

    # A diagonal line from top-left to bottom-right
    grid = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        grid[SIZE - 1 - i][i] = 1.0
    save_song("song_1.txt", grid)

    # --- SONG 2: The Sine Wave (Curved Melody) ---
    # A smooth wave that goes up and down
    grid = np.zeros((SIZE, SIZE))
    for col in range(SIZE):
        # Math to calculate height of wave
        row = int(14 + 10 * math.sin(col * 0.3))
        if 0 <= row < SIZE:
            grid[row][col] = 1.0
    save_song("song_2.txt", grid)

    # --- SONG 3: Basic Chords (Blocky) ---
    # Three horizontal lines playing simultaneously
    grid = np.zeros((SIZE, SIZE))
    for col in range(SIZE):
        if col % 4 < 3: # Play for 3 steps, rest for 1
            grid[5][col] = 1.0  # Low note
            grid[12][col] = 0.8 # Mid note
            grid[19][col] = 0.6 # High note
    save_song("song_3.txt", grid)

    # --- SONG 4: The Drum Beat (Vertical Lines) ---
    # A full vertical bar every 4 steps (Kick Drum)
    grid = np.zeros((SIZE, SIZE))
    for col in range(0, SIZE, 4):
        for row in range(10): # Only low frequencies
            grid[row][col] = 1.0
    save_song("song_4.txt", grid)

    # --- SONG 5: The Trill (Zig-Zag) ---
    # Fast alternation between two notes
    grid = np.zeros((SIZE, SIZE))
    for col in range(SIZE):
        if col % 2 == 0:
            grid[15][col] = 1.0
        else:
            grid[17][col] = 1.0
    save_song("song_5.txt", grid)

    # --- SONG 6: The Triangle (Up and Down) ---
    # Goes up for half the song, then down
    grid = np.zeros((SIZE, SIZE))
    for col in range(SIZE):
        if col < SIZE // 2:
            row = col # Going up
        else:
            row = SIZE - 1 - col + (SIZE // 2) # Going down relative to peak
            row += 10 # Offset to keep it in range
        
        if 0 <= row < SIZE:
            grid[row][col] = 1.0
    save_song("song_6.txt", grid)

    # --- SONG 7: Random Jazz (Noise) ---
    # Sparse random notes
    grid = np.zeros((SIZE, SIZE))
    np.random.seed(42) # Keep it consistent
    for _ in range(30): # 30 random notes
        r = np.random.randint(0, SIZE)
        c = np.random.randint(0, SIZE)
        grid[r][c] = np.random.uniform(0.5, 1.0)
    save_song("song_7.txt", grid)

    # --- SONG 8: The Drone (Static) ---
    # One single note held for the entire duration
    grid = np.zeros((SIZE, SIZE))
    grid[14][:] = 1.0 # Middle C held forever
    save_song("song_8.txt", grid)

    # --- SONG 9: High Frequency Chirps ---
    # Only very high notes, short duration
    grid = np.zeros((SIZE, SIZE))
    for col in range(0, SIZE, 3):
        grid[25][col] = 1.0 # High pitch
        if col + 1 < SIZE:
            grid[26][col+1] = 0.5
    save_song("song_9.txt", grid)

if __name__ == "__main__":
    generate_songs()