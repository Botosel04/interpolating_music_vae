import numpy as np
from midiutil import MIDIFile

# CONFIGURATION
INPUT_FILE = "synthetic_songs/song_9.txt"
OUTPUT_FILE = "song9.mid"
TEMPO = 140

def main():
    print(f"--- DIAGNOSTIC MODE ---")
    try:
        data = np.loadtxt(INPUT_FILE)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 1. DIAGNOSE: What are the actual values?
    max_val = np.max(data)
    min_val = np.min(data)
    avg_val = np.mean(data)
    
    print(f"Data Stats:")
    print(f"  > Max Brightness: {max_val:.4f}")
    print(f"  > Min Brightness: {min_val:.4f}")
    print(f"  > Average:        {avg_val:.4f}")

    if max_val == 0:
        print("CRITICAL ERROR: The text file contains only Zeros.")
        print("Your VAE has suffered 'Model Collapse' or didn't train.")
        return

    # 2. THE FIX: Normalize (Auto-Volume)
    # We stretch the values so the quietest pixel is 0.0 and the loudest is 1.0
    print("Normalizing data range to 0.0 - 1.0...")
    normalized_data = (data - min_val) / (max_val - min_val)
    
    # Use a dynamic threshold based on the average
    # Everything above average brightness becomes a note
    threshold = np.mean(normalized_data) * 1.2
    print(f"Using calculated threshold: {threshold:.4f}")

    # 3. GENERATE MIDI
    grids = normalized_data.reshape(-1, 28, 28)
    midi = MIDIFile(1)
    midi.addTempo(track=0, time=0, tempo=TEMPO)

    current_time = 0
    note_count = 0

    for grid in grids:
        grid = np.flip(grid, axis=0) # Flip so low pitch is bottom

        for col in range(28):
            for row in range(28):
                brightness = grid[row][col]
                
                if brightness > threshold:
                    pitch = 48 + row       # 48 = C3
                    velocity = int(brightness * 127)
                    duration = 0.25
                    
                    midi.addNote(0, 0, pitch, current_time + (col * 0.25), duration, velocity)
                    note_count += 1
        
        current_time += 7.0 # Move forward 28 beats * 0.25

    # 4. SAVE
    with open(OUTPUT_FILE, "wb") as f:
        midi.writeFile(f)
    
    print(f"--- SUCCESS ---")
    print(f"Generated {note_count} notes.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()