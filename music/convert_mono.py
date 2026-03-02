import numpy as np
from midiutil import MIDIFile

INPUT_FILE = "latent_reconstruction_beta_2.txt"
OUTPUT_FILE = "Morph0-9/all_morphed.mid"

def main():
    print("Reading data...")
    try:
        data = np.loadtxt(INPUT_FILE)
    except:
        print("Error: File not found.")
        return

    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    grids = data.reshape(-1, 28, 28)
   
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 140)
    current_time = 0

    print("Generating clean melody...")
    for grid in grids:
        grid = np.flip(grid, axis=0) # Flip pitch
        
        for col in range(28):
            # Pick ONLY the single loudest pixel in this column
            col_data = grid[:, col]
            loudest_row = np.argmax(col_data)
            brightness = col_data[loudest_row]
            
            # Only play if it's actually somewhat bright
            if brightness > 0.4: 
                pitch = 48 + loudest_row
                velocity = 100
                duration = 0.25
                midi.addNote(0, 0, pitch, current_time + (col*0.25), duration, velocity)
        
        current_time += 7.0 # Move strictly to next bar

    with open(OUTPUT_FILE, "wb") as f:
        midi.writeFile(f)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()