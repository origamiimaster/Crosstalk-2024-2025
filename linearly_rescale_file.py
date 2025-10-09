# We know that there should be 37 positives in this data. Make it so....

# Shift predictions up and down to ensure only 37 values are above

import numpy as np

def rescale_probabilities(filename, target_count=40, threshold=0.5):
    # Read probabilities from file
    with open(filename, 'r') as f:
        probs = np.array([float(line.strip()) for line in f])
    
    # Rank and sort
    sorted_indices = np.argsort(probs)[::-1]  # Sort descending
    sorted_probs = probs[sorted_indices]
    
    # Define new thresholding scheme
    new_probs = np.zeros_like(sorted_probs)
    new_probs[:target_count] = np.linspace(threshold + 0.01, min(1.0, threshold + 0.5), target_count)  # Ensure > 0.5
    new_probs[target_count:] = np.linspace(max(0.0, threshold - 0.5), threshold - 0.01, len(sorted_probs) - target_count)
    
    # Ensure values remain within [0,1]
    new_probs = np.clip(new_probs, 0.0, 1.0)
    
    # Reassign probabilities maintaining original order
    rescaled_probs = np.zeros_like(probs)
    rescaled_probs[sorted_indices] = new_probs
    
    # Write back to file
    with open('rescaled_' + filename, 'w') as f:
        for prob in rescaled_probs:
            f.write(f"{prob}\n")
    
    print(f"Rescaled probabilities saved to rescaled_{filename}")

if __name__ == "__main__":
    rescale_probabilities("screen_results.txt")