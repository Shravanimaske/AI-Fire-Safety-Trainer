"""
Add Controlled Synthetic Noise to clean the audion sample 

This gives us a known "Before" State so we can objectively measure how 
much a noise-suppresion algorithm improves things afterward (via SNR),
rather than just guessing from listening 

"""

import os 
import sys 
import numpy as np 
import soundfile as sf 

def compute_snr(clean, noisy):
    """Signal to Noise Ratio in decibels(db)
    We estimate the noise as (noisy-clean), then compare the power of the original 
    to the power that estimated noise. Higher db = cleaner relative to the noise 
    
    """


    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise **2)

    if noise_power ==0:
        return float("int")

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db 

def add_noise(path, noise_level = 0.05):
    data, sample_rate = sf.read(path)
    if data.ndim >1:
        data = data[:,0]

    noise = np.random.normal(0, noise_level, data.shape)
    noisy_data = data + noise 

    noisy_data = np.clip(noisy_data, -1.0,1.0)

    snr = compute_snr(data, noisy_data)
    print(f"Added synthetic noise (level = {noise_level})")
    print(f"Resulting SNR: {snr:.2f} dB")

    base, ext = os.path.splitext(path)
    output_path = f"{base}_noisy{ext}"

    sf.write(output_path, noisy_data, sample_rate)
    print(f"Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) <2:
        print("Usage:python ai/preprocessing/add_noisy.py <path_to_wav_file> [noise_level]")
        sys.exit(1)

    input_path = sys.argv[1]
    level = float(sys.argv[2]) if len(sys.argv)>2 else 0.05 

    add_noise(input_path, level)

