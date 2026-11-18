"""
Record a short clip from your microphone and save it as a WAV file.

Usage:
    python ai/audio/record_audio.py

Records for DURATION_SECONDS at SAMPLE_RATE Hz, mono, and saves to
data/samples/test_recording.wav
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os

# ----- Settings -----
SAMPLE_RATE = 16000   # 16kHz: the rate Whisper expects.
RECORD_CHANNELS = 2   # Recorded in stereo, then downmixed to mono after
                       # capture, since some Windows audio drivers (WDM-KS)
                       # reject mono requests outright.
DURATION_SECONDS = 10 
OUTPUT_PATH = os.path.join("data", "samples", "test_recording.wav")

# We look up the microphone BY NAME rather than hardcoding a device index.
# Windows can re-enumerate (renumber) audio devices between sessions -
# e.g. after a driver refresh or reboot - so a hardcoded index like
# "device 8" can silently point at a completely different device later.
INPUT_DEVICE_NAME_HINT = "Microphone"
PREFERRED_DRIVER_HINT = "MME"  # MME is more forgiving about channel counts
                                 # than WDM-KS, which rejects any request
                                 # that doesn't exactly match the hardware.


def find_input_device():
    """
    Search all audio devices for one whose name contains our hint text
    and has at least one input channel. Prefer MME driver devices since
    they're more flexible about channel/rate requests.
    """
    devices = sd.query_devices()
    candidates = [
        (idx, d) for idx, d in enumerate(devices)
        if INPUT_DEVICE_NAME_HINT.lower() in d["name"].lower()
        and d["max_input_channels"] > 0
    ]

    if not candidates:
        raise RuntimeError(
            f"No input device found matching '{INPUT_DEVICE_NAME_HINT}'. "
            f"Run sd.query_devices() to see what's available."
        )

    # Prefer an MME device if one exists among the candidates
    for idx, d in candidates:
        if PREFERRED_DRIVER_HINT.lower() in d["name"].lower() or \
           "mme" in sd.query_hostapis(d["hostapi"])["name"].lower():
            print(f"Using device {idx}: {d['name']} "
                  f"({sd.query_hostapis(d['hostapi'])['name']})")
            return idx

    # Otherwise just take the first match
    idx, d = candidates[0]
    print(f"Using device {idx}: {d['name']} "
          f"({sd.query_hostapis(d['hostapi'])['name']})")
    return idx


def record_audio():
    input_device = find_input_device()

    print(f"Recording for {DURATION_SECONDS} seconds at {SAMPLE_RATE} Hz, "
          f"{RECORD_CHANNELS} channel(s) (will downmix to mono)...")
    print("Speak now!")

    # sd.rec() starts recording immediately and returns a numpy array
    # once finished. Shape will be (num_samples, channels).
    audio_data = sd.rec(
        int(DURATION_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=RECORD_CHANNELS,
        dtype="int16",   # 16-bit depth, as discussed
        device=input_device,
    )
    sd.wait()  # block until recording is finished

    print("Recording finished.")

    # Downmix stereo -> mono by averaging the two channels together.
    # audio_data shape is (num_samples, 2) right now; we want (num_samples,)
    # We convert to a wider int type first so the addition doesn't overflow
    # int16's range, then cast back down to int16 after averaging.
    mono_audio = audio_data.astype(np.int32).mean(axis=1).astype(np.int16)

    # Ensure the output folder exists before writing
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    sf.write(OUTPUT_PATH, mono_audio, SAMPLE_RATE)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    record_audio()

"""
Retrieval-Augmented Generation (RAG) is a way to make AI answers more reliable by combining searching for relevant 
information and then generating a response. Instead of guessing based only on old training data, it first finds useful data from external 
sources (like documents or databases) and then uses it to give a better answer.

Could you explain the complete end-to-end manufacturing workflow of this production line, starting from raw-material handling 
and material feeding, through CNC machining, automated assembly, robotic material handling, conveyor-based transportation, welding and
joining operations, and finally quality inspection and finished-product packaging? I would also like to understand how the PLC-based
control architecture, SCADA system, industrial IoT sensors, servo motors, pneumatic and hydraulic actuators, variable-frequency drives,
programmable logic controllers, robotic arms, proximity sensors, photoelectric sensors, temperature and pressure transmitters, 
and machine-vision systems are integrated into the production process. Additionally, how does the factory monitor Overall Equipment Effectiveness (OEE), 
 cycle time, takt time, throughput, downtime, scrap rate, first-pass yield, and process capability, and how are predictive maintenance, 
condition monitoring, vibration analysis, anomaly detection, and real-time telemetry used to identify equipment degradation and prevent unplanned
machine failures? Finally, could you explain how the Manufacturing Execution System (MES), Enterprise Resource Planning (ERP), 
Digital Twin, Industrial Internet of Things (IIoT), edge computing, cloud infrastructure, data acquisition systems, 
and AI-based analytics work together to optimize production efficiency, energy consumption, product traceability, quality control, and overall operational performance
"""