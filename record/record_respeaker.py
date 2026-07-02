import math
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pyaudio
import usb.core
from tuning import Tuning


RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6
OUTPUT_CHANNELS = 4
RESPEAKER_WIDTH = 2
RESPEAKER_INDEX = 1
CHUNK = 1024

DOA_MEASURE_SECONDS = 3.0
DOA_SAMPLE_INTERVAL = 0.3

OUTPUT_DIR = Path(__file__).resolve().parents[1] /"Respeaker_recordings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_source_count():
    while True:
        source_count = input("Input the number of sound sources (1, 2, or 3): ").strip()
        if source_count in {"1", "2", "3"}:
            return int(source_count)
        print("Please input 1, 2, or 3.")


def circular_average_degrees(angles):
    radians = np.deg2rad(angles)
    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))

    if math.isclose(mean_sin, 0.0, abs_tol=1e-12) and math.isclose(mean_cos, 0.0, abs_tol=1e-12):
        return None

    return int(round(math.degrees(math.atan2(mean_sin, mean_cos)) % 360))


def measure_doa(mic_tuning, sound_index):
    input(f"Prepare sound {sound_index}, then press Enter to measure DoA for 3 seconds...")
    print(f"Measuring DoA for sound {sound_index}...")

    angles = []
    failed_reads = 0
    end_time = time.monotonic() + DOA_MEASURE_SECONDS

    while time.monotonic() < end_time:
        try:
            direction = mic_tuning.direction
        except usb.core.USBError as exc:
            failed_reads += 1
            print(f"DoA read failed, retrying... ({exc})")
            time.sleep(DOA_SAMPLE_INTERVAL)
            continue

        if direction is not None:
            angles.append(float(direction) % 360)
            print(direction)
        time.sleep(DOA_SAMPLE_INTERVAL)

    if not angles:
        raise RuntimeError(
            f"No valid DoA samples were measured for sound {sound_index}. "
            f"All {failed_reads} USB read attempt(s) failed. Try unplugging and reconnecting "
            "the ReSpeaker, then test DOA.py before running this recorder again."
        )

    average_doa = circular_average_degrees(angles)
    if average_doa is None:
        raise RuntimeError(f"Could not calculate a stable average DoA for sound {sound_index}.")

    print(f"Sound {sound_index} average DoA: {average_doa} degrees")
    return average_doa


def get_next_recording_id(output_dir):
    max_id = 0

    for wav_path in output_dir.glob("*.wav"):
        match = re.match(r"^(?:fileid_)?(\d+)_", wav_path.stem)
        if match:
            max_id = max(max_id, int(match.group(1)))

    return max_id + 1


def build_output_path(output_dir, source_count, doas):
    doa_labels = [str(doa) if doa is not None else "NA" for doa in doas]
    recording_id = get_next_recording_id(output_dir)
    return output_dir / f"fileid_{recording_id}_sources_{source_count}_{doa_labels[0]}_{doa_labels[1]}_{doa_labels[2]}.wav"


def record_audio(stream, frames, stop_event):
    print("* recording... Press Enter again to stop.")

    while not stop_event.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)

        audio = np.frombuffer(data, dtype=np.int16)
        audio = audio.reshape(-1, RESPEAKER_CHANNELS)

        # Extract raw MIC1-MIC4 from the ReSpeaker 6-channel stream.
        raw_4ch = audio[:, 1:5]
        frames.append(raw_4ch.astype(np.int16).tobytes())


def save_recording(output_path, frames):
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(OUTPUT_CHANNELS)
        wf.setsampwidth(RESPEAKER_WIDTH)
        wf.setframerate(RESPEAKER_RATE)
        wf.writeframes(b"".join(frames))


def main():
    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if not dev:
        raise RuntimeError("ReSpeaker USB device was not found.")

    mic_tuning = Tuning(dev)
    source_count = get_source_count()

    doas = [None, None, None]
    for sound_index in range(1, source_count + 1):
        doas[sound_index - 1] = measure_doa(mic_tuning, sound_index)

    output_path = build_output_path(OUTPUT_DIR, source_count, doas)
    print(f"Output file will be: {output_path}")

    p = pyaudio.PyAudio()
    stream = None
    frames = []
    stop_event = threading.Event()

    try:
        stream = p.open(
            rate=RESPEAKER_RATE,
            format=p.get_format_from_width(RESPEAKER_WIDTH),
            channels=RESPEAKER_CHANNELS,
            input=True,
            input_device_index=RESPEAKER_INDEX,
            frames_per_buffer=CHUNK,
        )

        input(f"Prepare {source_count} sound source(s), then press Enter to start recording...")

        record_thread = threading.Thread(target=record_audio, args=(stream, frames, stop_event))
        record_thread.start()

        input("Press Enter to stop recording...")

        stop_event.set()
        record_thread.join()

        print("* done recording")
        save_recording(output_path, frames)
        print(f"Recording saved as {output_path}")

    finally:
        stop_event.set()
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()


if __name__ == "__main__":
    main()


