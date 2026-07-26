import time

import numpy as np
import pyaudio

RATE = 16000
CHANNELS = 6
WIDTH = 2
CHUNK = 1024
SECONDS = 3


def channel_dbfs(audio_int16):
    audio = audio_int16.astype(np.float64) / 32768.0
    audio = audio - audio.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(audio**2, axis=0) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def main():
    p = pyaudio.PyAudio()
    try:
        host_api_names = {
            i: p.get_host_api_info_by_index(i)["name"]
            for i in range(p.get_host_api_count())
        }

        print("Host APIs:")
        for i, name in host_api_names.items():
            print(f"  {i}: {name}")

        print("\nInput devices:")
        candidates = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] <= 0:
                continue
            host_api = int(info["hostApi"])
            print(
                f"  index={i}, host_api={host_api_names.get(host_api, host_api)}, "
                f"name={info['name']}, input_channels={info['maxInputChannels']}, "
                f"default_rate={info['defaultSampleRate']}"
            )
            if "respeaker" in info["name"].lower() and info["maxInputChannels"] >= CHANNELS:
                candidates.append(i)

        print("\nTesting ReSpeaker 6-channel candidates at 16 kHz:")
        for index in candidates:
            info = p.get_device_info_by_index(index)
            host_api = host_api_names.get(int(info["hostApi"]), info["hostApi"])
            print(f"\nindex={index}, host_api={host_api}, name={info['name']}")
            try:
                stream = p.open(
                    rate=RATE,
                    format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    input=True,
                    input_device_index=index,
                    frames_per_buffer=CHUNK,
                )
            except Exception as exc:
                print(f"  OPEN FAILED: {exc}")
                continue

            blocks = []
            start = time.monotonic()
            try:
                for _ in range(round(RATE * SECONDS / CHUNK)):
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    expected_bytes = CHUNK * CHANNELS * WIDTH
                    if len(data) != expected_bytes:
                        print(f"  BAD READ SIZE: expected {expected_bytes}, got {len(data)}")
                        break
                    blocks.append(np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS))
            except Exception as exc:
                print(f"  READ FAILED: {exc}")
            finally:
                stream.stop_stream()
                stream.close()

            elapsed = time.monotonic() - start
            if not blocks:
                print("  no blocks captured")
                continue

            audio = np.vstack(blocks)
            dbfs = channel_dbfs(audio)
            print(f"  captured_audio_seconds={audio.shape[0] / RATE:.2f}, elapsed_seconds={elapsed:.2f}")
            print("  dBFS: " + ", ".join(f"ch{ch}: {value:.2f}" for ch, value in enumerate(dbfs)))
            print("  peaks: " + ", ".join(str(int(v)) for v in np.max(np.abs(audio), axis=0)))
    finally:
        p.terminate()


if __name__ == "__main__":
    main()
