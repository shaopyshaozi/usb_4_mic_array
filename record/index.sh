python - <<'PY'
import pyaudio

p = pyaudio.PyAudio()

print("PyAudio device count:", p.get_device_count())

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(
        f"Index {i}: "
        f"name={info['name']!r}, "
        f"inputs={int(info['maxInputChannels'])}, "
        f"outputs={int(info['maxOutputChannels'])}, "
        f"default_rate={info['defaultSampleRate']}"
    )

p.terminate()
PY