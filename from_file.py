from __future__ import annotations

import pyaudio 
import numpy as np
import wave
import argparse
import sys
from scipy import fft
from scipy.signal import find_peaks
from collections import deque
import pyqtgraph as pg

CHUNK = 256
# RANGE = [25, 60]
RANGE = [30, 128]
#RANGE_OF_CONCERN = [30, 130]
# (340 - 370)
AVG_LEN = 16

class AudioSignatureDetecter:
    def __init__(self, wav_file_path: str, skip_bytes: int,
                 dct_chunk_size: int, analysis_range: tuple[int, int]):
        self._wav_file_path = wav_file_path
        self._wav_file = wave.open(wav_file_path, 'rb')
        self._wav_file.setpos(skip_bytes)
        self._dct_chunk_size = dct_chunk_size
        self._analysis_range = analysis_range
        self._win = pg.GraphicsLayoutWidget(show=True)
        self._win.setWindowTitle('data')
        self._plot = self._win.addPlot()
        self._curve = self._plot.plot(range(CHUNK))
        self._timer = pg.QtCore.QTimer()
        p = pyaudio.PyAudio()
        self._stream = p.open(
            format=p.get_format_from_width(self._wav_file.getsampwidth()),
            channels=self._wav_file.getnchannels(),
            rate=self._wav_file.getframerate(),
            output=True)
        self._rolling_data = deque([], maxlen=AVG_LEN)
        self._power_history = deque([], maxlen=512)
        self._peak_count_history = deque([], maxlen=10)


    def start(self):
        self._plot.setYRange(0, 10**6)
        self._timer.timeout.connect(self._timer_on_update)
        self._timer.start(1)

    def _timer_on_update(self):
        raw = self._wav_file.readframes(CHUNK)
        if len(raw) < CHUNK:
            return
        data = np.frombuffer(raw, dtype=np.int16)
        dct = np.abs(fft.dct(data))[RANGE[0]:RANGE[1]]
        self._rolling_data.append(dct)
        if len(self._rolling_data) < AVG_LEN:
            return
        self._power_history.append(np.mean(np.vstack(self._rolling_data), axis=0))
        peaks = find_peaks(self._power_history[-1], distance=5, prominence=(50000, None))
        self._peak_count_history.append(len(peaks[0]))
        if min(self._peak_count_history) >= 4:
            print('chiming!')
        self._curve.setData(self._power_history[-1])
        self._stream.write(raw)

def main(args: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--recording-path',
        help='a file to read recorded data from')
    result = vars(parser.parse_args(args))
    recording_path = result['recording_path']
    detector = AudioSignatureDetecter(
        recording_path, skip_bytes=250000, 
        dct_chunk_size=256, analysis_range=(30, 128))
    detector.start()
    pg.exec()
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))