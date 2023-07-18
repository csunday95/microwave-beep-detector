import pyaudio
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy import fft
from scipy.io import wavfile
import pyqtgraph as pg
from collections import deque

CHUNK = 512
FORMAT = pyaudio.paInt16 # data type formate
CHANNELS = 1 # Adjust to your number of channels
RATE = 48000 # Sample Rate

p = pyaudio.PyAudio()
count = p.get_device_count()
print(count)
for i in range(count):
    dev_info = p.get_device_info_by_index(i)
    print(i, dev_info['name'], dev_info['defaultSampleRate'])

stream = p.open(format=pyaudio.paFloat32, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK, input_device_index=6)
data = b''
freqs = fft.fftfreq(CHUNK, 1 / RATE)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle('data')
p1 = win.addPlot()
# p1.setLogMode(y=True)
# p1.setYRange(0, 3)
curve1 = p1.plot(range(CHUNK))
band = []
power_history = deque([], maxlen=512)

# ax.set_yscale('log')
# plot = ax.plot(range(CHUNK), range(CHUNK))[0]

def update():
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), np.float32)
    dct = np.abs(fft.dct(data, type=2))
    # plot.set_ydata(dct)
    power_history.append(np.sum(dct[10]))
    curve1.setData(np.array(power_history))

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)

if __name__ == '__main__':
    pg.exec()
