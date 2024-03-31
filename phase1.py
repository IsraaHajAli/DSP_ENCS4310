from tkinter import filedialog
import customtkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import scipy.io.wavfile as wav
import scipy.io.wavfile as wave
from scipy.signal import find_peaks, lfilter , butter
from scipy.io.wavfile import read
import wave

import pyaudio
from scipy.io import wavfile

char_frequencies = {'a': [100, 1100, 2500], 'b': [100, 1100, 3000], 'c': [100, 1100, 3500], 'd': [100, 1300, 2500],
                    'e': [100, 1300, 3000], 'f': [100, 1300, 3500], 'g': [100, 1500, 2500], 'h': [100, 1500, 3000],
                    'i': [100, 1500, 3500], 'j': [300, 1100, 2500], 'k': [300, 1100, 3000], 'l': [300, 1100, 3500],
                    'm': [300, 1300, 2500], 'n': [300, 1300, 3000], 'o': [300, 1300, 3500], 'p': [300, 1500, 2500],
                    'q': [300, 1500, 3000], 'r': [300, 1500, 3500], 's': [500, 1100, 2500], 't': [500, 1100, 3000],
                    'u': [500, 1100, 3500], 'v': [500, 1300, 2500], 'w': [500, 1300, 3000], 'x': [500, 1300, 3500],
                    'y': [500, 1500, 2500], 'z': [500, 1500, 3000], ' ': [500, 1500, 3500]}
def encode_character(char, duration=0.04, fs=44100):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    frequencies = char_frequencies .get(char.lower(), [0, 0, 0])  # Handle uppercase/lowercase

    signal = np.sin(2 * np.pi * frequencies[0] * t) + \
             np.sin(2 * np.pi * frequencies[1] * t) + \
             np.sin(2 * np.pi * frequencies[2] * t)

    signal = 0.5 * signal / np.max(np.abs(signal))
    signal_int16 = (signal * 32767).astype(np.int16)

    return fs, signal_int16

def encode_string(input_string, output_filename="output.wav"):
    fs, final_signal = encode_character(input_string[0])

    # Concatenate signals for each character in the input string
    final_signal = np.concatenate([encode_character(char)[1] for char in input_string])

    # Save the final signal as a WAV file
    wav.write(output_filename, fs, final_signal)
    return final_signal, fs



def clear():
    entry1.delete(0, tk.END)

def clear2():
    entry2.delete(0, tk.END)

def encode_plot(sentence):
    signal, fs = encode_string(sentence, output_filename="output.wav")





    # Plot the signal in time domain
    plt.plot(signal)
    plt.title('Signal in Time Domain')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    # Plot the signal in frequency domain
    plt.magnitude_spectrum(signal, Fs=fs)
    plt.title('Signal in Frequency Domain')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()

def calculate_cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def calculate_distance(freq1, freq2):
    return np.linalg.norm(np.array(freq1) - np.array(freq2))

def decode(filename):

    fs, data = wav.read(filename)


    if len(data.shape) > 1:
        data = data[:, 0]

    data = data - np.mean(data)

    frame_size = int(0.04 * fs)
    num_frames = int(np.ceil(len(data) / frame_size))
    frames = np.zeros((num_frames, frame_size))
    frames[:len(data)] = data.reshape(-1, frame_size)

    chars = []

    for frame in frames:
        fft = np.fft.rfft(frame)
        freqs = np.fft.rfftfreq(frame_size, 1 / fs)

        # Find prominent peaks in the spectrum
        peaks, _ = find_peaks(np.abs(fft), height=1000)

        # Map peaks to frequencies
        detected_freqs = freqs[peaks]

        # Find the character whose frequencies are closest to the detected frequencies
        closest_char = min(char_frequencies, key=lambda char: np.sum(np.abs(np.array(detected_freqs) - np.array(char_frequencies[char]))))
        chars.append(closest_char)

    return ''.join(chars)


def design_bandpass_filters(frequencies, fs):
    filters = {}
    for char, freqs in frequencies.items():
        char_filters = []
        for freq in freqs:
            low = freq - 50  # Adjust for a small margin
            high = freq + 50
            b, a = butter(N=4, Wn=[low / (fs / 2), high / (fs / 2)], btype='band')
            char_filters.append((b, a))
        filters[char] = char_filters
    return filters

def apply_bandpass_filters(signal, filters):
    filtered_signals = {}
    for char, char_filters in filters.items():
        filtered_char_signal = []
        for b, a in char_filters:
            filtered_char_signal.append(lfilter(b, a, signal))
        filtered_signals[char] = np.sum(filtered_char_signal, axis=0)
    return filtered_signals

def decode_fillter (filename):
    fs, data = wav.read(filename)

    if len(data.shape) > 1:
        data = data[:, 0]

    frame_size = int(0.04 * fs)
    num_frames = int(np.ceil(len(data) / frame_size))

    filters = design_bandpass_filters(char_frequencies, fs)

    decoded_chars = []

    for frame_idx in range(num_frames):
        frame_start = frame_idx * frame_size
        frame_end = min((frame_idx + 1) * frame_size, len(data))

        frame = data[frame_start:frame_end]

        filtered_signals = apply_bandpass_filters(frame, filters)

        max_energy_char = max(filtered_signals, key=lambda char: np.sum(np.abs(filtered_signals[char])**2))
        decoded_chars.append(max_energy_char)

    return ''.join(decoded_chars)


def decode_and_display():

    decoded_text = decode(entry2.get())
    entry2.configure(state='normal')
  #open a small window to display the decoded text
    window = tk.CTk()
    window.geometry("400x300")
    window.title("Decoded  Text using FFT")
    frame = tk.CTkFrame(master=window)
    frame.pack(pady=20, padx=60, fill="both", expand=True)
    label = tk.CTkLabel(master=frame, text=decoded_text, font=("Arial", 20))
    label.pack(pady=10, padx=10, anchor="w")
    window.mainloop()
    entry2.configure(state='readonly')
def decode_and_display2():

    decoded_text = decode(entry2.get())
    entry2.configure(state='normal')
  #open a small window to display the decoded text
    window = tk.CTk()
    window.geometry("400x300")
    window.title("Decoded Text using fillter ")
    frame = tk.CTkFrame(master=window)
    frame.pack(pady=20, padx=60, fill="both", expand=True)
    label = tk.CTkLabel(master=frame, text=decoded_text, font=("Arial", 20))
    label.pack(pady=10, padx=10, anchor="w")
    window.mainloop()
    entry2.configure(state='readonly')

tk.set_appearance_mode("dark")
tk.set_widget_scaling(1)
tk.set_default_color_theme("green")


def play_sound():
    filename = 'output.wav'
    chunk = 1024

    # initialize the PyAudio class
    p = pyaudio.PyAudio()

    # open the .wav file in read mode using wave module
    wf = wave.open(filename, 'rb')

    # open stream based on the wave object which has been input.
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)

    # read data (based on the chunk size)
    data = wf.readframes(chunk)

    # play stream (looping from beginning of the file to the end)
    while len(data) > 0:
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(chunk)

    # cleanup stuff.
    stream.close()
    p.terminate()

def browse_file(entry_widget):
    file_path = filedialog.askopenfilename()
    entry_widget.configure(state='normal')
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, file_path)
    entry_widget.configure(state='readonly')

root = tk.CTk()
root.geometry("800x700")
root.title("Encoder/Decoder GUI")

frame = tk.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)
label = tk.CTkLabel(master=frame, text="Welcome to the Encoder/Decoder", font=("Arial", 20))
label.pack(pady=10, padx=10, anchor="w")

label_entry1 = tk.CTkLabel(master=frame, text="Sentence to be encoded:", font=("Arial", 16))
label_entry1.pack(pady=5, padx=10, anchor="w")

entry1 = tk.CTkEntry(master=frame, placeholder_text="Enter the sentence to be encoded: ", width=300)
entry1.pack(pady=12, padx=15)
button1 = tk.CTkButton(master=frame, text="Encode", command=lambda: encode_plot(entry1.get()))
button1.pack(pady=12, padx=5)

sound_button = tk.CTkButton(master=frame, text="Sound", command=play_sound)
sound_button.pack(pady=12, padx=5)
clear_button = tk.CTkButton(master=frame, text="Clear", command=clear)
clear_button.pack(pady=12, padx=5)

label_entry2 = tk.CTkLabel(master=frame, text="Sound to be decoded:", font=("Arial", 16))
label_entry2.pack(pady=5, padx=10, anchor="w")

entry2 = tk.CTkEntry(master=frame, placeholder_text="WAV file to be decoded: ", width=300)
entry2.configure(state='readonly')
entry2.pack(pady=12, padx=30)

browse_button2 = tk.CTkButton(master=frame, text="Browse", command=lambda: browse_file(entry2))
browse_button2.pack(pady=5, padx=10)

button2 = tk.CTkButton(master=frame, text="Decode by frequency analysis", command=decode_and_display)
button2.pack(pady=12, padx=10)

button3 = tk.CTkButton(master=frame, text="Decode by using filler", command=decode_and_display2)
button3.pack(pady=12, padx=10)
clear_button2 = tk.CTkButton(master=frame, text="Clear", command=clear2)
clear_button2.pack(pady=12, padx=5)

root.mainloop()