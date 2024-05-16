"""Audio Processing Application.

This module provides a PyQt5 GUI application for loading, filtering, and transforming audio files.
"""

import sys

import librosa
import numpy as np
import pyqtgraph as pg
import soundfile as sf
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import butter, lfilter


class AudioApp(QMainWindow):
    """Main window for the audio processing application."""

    def __init__(self):
        """Initialize the audio processing application."""
        super().__init__()
        self.setWindowTitle("Audio Processing App")
        self.setGeometry(0, 0, 969, 526)
        self.audio_data = None
        self.filtered_data = None
        self.sample_rate = None
        self.initUI()

    def initUI(self):
        """Set up the user interface for the application."""
        main_layout = QHBoxLayout()
        controls_layout = QVBoxLayout()
        graphs_layout = QHBoxLayout()

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        # Load file button
        self.load_button = QPushButton("Cargar archivo")
        self.load_button.clicked.connect(self.load_file)
        controls_layout.addWidget(self.load_button)

        # Filter type selector
        self.filter_selector = QComboBox()
        self.filter_selector.addItems(["Pasa-bajas", "Pasa-altas", "Pasa-banda"])
        controls_layout.addWidget(self.filter_selector)

        # Frequency slider and label
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setMinimum(1)
        self.freq_slider.setMaximum(5000)
        self.freq_slider.setValue(1000)
        self.freq_slider.setTickInterval(100)
        self.freq_slider.setTickPosition(QSlider.TicksBelow)
        controls_layout.addWidget(self.freq_slider)
        self.freq_label_text = "Frecuencia de corte: "
        self.freq_value = 1000
        self.freq_value_label = QLabel(f"{self.freq_label_text}{self.freq_value}", self)
        controls_layout.addWidget(self.freq_value_label)
        self.freq_slider.valueChanged.connect(self.updateFreqLabel)

        # Low frequency slider and label
        self.low_freq_slider = QSlider(Qt.Horizontal)
        self.low_freq_slider.setMinimum(1)
        self.low_freq_slider.setMaximum(5000)
        self.low_freq_slider.setValue(500)
        self.low_freq_slider.setTickInterval(100)
        self.low_freq_slider.setTickPosition(QSlider.TicksBelow)
        controls_layout.addWidget(self.low_freq_slider)
        self.low_freq_label_text = "Frecuencia de corte baja: "
        self.low_freq_value_label = QLabel(
            f"{self.low_freq_label_text}{self.low_freq_slider.value()}", self
        )
        controls_layout.addWidget(self.low_freq_value_label)
        self.low_freq_slider.valueChanged.connect(self.updateLowFreqLabel)

        # Order slider and label
        self.order_slider = QSlider(Qt.Horizontal)
        self.order_slider.setMinimum(1)
        self.order_slider.setMaximum(7)
        self.order_slider.setValue(1)
        self.order_slider.setTickInterval(1)
        self.order_slider.setTickPosition(QSlider.TicksBelow)
        controls_layout.addWidget(self.order_slider)
        self.order_value_label = QLabel("Orden de filtro: 1", self)
        controls_layout.addWidget(self.order_value_label)
        self.order_slider.valueChanged.connect(self.updateOrderLabel)

        # Apply filter button
        self.apply_filter_button = QPushButton("Aplicar Filtro")
        self.apply_filter_button.clicked.connect(self.apply_filter)
        controls_layout.addWidget(self.apply_filter_button)

        # Apply transform button
        self.transform_button = QPushButton("Aplicar Transformada")
        self.transform_button.clicked.connect(self.apply_transform)
        controls_layout.addWidget(self.transform_button)

        # Save result button
        self.save_button = QPushButton("Guardar Resultado")
        self.save_button.clicked.connect(self.save_result)
        controls_layout.addWidget(self.save_button)

        main_layout.addLayout(controls_layout)

        # Visualization widgets
        self.original_plot_widget = pg.PlotWidget(title="Señal Original")
        self.processed_plot_widget = pg.PlotWidget(title="Señal Procesada")
        graphs_layout.addWidget(self.original_plot_widget)
        graphs_layout.addWidget(self.processed_plot_widget)
        main_layout.addLayout(graphs_layout)

    def load_file(self):
        """Load an audio file and plot its waveform."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav *.mp3 *.aac)"
        )
        if file_name:
            self.audio_data, self.sample_rate = librosa.load(file_name, sr=None)
            self.plot_audio(self.original_plot_widget, self.audio_data)

    def updateOrderLabel(self, value):
        """Update the order label with the current slider value."""
        description_text = "Orden del filtro: "
        self.order_value_label.setText(f"{description_text}{value}")

    def updateLowFreqLabel(self, value):
        """Update the low frequency label with the current slider value."""
        self.low_freq_value_label.setText(f"{self.low_freq_label_text}{value}")

    def updateFreqLabel(self, value):
        """Update the frequency label with the current slider value."""
        description_text = "Frecuencia de corte: "
        self.freq_value_label.setText(f"{description_text}{value}")

    def plot_audio(self, plot_widget, data):
        """Plot audio data on the specified plot widget."""
        plot_widget.clear()
        plot_widget.plot(data)

    def apply_filter(self):
        """Apply the selected filter to the loaded audio data."""
        filter_mappings = {
            "pasabajas": "lowpass",
            "pasaaltas": "highpass",
            "pasabanda": "bandpass",
        }
        if self.audio_data is None:
            return
        order = self.order_slider.value()
        high_cutoff = self.freq_slider.value()  # Frecuencia alta de corte
        low_cutoff = self.low_freq_slider.value()  # Frecuencia baja de corte
        selected_text = self.filter_selector.currentText().lower().replace("-", "")
        btype = filter_mappings.get(selected_text)

        # Ajustar el valor de Wn basado en el tipo de filtro
        if btype == "bandpass" or btype == "bandstop":
            Wn = [
                low_cutoff / (0.5 * self.sample_rate),
                high_cutoff / (0.5 * self.sample_rate),
            ]
        else:
            Wn = high_cutoff / (0.5 * self.sample_rate)

        b, a = butter(order, Wn, btype=btype)
        self.filtered_data = lfilter(b, a, self.audio_data)
        self.plot_audio(self.processed_plot_widget, self.filtered_data)

    def apply_transform(self):
        """Apply the Fourier transform to the loaded audio data."""
        if self.audio_data is None:
            return
        fft_result = np.abs(np.fft.fft(self.audio_data))
        frequencies = np.fft.fftfreq(len(fft_result), 1 / self.sample_rate)
        self.processed_plot_widget.clear()
        self.processed_plot_widget.plot(
            frequencies[: len(frequencies) // 2], fft_result[: len(fft_result) // 2]
        )

    def save_result(self):
        """Save the filtered audio data to a file."""
        sf.write("filtered_audio.wav", self.filtered_data, self.sample_rate, "PCM_24")


app = QApplication(sys.argv)
window = AudioApp()
window.show()
sys.exit(app.exec_())
