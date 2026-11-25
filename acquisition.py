"""Optional BrainFlow source (synthetic board needs no hardware). Resamples to 128 Hz, 4 channels."""
import time
import numpy as np
from scipy.signal import resample_poly
from .core.signal import FS, N, PAD


class BrainFlowSource:
    def __init__(self, board_id=-1, serial_port=None):
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("brainflow is not installed (pip install brainflow)") from e
        params = BrainFlowInputParams()
        if serial_port:
            params.serial_port = serial_port
        self.board_id = board_id
        self.fs = BoardShim.get_sampling_rate(board_id)
        self.channels = BoardShim.get_eeg_channels(board_id)[:4]
        self.board = BoardShim(board_id, params)
        self.board.prepare_session()
        self.board.start_stream()
        self.need = int(np.ceil((N + PAD) * self.fs / FS)) + 8

    def get_window(self):
        data = self.board.get_current_board_data(self.need)
        while data.shape[1] < self.need:
            time.sleep(0.05)
            data = self.board.get_current_board_data(self.need)
        x = resample_poly(data[self.channels], FS, self.fs, axis=1)
        return x[:, -(N + PAD):]

    def close(self):
        try:
            self.board.stop_stream()
            self.board.release_session()
        except Exception:  # pragma: no cover
            pass
