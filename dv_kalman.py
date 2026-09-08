import time
from pathlib import Path

import adi
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

URI = "ip:169.254.92.202"
# Select "FMCOMMS3" or "FMCOMMS5". Both use chip-A 2TX/2RX here.
SDR_TYPE = "FMCOMMS5"

FC_HZ = int(2.45e9)
FS_HZ = int(30.72e6)
RF_BW_HZ = int(25e6)

RX_CHANNELS = [0, 1]       # chip-A RX0 and RX1
TX_CHANNELS = [0, 1]       # chip-A TX0 and TX1

RX_GAIN_DB = [40, 40]
TX_GAIN_DB = [-25, -25]

# Two exact-bin tones: +/- Fs/4.
# They are exact DFT bins for every sample count divisible by 4.
F1_OFFSET_HZ = -FS_HZ / 4.0
F2_OFFSET_HZ = +FS_HZ / 4.0
DELTA_F_HZ = F2_OFFSET_HZ - F1_OFFSET_HZ

CALIBRATION_RANGE_M = 0.50
RANGE_SIGN = -1.0

# Digital waveform scale. Keep headroom below the signed 14-bit peak.
TX_SCALE = 2 ** 13

# Continuous live waveform and RX capture length.
LIVE_N = 16384

COUPLING_SLOT_N = 8192
COUPLING_LONG_GUARD_N = 2048
COUPLING_SHORT_GUARD_N = 512
COUPLING_EDGE_SKIP_N = 512
COUPLING_RX_BUFFER_N = 65536
COUPLING_CAPTURES = 24

BACKGROUND_FRAMES = 80
CALIBRATION_FRAMES = 80
LIVE_FRAMES = 12

SETTLE_TIME_SEC = 2.0
FLUSH_COUNT = 30

# Display smoothing: 0 means no memory, 0.9 means strong smoothing.
RANGE_EMA_ALPHA = 0.80

VELOCITY_EMA_ALPHA = 0.70

MIN_ECHO_TO_BG_DB = -35.0
MAX_RX_DISAGREEMENT_M = 0.25

# Antenna-array phase error diagnostics (Stage 7).
# ANTENNA_ERROR_EMA_ALPHA controls how quickly the tracked per-RX error
# reacts to new disagreement between that RX and the fused estimate.
ANTENNA_ERROR_EMA_ALPHA = 0.90
MAX_ANTENNA_ERROR_M = 0.15

# Kalman filter process noise: how much we expect the true velocity to
# randomly change per second (in m/s^2). Larger = filter trusts new
# measurements more; smaller = filter trusts its own motion model more.
KALMAN_PROCESS_NOISE_MPS2 = 0.6

# How far ahead (seconds) the short-horizon range/velocity forecast looks.
PREDICTION_HORIZON_SEC = 1.0

ENABLE_PLOT = True
LIVE_PAUSE_SEC = 0.10

C = 299_792_458.0
EPS = 1e-15

F1_RF_HZ = FC_HZ + F1_OFFSET_HZ
F2_RF_HZ = FC_HZ + F2_OFFSET_HZ

R_UNAMBIG_M = C / (2.0 * DELTA_F_HZ)
R_PER_DEG_M = C / (720.0 * DELTA_F_HZ)

COUPLING_CYCLE_N = (
    COUPLING_LONG_GUARD_N
    + COUPLING_SLOT_N
    + COUPLING_SHORT_GUARD_N
    + COUPLING_SLOT_N
)

assert LIVE_N % 4 == 0
assert COUPLING_SLOT_N % 4 == 0
assert COUPLING_EDGE_SKIP_N * 2 < COUPLING_SLOT_N
assert COUPLING_RX_BUFFER_N > COUPLING_CYCLE_N

def wrap_pi(x):
    """Wrap radians to [-pi, pi)."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def rms_dbfs(x):
    """
    Relative digital RMS level, not calibrated RF power in dBm.
    """
    return 20.0 * np.log10(
        np.sqrt(np.mean(np.abs(x) ** 2)) / 2048.0 + EPS
    )


def extract_rx_matrix(data):
    """
    Return a complex array with shape (2, number_of_samples).
    """
    if isinstance(data, (list, tuple)):
        arr = np.vstack([np.asarray(data[i]) for i in range(2)])
    else:
        arr = np.asarray(data)
        if arr.ndim == 1:
            raise RuntimeError("Only one RX channel was returned.")
        arr = arr[:2]

    if arr.shape[0] != 2:
        raise RuntimeError(
            "Expected two RX channels, received shape {}.".format(arr.shape)
        )

    return arr.astype(np.complex128, copy=False)


def stop_tx(sdr):
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass

    try:
        sdr.dds_enabled = [0] * len(sdr.dds_enabled)
    except Exception:
        pass


def flush_rx(sdr, count=FLUSH_COUNT):
    for _ in range(count):
        sdr.rx()

def make_two_tone(n_samples):
    """
    Generate the two coherent complex tones at +/- Fs/4.
    """
    n = np.arange(n_samples)

    x = (
        np.exp(1j * 2.0 * np.pi * F1_OFFSET_HZ * n / FS_HZ)
        + np.exp(1j * 2.0 * np.pi * F2_OFFSET_HZ * n / FS_HZ)
    )

    x /= np.max(np.abs(x)) + EPS
    x *= TX_SCALE

    return x.astype(np.complex64)


def estimate_two_tone_phasors(x):
    """
    Coherently estimate the two complex tone amplitudes.

    The local sample index starts at zero. This is important during
    the TDM coupling calibration because each slot is extracted using
    the same local phase reference.
    """
    x = np.asarray(x, dtype=np.complex128)
    n = np.arange(len(x))
    win = np.hanning(len(x))
    win_sum = np.sum(win) + EPS

    x = x - np.mean(x)
    xw = x * win

    z1 = np.sum(
        xw * np.exp(-1j * 2.0 * np.pi * F1_OFFSET_HZ * n / FS_HZ)
    ) / win_sum

    z2 = np.sum(
        xw * np.exp(-1j * 2.0 * np.pi * F2_OFFSET_HZ * n / FS_HZ)
    ) / win_sum

    return np.array([z1, z2], dtype=np.complex128)


def estimate_rx_tone_matrix(rx_matrix):
    """
    Return Z with shape:
        RX x tone = (2, 2)
    """
    z = np.zeros((2, 2), dtype=np.complex128)

    for rx_idx in range(2):
        z[rx_idx] = estimate_two_tone_phasors(rx_matrix[rx_idx])

    return z

def make_tdm_coupling_waveforms():
    """
    Build one cyclic two-channel calibration waveform:

      long zero guard
      TX0-only two-tone slot
      short zero guard
      TX1-only two-tone slot

    Both coupling columns are therefore measured during one coherent
    dual-channel TX run, avoiding arbitrary phase changes caused by
    restarting TX0 and TX1 separately.
    """
    tone = make_two_tone(COUPLING_SLOT_N)
    z_long = np.zeros(COUPLING_LONG_GUARD_N, dtype=np.complex64)
    z_short = np.zeros(COUPLING_SHORT_GUARD_N, dtype=np.complex64)
    z_slot = np.zeros(COUPLING_SLOT_N, dtype=np.complex64)

    tx0 = np.concatenate([z_long, tone, z_short, z_slot])
    tx1 = np.concatenate([z_long, z_slot, z_short, tone])

    return tx0, tx1


def _segment_mean(prefix, start, length):
    return (
        prefix[start + length] - prefix[start]
    ) / float(length)


def locate_tdm_cycle(rx_matrix):
    """
    Locate the start of a complete received TDM cycle by comparing
    active-slot energy with guard energy.

    A coarse search is followed by sample-level refinement.
    """
    power = np.mean(np.abs(rx_matrix) ** 2, axis=0)
    prefix = np.concatenate([[0.0], np.cumsum(power)])

    max_start = len(power) - COUPLING_CYCLE_N
    if max_start < 0:
        raise RuntimeError("RX buffer is shorter than one coupling cycle.")

    def score(start):
        g0 = _segment_mean(
            prefix,
            start,
            COUPLING_LONG_GUARD_N,
        )

        slot0_start = start + COUPLING_LONG_GUARD_N
        a0 = _segment_mean(
            prefix,
            slot0_start,
            COUPLING_SLOT_N,
        )

        g1_start = slot0_start + COUPLING_SLOT_N
        g1 = _segment_mean(
            prefix,
            g1_start,
            COUPLING_SHORT_GUARD_N,
        )

        slot1_start = g1_start + COUPLING_SHORT_GUARD_N
        a1 = _segment_mean(
            prefix,
            slot1_start,
            COUPLING_SLOT_N,
        )

        active = 0.5 * (a0 + a1)
        guard = 0.5 * (g0 + g1)

        return active / (guard + EPS)

    coarse_step = 64
    coarse_starts = range(0, max_start + 1, coarse_step)

    best_start = 0
    best_score = -np.inf

    for start in coarse_starts:
        value = score(start)
        if value > best_score:
            best_score = value
            best_start = start

    refine_lo = max(0, best_start - coarse_step)
    refine_hi = min(max_start, best_start + coarse_step)

    for start in range(refine_lo, refine_hi + 1):
        value = score(start)
        if value > best_score:
            best_score = value
            best_start = start

    return best_start, 10.0 * np.log10(best_score + EPS)


def extract_coupling_matrix_from_capture(rx_matrix):
    """
    Extract Hc with shape:
        RX x TX x tone = (2, 2, 2)
    """
    cycle_start, activity_ratio_db = locate_tdm_cycle(rx_matrix)

    slot0_start = (
        cycle_start
        + COUPLING_LONG_GUARD_N
        + COUPLING_EDGE_SKIP_N
    )

    slot1_start = (
        cycle_start
        + COUPLING_LONG_GUARD_N
        + COUPLING_SLOT_N
        + COUPLING_SHORT_GUARD_N
        + COUPLING_EDGE_SKIP_N
    )

    useful_n = COUPLING_SLOT_N - 2 * COUPLING_EDGE_SKIP_N

    slot0 = rx_matrix[:, slot0_start:slot0_start + useful_n]
    slot1 = rx_matrix[:, slot1_start:slot1_start + useful_n]

    if slot0.shape[1] != useful_n or slot1.shape[1] != useful_n:
        raise RuntimeError("A coupling slot was cut by the RX buffer.")

    h = np.zeros((2, 2, 2), dtype=np.complex128)

    # First TX column.
    h[:, 0, :] = estimate_rx_tone_matrix(slot0)

    # Second TX column.
    h[:, 1, :] = estimate_rx_tone_matrix(slot1)

    return h, activity_ratio_db


def capture_coupling_matrix(sdr, n_captures):
    """
    Average multiple coherent TDM coupling captures.

    A per-tone global phase alignment is allowed because a global phase
    at one tone does not affect the TX-space leakage minimization.
    """
    h_ref = None
    h_sum = np.zeros((2, 2, 2), dtype=np.complex128)
    valid = 0
    ratios = []

    for idx in range(n_captures):
        rx = extract_rx_matrix(sdr.rx())

        try:
            h_now, ratio_db = extract_coupling_matrix_from_capture(rx)
        except RuntimeError as exc:
            print(
                "\rCoupling {}/{} skipped: {}"
                .format(idx + 1, n_captures, exc),
                end="",
                flush=True,
            )
            continue

        if h_ref is None:
            h_ref = h_now.copy()
        else:
            # Align each tone by one common phase over all RX/TX entries.
            for tone_idx in range(2):
                phase = np.angle(
                    np.sum(
                        h_now[:, :, tone_idx]
                        * np.conj(h_ref[:, :, tone_idx])
                    )
                    + EPS
                )

                h_now[:, :, tone_idx] *= np.exp(-1j * phase)

        h_sum += h_now
        valid += 1
        ratios.append(ratio_db)

        print(
            "\rCoupling capture {}/{} | valid={} | activity/guard={:.1f} dB"
            .format(idx + 1, n_captures, valid, ratio_db),
            end="",
            flush=True,
        )

    print()

    if valid == 0:
        raise RuntimeError("No valid coherent coupling captures.")

    return h_sum / valid, float(np.mean(ratios)), valid


def compute_common_tx_weight(h_coupling):
    """
    Compute one common unit-norm TX weight for both tones.

    For tone m:
        H_m has shape RX x TX.

    We minimize:
        sum_m ||H_m w||^2
        = w^H [sum_m H_m^H H_m] w

    The solution is the eigenvector belonging to the smallest
    eigenvalue of the aggregate Gram matrix.
    """
    gram = np.zeros((2, 2), dtype=np.complex128)

    for tone_idx in range(2):
        h_tone = h_coupling[:, :, tone_idx]
        gram += h_tone.conj().T @ h_tone

    eigenvalues, eigenvectors = np.linalg.eigh(gram)

    weight = eigenvectors[:, np.argmin(eigenvalues)]
    weight /= np.linalg.norm(weight) + EPS

    # Set TX0 coefficient as the phase reference.
    weight *= np.exp(-1j * np.angle(weight[0]))

    baseline_power = 0.0
    weighted_power = 0.0

    for tone_idx in range(2):
        h_tone = h_coupling[:, :, tone_idx]
        baseline_power += np.linalg.norm(h_tone[:, 0]) ** 2
        weighted_power += np.linalg.norm(h_tone @ weight) ** 2

    predicted_suppression_db = 10.0 * np.log10(
        (baseline_power + EPS) / (weighted_power + EPS)
    )

    return weight, eigenvalues, predicted_suppression_db, baseline_power


# ============================================================
# STAGES 4-7: CONTINUOUS WEIGHTED TWO-TONE OPERATION
# ============================================================

def start_weighted_two_tone_tx(sdr, weight):
    """
    Start one continuous dual-TX cyclic two-tone waveform.

    The same complex TX weight is applied to both tones so the TX
    weighting does not create an artificial inter-tone delay.
    """
    base = make_two_tone(LIVE_N)

    tx0 = (weight[0] * base).astype(np.complex64)
    tx1 = (weight[1] * base).astype(np.complex64)

    peak = max(
        np.max(np.abs(tx0)),
        np.max(np.abs(tx1)),
        1.0,
    )

    if peak > (2 ** 14 - 1):
        scale = (2 ** 14 - 1) / peak
        tx0 *= scale
        tx1 *= scale

    stop_tx(sdr)
    sdr.tx_enabled_channels = TX_CHANNELS
    sdr.tx_cyclic_buffer = True
    sdr.tx([tx0, tx1])


def capture_average_tone_matrix(sdr, n_frames, label):
    """
    Capture and coherently average a 2-RX x 2-tone matrix.

    Each frame is aligned to the first frame by one common phase only.
    A single common rotation preserves the inter-tone phase used for
    ranging.
    """
    z_ref = None
    z_sum = np.zeros((2, 2), dtype=np.complex128)
    powers_sum = np.zeros(2)
    valid = 0

    for idx in range(n_frames):
        rx = extract_rx_matrix(sdr.rx())

        if rx.shape[1] < LIVE_N:
            continue

        rx = rx[:, :LIVE_N]
        z_now = estimate_rx_tone_matrix(rx)

        if z_ref is None:
            z_ref = z_now.copy()
        else:
            common_phase = np.angle(
                np.sum(z_now * np.conj(z_ref)) + EPS
            )
            z_now *= np.exp(-1j * common_phase)

        z_sum += z_now
        powers_sum += np.array(
            [rms_dbfs(rx[0]), rms_dbfs(rx[1])]
        )
        valid += 1

        print(
            "\r{}: {}/{} frames | valid={}"
            .format(label, idx + 1, n_frames, valid),
            end="",
            flush=True,
        )

    print()

    if valid == 0:
        raise RuntimeError("No valid frames during {}.".format(label))

    return z_sum / valid, powers_sum / valid, valid


def align_and_subtract_background(z_scene, z_background):
    """
    Remove one common phase rotation and then subtract the stored
    complex background independently at every RX/tone entry.
    """
    common_phase = np.angle(
        np.sum(z_scene * np.conj(z_background)) + EPS
    )

    z_aligned = z_scene * np.exp(-1j * common_phase)
    z_target = z_aligned - z_background

    echo_to_bg_db = 20.0 * np.log10(
        np.linalg.norm(z_target)
        / (np.linalg.norm(z_background) + EPS)
        + EPS
    )

    return z_target, common_phase, echo_to_bg_db


def intertone_phase_vector(z_target):
    """
    For each RX:
        q_r = z_r(f2) * conj(z_r(f1))
    """
    return z_target[:, 1] * np.conj(z_target[:, 0])


def fuse_relative_phase(q_live, q_cal, override_weights=None):
    """
    Compare the live inter-tone phasor with the known-range calibration
    phasor and fuse the two RX channels in the complex phase domain.

    By default, stronger RX channels receive more weight (amplitude-based
    reliability). Pass ``override_weights`` (e.g. from an
    AntennaErrorTracker) to fuse with weights that also account for each
    antenna's recent phase-error track record instead of amplitude alone.
    """
    q_live_unit = q_live / (np.abs(q_live) + EPS)
    q_cal_unit = q_cal / (np.abs(q_cal) + EPS)

    q_relative = q_live_unit * np.conj(q_cal_unit)

    # Reliability uses both current and calibration tone-pair strength.
    reliability = np.sqrt(
        np.abs(q_live) * np.abs(q_cal)
    )

    reliability /= np.sum(reliability) + EPS

    fusion_weights = (
        reliability if override_weights is None else override_weights
    )

    fused_complex = np.sum(fusion_weights * q_relative)
    fused_phase = np.angle(fused_complex + EPS)

    per_rx_phase = np.angle(q_relative)

    phase_disagreement = abs(
        wrap_pi(per_rx_phase[0] - per_rx_phase[1])
    )

    range_disagreement_m = (
        C * phase_disagreement
        / (4.0 * np.pi * DELTA_F_HZ)
    )

    return (
        fused_phase,
        per_rx_phase,
        reliability,
        range_disagreement_m,
    )


class PhaseUnwrapper:
    """
    Stateful unwrap for the fused live relative phase.
    """

    def __init__(self):
        self.previous_raw = None
        self.unwrapped = None

    def update(self, raw_phase):
        if self.previous_raw is None:
            self.previous_raw = raw_phase
            self.unwrapped = raw_phase
            return self.unwrapped

        increment = wrap_pi(raw_phase - self.previous_raw)
        self.unwrapped += increment
        self.previous_raw = raw_phase

        return self.unwrapped


class AntennaErrorTracker:
    """
    Tracks each RX antenna's phase error against the fused estimate and
    turns that running error into an adaptive trust weight.

    "Antenna error" here means how far one RX channel's relative-phase
    (and therefore range) estimate drifts from the fused estimate over
    time -- e.g. from a loose cable, connector, or antenna misalignment
    that calibration did not fully capture. Rather than a fixed 50/50
    or purely amplitude-based split between RX0/RX1, this lets the code
    itself decide, run to run, which antenna to trust more.
    """

    def __init__(self, n_rx=2, ema_alpha=ANTENNA_ERROR_EMA_ALPHA):
        self.ema_alpha = ema_alpha
        # Start with a small non-zero assumed error so early weights
        # are not wildly unstable before enough data has accumulated.
        self.error_ema_rad2 = np.full(n_rx, np.deg2rad(2.0) ** 2)

    def update(self, per_rx_phase, fused_phase, amplitude_weights):
        """
        per_rx_phase: wrapped relative phase per RX (radians).
        fused_phase: the current fused relative phase (radians).
        amplitude_weights: per-RX reliability from signal strength alone.

        Returns (combined_weight, error_deg, error_m).
        """
        deviation = np.array([
            wrap_pi(phase - fused_phase) for phase in per_rx_phase
        ])

        self.error_ema_rad2 = (
            self.ema_alpha * self.error_ema_rad2
            + (1.0 - self.ema_alpha) * deviation ** 2
        )

        # An antenna with a larger tracked error is trusted less.
        consistency_weight = 1.0 / (self.error_ema_rad2 + EPS)

        combined_weight = consistency_weight * amplitude_weights
        combined_weight /= np.sum(combined_weight) + EPS

        error_rad = np.sqrt(self.error_ema_rad2)
        error_deg = np.rad2deg(error_rad)
        error_m = C * error_rad / (4.0 * np.pi * DELTA_F_HZ)

        return combined_weight, error_deg, error_m


class RangeVelocityKalmanFilter:
    """
    Constant-velocity Kalman filter over state [range, velocity].

    This replaces plain exponential smoothing with a proper recursive
    estimator: it fuses the noisy fused-phase range measurement with a
    simple motion model, produces a velocity estimate that is not just
    a noisy frame-to-frame difference, and supports extrapolating a
    short distance/velocity forecast ahead of the latest measurement.
    """

    def __init__(
        self,
        process_noise_std_mps2=KALMAN_PROCESS_NOISE_MPS2,
    ):
        self.x = np.array([0.0, 0.0])
        self.p = np.diag([0.05 ** 2, 0.5 ** 2])
        self.process_noise_std_mps2 = process_noise_std_mps2
        self.initialized = False

    def _predict(self, dt_s):
        dt_s = max(dt_s, 1e-6)
        f = np.array([[1.0, dt_s], [0.0, 1.0]])

        q_std = self.process_noise_std_mps2
        q = q_std ** 2 * np.array([
            [dt_s ** 4 / 4.0, dt_s ** 3 / 2.0],
            [dt_s ** 3 / 2.0, dt_s ** 2],
        ])

        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q

    def _update(self, range_measurement_m, measurement_noise_std_m):
        h = np.array([1.0, 0.0])
        r = measurement_noise_std_m ** 2

        y = range_measurement_m - h @ self.x
        s = h @ self.p @ h.T + r
        k = (self.p @ h) / (s + EPS)

        self.x = self.x + k * y
        self.p = self.p - np.outer(k, h) @ self.p

    def step(self, range_measurement_m, dt_s, measurement_noise_std_m):
        """
        Advance the filter by dt_s and fold in a new range measurement.
        Returns (filtered_range_m, filtered_velocity_mps).
        """
        if not self.initialized:
            self.x = np.array([range_measurement_m, 0.0])
            self.initialized = True
        else:
            self._predict(dt_s)
            self._update(range_measurement_m, measurement_noise_std_m)

        return self.x[0], self.x[1]

    def predict_ahead(self, horizon_s):
        """
        Extrapolate the current state forward without consuming a new
        measurement. Used for the short-horizon forecast printout and
        for the GUI forecast readout; it does not feed back into the
        filter's own state.

        Returns (forecast_range_m, forecast_velocity_mps). Under this
        constant-velocity model the forecast velocity is just the
        current filtered velocity -- only the range is extrapolated
        forward in time; the velocity itself isn't expected to change
        over the horizon.
        """
        range_m, velocity_mps = self.x
        forecast_range_m = range_m + velocity_mps * horizon_s
        forecast_velocity_mps = velocity_mps
        return forecast_range_m, forecast_velocity_mps


def measurement_noise_std_m(echo_bg_db, rx_disagreement_m):
    """
    Turn the existing quality diagnostics into a Kalman measurement-noise
    estimate: a weak echo or antennas that disagree with each other makes
    this frame's range measurement less trustworthy, so the filter should
    lean more on its own motion model instead of jumping to match it.
    """
    weak_echo_penalty = max(0.0, MIN_ECHO_TO_BG_DB - echo_bg_db)

    return (
        0.01
        + 0.02 * weak_echo_penalty
        + 2.0 * rx_disagreement_m
    )


# ============================================================
# LIVE PLOT
# ============================================================

class RangePlot:
    """Presentation dashboard only; it does not alter the ranging DSP."""

    BG = "#f4f7fb"
    PANEL = "#ffffff"
    GRID = "#d7e0ea"
    TEXT = "#172033"
    MUTED = "#64748b"
    CYAN = "#0077b6"
    BLUE = "#4169e1"
    GREEN = "#149b62"
    AMBER = "#d98200"
    RED = "#d64550"
    PURPLE = "#7c3aed"

    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 8), facecolor=self.BG)
        self.fig.canvas.manager.set_window_title(
            "SPECTRA | Two-Tone MFCW Range and Velocity"
        )
        self.fig.text(0.045, 0.945, "SPECTRA", color=self.CYAN,
                      fontsize=20, fontweight="bold")
        self.fig.text(0.168, 0.95, "TWO-TONE MFCW RADAR DASHBOARD",
                      color=self.MUTED, fontsize=10, fontweight="bold")
        self.status = self.fig.text(0.955, 0.95, "INITIALIZING",
                                    color=self.AMBER, fontsize=10,
                                    ha="right", fontweight="bold")

        self.distance_card = self._card([0.045, 0.69, 0.275, 0.19])
        self.velocity_card = self._card([0.34, 0.69, 0.275, 0.19])
        self.quality_card = self._card([0.635, 0.69, 0.32, 0.19])
        self.range_ax = self._chart([0.045, 0.105, 0.56, 0.50],
                                    "RANGE HISTORY", "Distance (m)")
        self.velocity_ax = self._chart([0.635, 0.105, 0.32, 0.50],
                                       "VELOCITY HISTORY", "Velocity (m/s)")

        self._metric_card(self.distance_card, "DISTANCE", "m", self.CYAN)
        self._metric_card(self.velocity_card, "VELOCITY", "m/s", self.GREEN)
        self.distance_value = self.distance_card.text(
            0.06, 0.42, "--", transform=self.distance_card.transAxes,
            color=self.TEXT, fontsize=32, fontweight="bold", va="center")
        self.velocity_value = self.velocity_card.text(
            0.06, 0.42, "--", transform=self.velocity_card.transAxes,
            color=self.TEXT, fontsize=32, fontweight="bold", va="center")
        self.motion_label = self.velocity_card.text(
            0.06, 0.20, "WAITING FOR LIVE DATA", transform=self.velocity_card.transAxes,
            color=self.MUTED, fontsize=9, fontweight="bold")

        # --- Forecast readouts: future distance / future velocity ---
        # Sourced from RangeVelocityKalmanFilter.predict_ahead().
        self.distance_forecast = self.distance_card.text(
            0.06, 0.06,
            "+{:.0f}s \u2192 -- m".format(PREDICTION_HORIZON_SEC),
            transform=self.distance_card.transAxes,
            color=self.PURPLE, fontsize=10, fontweight="bold")
        self.velocity_forecast = self.velocity_card.text(
            0.06, 0.06,
            "+{:.0f}s \u2192 -- m/s".format(PREDICTION_HORIZON_SEC),
            transform=self.velocity_card.transAxes,
            color=self.PURPLE, fontsize=10, fontweight="bold")

        self.quality_card.text(0.06, 0.77, "MEASUREMENT CONFIDENCE",
                               transform=self.quality_card.transAxes,
                               color=self.MUTED, fontsize=9, fontweight="bold")
        self.quality_value = self.quality_card.text(
            0.06, 0.39, "--", transform=self.quality_card.transAxes,
            color=self.TEXT, fontsize=24, fontweight="bold")
        self.quality_detail = self.quality_card.text(
            0.06, 0.16, "echo/background  -- dB",
            transform=self.quality_card.transAxes, color=self.MUTED, fontsize=9)
        self.quality_bar = self.quality_card.barh(
            [0.06], [0], height=0.11, left=-45, color=self.AMBER)[0]
        self.quality_card.set_xlim(-45, 20)
        self.quality_card.set_ylim(-0.15, 1)
        self.quality_card.set_yticks([])
        self.quality_card.set_xticks([-45, -35, -15, 5, 20])
        self.quality_card.tick_params(colors=self.MUTED, labelsize=7)
        self.quality_card.axvspan(-45, -35, color=self.RED, alpha=0.25)
        self.quality_card.axvspan(-35, -15, color=self.AMBER, alpha=0.20)
        self.quality_card.axvspan(-15, 20, color=self.GREEN, alpha=0.16)

        self.range_raw, = self.range_ax.plot([], [], color=self.BLUE,
                                              lw=1.1, alpha=0.55,
                                              label="Fused raw")
        self.range_smooth, = self.range_ax.plot([], [], color=self.CYAN,
                                                 lw=2.6, label="Display range")
        self.rx0_line, = self.range_ax.plot([], [], color=self.AMBER,
                                             lw=0.9, ls="--", alpha=0.6,
                                             label="RX0")
        self.rx1_line, = self.range_ax.plot([], [], color=self.GREEN,
                                             lw=0.9, ls="--", alpha=0.6,
                                             label="RX1")
        # Forecast point projected ahead of the latest range sample.
        self.range_forecast_point, = self.range_ax.plot(
            [], [], color=self.PURPLE, marker="D", markersize=6,
            linestyle="None", label="+{:.0f}s forecast".format(
                PREDICTION_HORIZON_SEC
            ))
        self.range_forecast_line, = self.range_ax.plot(
            [], [], color=self.PURPLE, lw=1.2, ls=":", alpha=0.8)
        self.range_ax.legend(loc="upper left", ncol=5, fontsize=7.5,
                             facecolor=self.PANEL, edgecolor=self.GRID,
                             labelcolor=self.TEXT)
        self.velocity_line, = self.velocity_ax.plot([], [], color=self.GREEN,
                                                    lw=2.2, label="Velocity")
        self.velocity_forecast_point, = self.velocity_ax.plot(
            [], [], color=self.PURPLE, marker="D", markersize=6,
            linestyle="None", label="+{:.0f}s forecast".format(
                PREDICTION_HORIZON_SEC
            ))
        self.velocity_forecast_line, = self.velocity_ax.plot(
            [], [], color=self.PURPLE, lw=1.2, ls=":", alpha=0.8)
        self.velocity_ax.axhline(0, color=self.MUTED, lw=0.8, alpha=0.6)
        self.velocity_ax.legend(loc="upper left", ncol=2, fontsize=7.5,
                                facecolor=self.PANEL, edgecolor=self.GRID,
                                labelcolor=self.TEXT)

        self.x = []
        self.raw = []
        self.smooth = []
        self.velocity = []
        self.rx0 = []
        self.rx1 = []
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _card(self, rect):
        ax = self.fig.add_axes(rect, facecolor=self.PANEL)
        for spine in ax.spines.values():
            spine.set_color(self.GRID)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def _metric_card(self, ax, title, unit, color):
        ax.text(0.06, 0.79, title, transform=ax.transAxes, color=self.MUTED,
                fontsize=9, fontweight="bold")
        ax.text(0.94, 0.79, unit, transform=ax.transAxes, color=color,
                fontsize=10, ha="right", fontweight="bold")
        ax.plot([0.06, 0.94], [0.68, 0.68], transform=ax.transAxes,
                color=self.GRID, lw=1)

    def _chart(self, rect, title, ylabel):
        ax = self.fig.add_axes(rect, facecolor=self.PANEL)
        for spine in ax.spines.values():
            spine.set_color(self.GRID)
        ax.set_title(title, loc="left", color=self.MUTED, fontsize=9,
                     fontweight="bold", pad=10)
        ax.set_xlabel("Live updates", color=self.MUTED, fontsize=8)
        ax.set_ylabel(ylabel, color=self.MUTED, fontsize=8)
        ax.tick_params(colors=self.MUTED, labelsize=8)
        ax.grid(True, color=self.GRID, alpha=0.55, linewidth=0.7)
        return ax

    def update(self, raw_range, smooth_range, velocity_mps, rx_ranges,
               echo_bg_db, rx_disagreement_m, status_flags,
               forecast_range_m=None, forecast_velocity_mps=None):
        self.x.append(len(self.x))
        self.raw.append(raw_range)
        self.smooth.append(smooth_range)
        self.velocity.append(velocity_mps)
        self.rx0.append(rx_ranges[0])
        self.rx1.append(rx_ranges[1])
        max_points = 160
        self.x = self.x[-max_points:]
        self.raw = self.raw[-max_points:]
        self.smooth = self.smooth[-max_points:]
        self.velocity = self.velocity[-max_points:]
        self.rx0 = self.rx0[-max_points:]
        self.rx1 = self.rx1[-max_points:]

        self.range_raw.set_data(self.x, self.raw)
        self.range_smooth.set_data(self.x, self.smooth)
        self.rx0_line.set_data(self.x, self.rx0)
        self.rx1_line.set_data(self.x, self.rx1)
        self.velocity_line.set_data(self.x, self.velocity)

        # Forecast is drawn one "step" ahead of the last live-update
        # index, as a projected point plus a dotted connector.
        last_x = self.x[-1]
        forecast_x = last_x + 1

        if forecast_range_m is not None:
            self.range_forecast_point.set_data([forecast_x], [forecast_range_m])
            self.range_forecast_line.set_data(
                [last_x, forecast_x], [smooth_range, forecast_range_m]
            )
        else:
            self.range_forecast_point.set_data([], [])
            self.range_forecast_line.set_data([], [])

        if forecast_velocity_mps is not None:
            self.velocity_forecast_point.set_data(
                [forecast_x], [forecast_velocity_mps]
            )
            self.velocity_forecast_line.set_data(
                [last_x, forecast_x], [velocity_mps, forecast_velocity_mps]
            )
        else:
            self.velocity_forecast_point.set_data([], [])
            self.velocity_forecast_line.set_data([], [])

        self.range_ax.relim(); self.range_ax.autoscale_view()
        self.velocity_ax.relim(); self.velocity_ax.autoscale_view()

        self.distance_value.set_text("{:.3f}".format(smooth_range))
        self.velocity_value.set_text("{:+.3f}".format(velocity_mps))

        if forecast_range_m is not None:
            self.distance_forecast.set_text(
                "+{:.0f}s \u2192 {:.3f} m".format(
                    PREDICTION_HORIZON_SEC, forecast_range_m
                )
            )
        if forecast_velocity_mps is not None:
            self.velocity_forecast.set_text(
                "+{:.0f}s \u2192 {:+.3f} m/s".format(
                    PREDICTION_HORIZON_SEC, forecast_velocity_mps
                )
            )

        if velocity_mps > 0.03:
            motion, motion_color = "MOVING AWAY", self.GREEN
        elif velocity_mps < -0.03:
            motion, motion_color = "MOVING CLOSER", self.CYAN
        else:
            motion, motion_color = "STATIONARY", self.MUTED
        self.motion_label.set_text(motion)
        self.motion_label.set_color(motion_color)

        is_good = echo_bg_db >= MIN_ECHO_TO_BG_DB and rx_disagreement_m <= MAX_RX_DISAGREEMENT_M
        quality = "TRACK LOCKED" if is_good else "CHECK SIGNAL"
        quality_color = self.GREEN if is_good else self.AMBER
        if echo_bg_db < MIN_ECHO_TO_BG_DB - 10:
            quality, quality_color = "WEAK ECHO", self.RED
        self.quality_value.set_text(quality)
        self.quality_value.set_color(quality_color)
        self.quality_detail.set_text("echo/background  {:+.1f} dB".format(echo_bg_db))
        bar_value = np.clip(echo_bg_db, -45, 20)
        self.quality_bar.set_width(bar_value + 45)
        self.quality_bar.set_color(quality_color)
        self.status.set_text("LIVE  •  " + " / ".join(status_flags))
        self.status.set_color(quality_color)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# ============================================================
# SDR CONFIGURATION
# ============================================================

def configure_sdr():
    board = SDR_TYPE.strip().upper()
    print("Connecting to {}/ZC702: {}".format(board, URI))

    if board == "FMCOMMS5":
        sdr = adi.FMComms5(uri=URI)
    elif board == "FMCOMMS3":
        sdr = adi.ad9361(uri=URI)
    else:
        raise ValueError(
            'SDR_TYPE must be "FMCOMMS3" or "FMCOMMS5", not {!r}.'
            .format(SDR_TYPE)
        )

    sdr.rx_enabled_channels = RX_CHANNELS
    sdr.tx_enabled_channels = TX_CHANNELS

    sdr.sample_rate = FS_HZ

    sdr.rx_lo = FC_HZ
    sdr.tx_lo = FC_HZ

    # Keep chip B tuned but disabled at minimum gains.
    try:
        sdr.rx_lo_chip_b = FC_HZ
        sdr.tx_lo_chip_b = FC_HZ
    except Exception:
        pass

    sdr.rx_rf_bandwidth = RF_BW_HZ
    sdr.tx_rf_bandwidth = RF_BW_HZ

    try:
        sdr.rx_rf_bandwidth_chip_b = RF_BW_HZ
        sdr.tx_rf_bandwidth_chip_b = RF_BW_HZ
    except Exception:
        pass

    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"

    sdr.rx_hardwaregain_chan0 = RX_GAIN_DB[0]
    sdr.rx_hardwaregain_chan1 = RX_GAIN_DB[1]

    sdr.tx_hardwaregain_chan0 = TX_GAIN_DB[0]
    sdr.tx_hardwaregain_chan1 = TX_GAIN_DB[1]

    try:
        sdr.gain_control_mode_chip_b_chan0 = "manual"
        sdr.gain_control_mode_chip_b_chan1 = "manual"
        sdr.rx_hardwaregain_chip_b_chan0 = 0
        sdr.rx_hardwaregain_chip_b_chan1 = 0
        sdr.tx_hardwaregain_chip_b_chan0 = -89.75
        sdr.tx_hardwaregain_chip_b_chan1 = -89.75
    except Exception:
        pass

    try:
        sdr._rxadc.set_kernel_buffers_count(1)
    except Exception:
        pass

    return sdr


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    sdr = configure_sdr()

    print("\n================ SYSTEM PARAMETERS ================")
    print("SDR type                 :", SDR_TYPE)
    print("Carrier                  : {:.6f} GHz".format(FC_HZ / 1e9))
    print("Tone 1 RF                : {:.6f} GHz".format(F1_RF_HZ / 1e9))
    print("Tone 2 RF                : {:.6f} GHz".format(F2_RF_HZ / 1e9))
    print("Tone separation          : {:.3f} MHz".format(DELTA_F_HZ / 1e6))
    print("Unambiguous range period : {:.3f} m".format(R_UNAMBIG_M))
    print("Range per phase degree   : {:.2f} cm/deg".format(
        R_PER_DEG_M * 100.0
    ))
    print("Known calibration range  : {:.3f} m".format(
        CALIBRATION_RANGE_M
    ))

    range_plot = RangePlot() if ENABLE_PLOT else None

    try:
        # ========================================================
        # STAGE 1
        # ========================================================
        print("\n===== STAGE 1: CONFIGURE 2TX / 2RX TWO-TONE SYSTEM =====")
        print("RX channels:", RX_CHANNELS)
        print("TX channels:", TX_CHANNELS)

        # ========================================================
        # STAGE 2
        # ========================================================
        print("\n===== STAGE 2: COHERENT 2x2 COUPLING MEASUREMENT =====")
        print("Clear the scene. Remove the reflector.")
        print("Do not move antennas or RF cables after this stage.")
        input("Press Enter to start coherent TX coupling calibration...")

        tx0_cal, tx1_cal = make_tdm_coupling_waveforms()

        stop_tx(sdr)
        sdr.rx_buffer_size = COUPLING_RX_BUFFER_N
        sdr.tx_cyclic_buffer = True
        sdr.tx([tx0_cal, tx1_cal])

        time.sleep(SETTLE_TIME_SEC)
        flush_rx(sdr)

        h_coupling, activity_db, coupling_valid = capture_coupling_matrix(
            sdr,
            COUPLING_CAPTURES,
        )

        print("Valid coupling captures :", coupling_valid)
        print("Mean slot/guard contrast: {:.1f} dB".format(activity_db))

        for tone_idx, tone_name in enumerate(("f1", "f2")):
            print(
                "{} |H|:\n{}"
                .format(
                    tone_name,
                    np.array2string(
                        np.abs(h_coupling[:, :, tone_idx]),
                        precision=3,
                    ),
                )
            )

        # ========================================================
        # STAGE 3
        # ========================================================
        print("\n===== STAGE 3: COMMON DUAL-TX LEAKAGE WEIGHT =====")

        (
            tx_weight,
            gram_eigenvalues,
            predicted_suppression_db,
            baseline_coupling_power,
        ) = compute_common_tx_weight(h_coupling)

        print(
            "TX weight:"
            "\n  w0 = {:.6f} {:+.6f}j"
            "\n  w1 = {:.6f} {:+.6f}j"
            .format(
                tx_weight[0].real,
                tx_weight[0].imag,
                tx_weight[1].real,
                tx_weight[1].imag,
            )
        )

        print(
            "Aggregate Gram eigenvalues:",
            np.array2string(gram_eigenvalues, precision=6),
        )

        print(
            "Predicted aggregate suppression relative to TX0-only:"
            " {:.2f} dB"
            .format(predicted_suppression_db)
        )

        # ========================================================
        # STAGE 4
        # ========================================================
        print("\n===== STAGE 4: START WEIGHTED DUAL-TX TWO-TONE =====")

        sdr.rx_buffer_size = LIVE_N
        start_weighted_two_tone_tx(sdr, tx_weight)

        time.sleep(SETTLE_TIME_SEC)
        flush_rx(sdr)

        z_weighted_clear, weighted_powers, _ = capture_average_tone_matrix(
            sdr,
            20,
            "Weighted clear-scene check",
        )

        observed_weighted_power = np.linalg.norm(z_weighted_clear) ** 2

        observed_suppression_db = 10.0 * np.log10(
            (baseline_coupling_power + EPS)
            / (observed_weighted_power + EPS)
        )

        print(
            "Weighted clear-scene RX levels: [{:.2f}, {:.2f}] dBFS"
            .format(weighted_powers[0], weighted_powers[1])
        )

        print(
            "Approx. observed suppression relative to TX0 coupling:"
            " {:.2f} dB"
            .format(observed_suppression_db)
        )

        print(
            "Note: the observed value also includes static room clutter,"
            " so it need not equal the coupling-only prediction."
        )

        # ========================================================
        # STAGE 5
        # ========================================================
        print("\n===== STAGE 5: EMPTY-SCENE BACKGROUND CAPTURE =====")
        print("Keep the scene clear and stationary.")
        input("Press Enter to capture the complex background/leakage...")

        time.sleep(SETTLE_TIME_SEC)
        flush_rx(sdr)

        z_background, background_powers, background_valid = (
            capture_average_tone_matrix(
                sdr,
                BACKGROUND_FRAMES,
                "Background",
            )
        )

        print("Background valid frames:", background_valid)
        print(
            "Background RX levels: [{:.2f}, {:.2f}] dBFS"
            .format(background_powers[0], background_powers[1])
        )

        # ========================================================
        # STAGE 6
        # ========================================================
        print("\n===== STAGE 6: KNOWN-RANGE PHASE CALIBRATION =====")
        print(
            "Place one strong reflector at {:.3f} m."
            .format(CALIBRATION_RANGE_M)
        )
        print("Keep it near broadside and stationary.")
        input("Press Enter to capture the known-range reflector...")

        time.sleep(SETTLE_TIME_SEC)
        flush_rx(sdr)

        z_cal_scene, cal_powers, cal_valid = capture_average_tone_matrix(
            sdr,
            CALIBRATION_FRAMES,
            "Known-range calibration",
        )

        z_cal_target, cal_common_phase, cal_echo_bg_db = (
            align_and_subtract_background(
                z_cal_scene,
                z_background,
            )
        )

        q_cal = intertone_phase_vector(z_cal_target)

        print("Calibration valid frames:", cal_valid)
        print(
            "Calibration RX levels: [{:.2f}, {:.2f}] dBFS"
            .format(cal_powers[0], cal_powers[1])
        )
        print(
            "Calibration echo/background: {:.2f} dB"
            .format(cal_echo_bg_db)
        )
        print(
            "Calibration common phase alignment: {:.2f} deg"
            .format(np.rad2deg(cal_common_phase))
        )
        print(
            "Calibration inter-tone phases: "
            "RX0={:.2f} deg, RX1={:.2f} deg"
            .format(
                np.rad2deg(np.angle(q_cal[0])),
                np.rad2deg(np.angle(q_cal[1])),
            )
        )

        if cal_echo_bg_db < MIN_ECHO_TO_BG_DB:
            print(
                "WARNING: the calibration reflector is weak relative to"
                " the stored background. Use a larger/closer reflector."
            )

        # ========================================================
        # STAGE 7
        # ========================================================
        print("\n===== STAGE 7: LIVE TWO-TONE RANGE =====")
        print(
            "Move the same dominant reflector continuously from the"
            " calibration position."
        )
        print("Press Ctrl+C to stop.\n")

        phase_unwrapper = PhaseUnwrapper()
        antenna_tracker = AntennaErrorTracker()
        kalman = RangeVelocityKalmanFilter()
        previous_time_s = None

        while True:
            z_scene, scene_powers, live_valid = capture_average_tone_matrix(
                sdr,
                LIVE_FRAMES,
                "Live",
            )

            z_target, common_phase, echo_bg_db = (
                align_and_subtract_background(
                    z_scene,
                    z_background,
                )
            )

            q_live = intertone_phase_vector(z_target)

            # First pass: amplitude-only fusion, needed to get per-RX
            # phases to compare against for the antenna-error tracker.
            (
                _,
                per_rx_phase,
                amplitude_weights,
                rx_disagreement_m,
            ) = fuse_relative_phase(q_live, q_cal)

            # The antenna-error tracker decides, from each RX's recent
            # track record, how much to trust it -- not just amplitude.
            adaptive_weights, antenna_error_deg, antenna_error_m = (
                antenna_tracker.update(
                    per_rx_phase, np.angle(np.sum(
                        amplitude_weights
                        * np.exp(1j * per_rx_phase)
                    )),
                    amplitude_weights,
                )
            )

            # Second pass: fuse again with the adaptive weights applied.
            (
                fused_raw_phase,
                per_rx_phase,
                _,
                rx_disagreement_m,
            ) = fuse_relative_phase(
                q_live, q_cal, override_weights=adaptive_weights
            )

            fused_unwrapped_phase = phase_unwrapper.update(
                fused_raw_phase
            )

            fused_range_m = (
                CALIBRATION_RANGE_M
                + RANGE_SIGN
                * C
                * fused_unwrapped_phase
                / (4.0 * np.pi * DELTA_F_HZ)
            )

            # Per-RX values are wrapped instantaneous diagnostics.
            per_rx_range_m = (
                CALIBRATION_RANGE_M
                + RANGE_SIGN
                * C
                * per_rx_phase
                / (4.0 * np.pi * DELTA_F_HZ)
            )

            # Positive velocity means the reflector is moving farther away.
            sample_time_s = time.monotonic()
            dt_s = (
                0.0
                if previous_time_s is None
                else sample_time_s - previous_time_s
            )
            previous_time_s = sample_time_s

            meas_noise_std_m = measurement_noise_std_m(
                echo_bg_db, rx_disagreement_m
            )

            smoothed_range, smoothed_velocity_mps = kalman.step(
                fused_range_m, dt_s, meas_noise_std_m
            )

            # Future distance and future estimated velocity: a
            # short-horizon forecast extrapolated from the Kalman
            # filter's current state, without consuming a new
            # measurement.
            predicted_range_ahead_m, predicted_velocity_ahead_mps = (
                kalman.predict_ahead(PREDICTION_HORIZON_SEC)
            )

            confidence_flags = []

            if echo_bg_db < MIN_ECHO_TO_BG_DB:
                confidence_flags.append("WEAK ECHO")

            if rx_disagreement_m > MAX_RX_DISAGREEMENT_M:
                confidence_flags.append("RX DISAGREE")

            if np.max(antenna_error_m) > MAX_ANTENNA_ERROR_M:
                confidence_flags.append("ANTENNA DRIFT")

            if not confidence_flags:
                confidence_flags.append("OK")

            print(
                "RANGE | fused={:7.3f} m | smooth={:7.3f} m | "
                "velocity={:+6.3f} m/s | "
                "+{:.0f}s->dist={:7.3f} m, vel={:+6.3f} m/s | "
                "RX0={:7.3f} | RX1={:7.3f} | spread={:5.3f} m | "
                "echo/bg={:6.1f} dB | weights=[{:.2f},{:.2f}] | "
                "ant.err=[{:.2f},{:.2f}] deg | "
                "P=[{:6.1f},{:6.1f}] dBFS | {}"
                .format(
                    fused_range_m,
                    smoothed_range,
                    smoothed_velocity_mps,
                    PREDICTION_HORIZON_SEC,
                    predicted_range_ahead_m,
                    predicted_velocity_ahead_mps,
                    per_rx_range_m[0],
                    per_rx_range_m[1],
                    rx_disagreement_m,
                    echo_bg_db,
                    adaptive_weights[0],
                    adaptive_weights[1],
                    antenna_error_deg[0],
                    antenna_error_deg[1],
                    scene_powers[0],
                    scene_powers[1],
                    ", ".join(confidence_flags),
                )
            )

            if range_plot is not None:
                range_plot.update(
                    fused_range_m,
                    smoothed_range,
                    smoothed_velocity_mps,
                    per_rx_range_m,
                    echo_bg_db,
                    rx_disagreement_m,
                    confidence_flags,
                    forecast_range_m=predicted_range_ahead_m,
                    forecast_velocity_mps=predicted_velocity_ahead_mps,
                )

            time.sleep(LIVE_PAUSE_SEC)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        stop_tx(sdr)
        print("TX stopped and cyclic buffer destroyed.")

        if ENABLE_PLOT:
            try:
                plt.ioff()
                plt.show()
            except Exception:
                pass


if __name__ == "__main__":
    main()