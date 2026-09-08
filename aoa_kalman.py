import adi
import numpy as np
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ============================================================
# FMCOMMS5 + ZC702 : Dual-TX leakage suppression + reflector AoA
# with FOUR DOA estimators run together: MUSIC, ESPRIT, MVDR, ML.
#
#   N_RX = 2 -> chip A only (RX0,RX1) ; N_RX = 4 -> both chips.
#
# All four estimators run every frame on the same background-subtracted
# covariance. The live needle shows a FUSED estimate; the console and
# GUI show all four plus their spread (agreement = confidence).
# ============================================================

URI = "ip:169.254.92.202"
FC = int(2.3e9)
FS = int(2e6)

# ---------- ARRAY SIZE KNOB ----------
N_RX = 2                      # 2 (chip A) or 4 (both AD9361s)
assert N_RX in (2, 4)
RX_CHANNELS = [0, 1] if N_RX == 2 else [0, 1, 2, 3]
N_ELEM = N_RX

# ---------- DOA fusion knobs ----------
DOA_K = 1                     # one reflector after subtraction
DISPLAY_SCHEME = "consensus"  # "consensus" | "median" | "music"
CONSENSUS_TOL_DEG = 8.0       # drop methods this far from the median
SPREAD_GOOD_DEG = 5.0         # spread below this = high agreement (green)
SPREAD_BAD_DEG = 20.0         # spread above this = distrust (red)
LOADING_FRAC = 1e-2           # diagonal loading on R (needed for MVDR)
SCAN_DEG = np.linspace(-90, 90, 721)   # 0.25 deg grid for MVDR/ML

# ---------- Kalman future prediction ----------
KALMAN_PREDICTION_HORIZON_S = 1.0   # predict this far into the future
KALMAN_PREDICTION_STEPS = 20
KALMAN_Q_ANGLE = 0.8               # process noise: angle
KALMAN_Q_RATE = 1.5                # process noise: angular rate
KALMAN_R_ANGLE = 6.0               # measurement noise variance (deg^2)

# ---------- Tone calibration ----------
TONE_OFFSET = int(200e3)
TONE_RX_BW = int(800e3)
TONE_TX_BW = int(800e3)
TONE_NUM_SAMPLES = 8192
CAL_AVG_COUNT = 50

# ---------- OFDM waveform ----------
NFFT = 128
CP = 32
SYM_LEN = NFFT + CP
NUM_DATA_SYMS = 20
OFDM_RX_BUFFER_SIZE = 65536
OFDM_RX_BW = int(1.5e6)
OFDM_TX_BW = int(1.5e6)
COUPLING_FRAMES = 40
BG_FRAMES = 60
SCENE_FRAMES = 20

# ---------- Gains ----------
TX_GAIN_A_CH0 = -25
TX_GAIN_A_CH1 = -25
TX_GAIN_B_CH0 = -89.75
TX_GAIN_B_CH1 = -89.75
RX_GAIN_A_CH0 = 40
RX_GAIN_A_CH1 = 40
RX_GAIN_B_CH0 = 40
RX_GAIN_B_CH1 = 40

# ---------- Common ----------
SETTLE_TIME_SEC = 2.0
FLUSH_COUNT = 50
WARN_DBFS = -60
C = 3e8
wavelength = C / FC
d = wavelength / 2
AOA_SIGN = -1
TX_SCALE = 1.0

active_sc = np.r_[np.arange(-50, 0), np.arange(1, 51)]
pilot_sc = np.array([-45, -35, -25, -15, -5, 5, 15, 25, 35, 45])
data_sc = np.array([k for k in active_sc if k not in pilot_sc])


def fmt_vec(v, prec=1):
    return "[" + ", ".join("{:.{p}f}".format(float(x), p=prec) for x in v) + "]"


# ============================================================
# KALMAN FILTER - AOA FUTURE PREDICTION
# State: [angle, angular_velocity] in [deg, deg/s]
# Constant-angular-rate model. No trajectory is constructed.
# ============================================================
class AngleKalmanFilter:
    def __init__(self, q_angle=KALMAN_Q_ANGLE, q_rate=KALMAN_Q_RATE,
                 r_angle=KALMAN_R_ANGLE):
        self.x = np.zeros((2, 1), dtype=float)
        self.P = np.diag([100.0, 100.0])
        self.q_angle = float(q_angle)
        self.q_rate = float(q_rate)
        self.r = float(r_angle)
        self.initialized = False

    def update(self, measurement_deg, dt):
        dt = float(np.clip(dt, 1e-3, 2.0))
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([[0.5 * dt * dt], [dt]])
        Q = (G @ G.T) * np.diag([self.q_angle, self.q_rate])
        # Keep Q positive and numerically stable.
        Q = np.diag([max(Q[0, 0], 1e-8), max(Q[1, 1], 1e-8)])

        if not self.initialized:
            self.x[:, 0] = [measurement_deg, 0.0]
            self.initialized = True
        else:
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q

        z = float(measurement_deg)
        H = np.array([[1.0, 0.0]])
        innovation = z - float(self.x[0, 0])
        # Angle is circular; choose the shortest residual.
        innovation = wrap180(innovation)
        S = float((H @ self.P @ H.T)[0, 0] + self.r)
        K = (self.P @ H.T) / max(S, 1e-9)
        self.x = self.x + K * innovation
        I = np.eye(2)
        self.P = (I - K @ H) @ self.P
        self.x[0, 0] = wrap180(self.x[0, 0])
        return float(self.x[0, 0]), float(self.x[1, 0])

    def predict_future(self, horizon_s=KALMAN_PREDICTION_HORIZON_S,
                       steps=KALMAN_PREDICTION_STEPS):
        steps = max(2, int(steps))
        times = np.linspace(0.0, float(horizon_s), steps + 1)[1:]
        angle0, rate = float(self.x[0, 0]), float(self.x[1, 0])
        angles = np.array([wrap180(angle0 + rate * t) for t in times])
        return times, angles


# ============================================================
# GUI dashboard
# LIGHT BLUE / WHITE THEME
# GUI LAYOUT ONLY
# ============================================================

_BG = "#F4F8FC"
_PANEL = "#FFFFFF"
_PANEL_BLUE = "#EAF3FF"
_FG = "#17324D"
_MUTE = "#60758A"

_ACCENT = "#1976D2"
_BLUE_DARK = "#0B4F9C"
_GREEN = "#2EAD63"
_AMBER = "#E7A21A"
_RED = "#D64545"
_GRID = "#D7E2EC"
_BORDER = "#B8CADB"


def _ber_color(b):
    if b is None:
        return _MUTE
    if b <= 0.10:
        return _GREEN
    if b <= 0.30:
        return _AMBER
    return _RED


def _pwr_color(p):
    if p > -6:
        return _AMBER
    if p < WARN_DBFS:
        return _RED
    return _ACCENT


def _combined_color(ratio_db, spread):
    echo_bad = ratio_db is not None and ratio_db < 5
    spread_bad = spread is not None and spread >= SPREAD_BAD_DEG

    if echo_bad or spread_bad:
        return _RED

    echo_ok = ratio_db is None or ratio_db >= 12
    spread_ok = spread is None or spread <= SPREAD_GOOD_DEG

    if echo_ok and spread_ok:
        return _GREEN

    return _AMBER


class ReflectorAoAGUI:
    """
    Visual dashboard only.

    The AoA/SDR/DSP code outside this class is unchanged.
    """

    def __init__(self, limit_deg=90):
        self.limit_deg = limit_deg
        self.hist = []
        self.tips = []

        self.HMAX = 60
        self.ECHO_MIN = -10
        self.ECHO_MAX = 25
        self.PMIN = -80
        self.PMAX = 0

        plt.ion()

        # ========================================================
        # MAIN WINDOW
        # ========================================================
        self.fig = plt.figure(figsize=(13.2, 7.9))
        self.fig.patch.set_facecolor(_BG)

        try:
            self.fig.canvas.manager.set_window_title(
                "FMCOMMS5 - Reflector AoA Dashboard"
            )
        except Exception:
            pass

        # Header
        self.fig.text(
            0.035, 0.962,
            "REFLECTOR ANGLE OF ARRIVAL",
            ha="left", va="center",
            color=_BLUE_DARK,
            fontsize=18,
            fontweight="bold",
        )

        self.fig.text(
            0.035, 0.928,
            "FMCOMMS5 / ZC702   •   MUSIC   •   ESPRIT   •   MVDR   •   ML",
            ha="left", va="center",
            color=_MUTE,
            fontsize=9.5,
        )

        self.fig.text(
            0.965, 0.962,
            "{}-RX".format(N_RX),
            ha="right", va="center",
            color=_ACCENT,
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=_PANEL_BLUE,
                edgecolor=_BORDER,
                linewidth=0.8,
            ),
        )

        # ========================================================
        # LEFT: LARGE AOA DIAL CARD
        # ========================================================
        self.ax = self.fig.add_axes(
            [0.035, 0.115, 0.555, 0.765]
        )

        ax = self.ax
        ax.set_facecolor(_PANEL)
        ax.set_aspect("equal")
        ax.axis("off")

        ax.set_xlim(-1.38, 1.38)
        ax.set_ylim(-0.68, 1.35)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(_BORDER)
            spine.set_linewidth(1.0)

        # Dial arcs
        arc = np.linspace(-90.0, 90.0, 361)

        outer_x = np.cos(np.deg2rad(90.0 - arc))
        outer_y = np.sin(np.deg2rad(90.0 - arc))

        ax.plot(
            outer_x, outer_y,
            color=_ACCENT,
            lw=3.5,
            zorder=1,
        )

        ax.plot(
            0.84 * outer_x,
            0.84 * outer_y,
            color=_GRID,
            lw=1.2,
            zorder=1,
        )

        # Ticks only. Numeric labels are positioned separately below.
        for a in range(-90, 91, 15):
            th = np.deg2rad(90.0 - a)
            major = (a % 45 == 0)

            r0 = 0.86 if major else 0.92

            ax.plot(
                [
                    r0 * np.cos(th),
                    np.cos(th),
                ],
                [
                    r0 * np.sin(th),
                    np.sin(th),
                ],
                color=_BLUE_DARK if major else _BORDER,
                lw=2.2 if major else 1.0,
                solid_capstyle="round",
                zorder=2,
            )

        # Major labels with safe, explicit positions.
        major_labels = (
            (-90, -1.10, 0.00, "right"),
            (-45, -0.79, 0.79, "center"),
            (0,    0.00, 1.045, "center"),
            (45,   0.79, 0.79, "center"),
            (90,   1.10, 0.00, "left"),
        )

        for a, x, y, ha in major_labels:
            ax.text(
                x, y,
                "0°" if a == 0 else "{:+d}°".format(a),
                ha=ha,
                va="center",
                color=_FG,
                fontsize=10.5,
                fontweight="bold" if a == 0 else "normal",
                zorder=4,
                clip_on=True,
            )

        # BROADSIDE is clearly separated from 0°.
        ax.text(
            0.0, 1.205,
            "BROADSIDE",
            ha="center",
            va="center",
            color=_BLUE_DARK,
            fontsize=9.5,
            fontweight="bold",
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor=_PANEL_BLUE,
                edgecolor=_BORDER,
                linewidth=0.8,
            ),
            clip_on=True,
        )

        # Needle
        self.trail, = ax.plot(
            [], [],
            color=_ACCENT,
            lw=2.0,
            alpha=0.22,
            zorder=3,
        )

        self.needle, = ax.plot(
            [0, 0],
            [0, 0.82],
            color=_ACCENT,
            lw=6,
            solid_capstyle="round",
            zorder=5,
        )

        self.future_needle, = ax.plot(
            [0, 0], [0, 0.82],
            color=_AMBER, lw=2.5, ls="--",
            solid_capstyle="round", zorder=4,
        )

        ax.plot(
            0, 0,
            "o",
            color=_BLUE_DARK,
            markeredgecolor=_PANEL,
            markeredgewidth=2,
            markersize=12,
            zorder=6,
        )

        self.angle_text = ax.text(
            0.0, -0.18,
            "--",
            ha="center",
            va="center",
            color=_BLUE_DARK,
            fontsize=38,
            fontweight="bold",
            zorder=6,
        )

        self.sub_text = ax.text(
            0.0, -0.36,
            "Waiting for measurement",
            ha="center",
            va="center",
            color=_MUTE,
            fontsize=10,
            zorder=6,
        )

        # Estimator card placed below the arc region, not over labels.
        self.methods_text = ax.text(
            -1.27, 0.56,
            "",
            ha="left",
            va="top",
            color=_FG,
            fontsize=9.5,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor=_PANEL_BLUE,
                edgecolor=_BORDER,
                linewidth=1.0,
            ),
            zorder=7,
        )

        self.title_text = self.fig.text(
            0.312, 0.865,
            "SYSTEM STARTING",
            ha="center",
            va="center",
            color=_FG,
            fontsize=12.5,
            fontweight="bold",
        )

        # ========================================================
        # RIGHT: FOUR SEPARATE, NON-OVERLAPPING CARDS
        # ========================================================

        # 1. Echo / Background
        self.ax_echo = self.fig.add_axes(
            [0.625, 0.735, 0.335, 0.130]
        )

        self._panel(
            self.ax_echo,
            "ECHO / BACKGROUND",
            "Signal confidence",
        )

        self.ax_echo.set_xlim(
            self.ECHO_MIN,
            self.ECHO_MAX,
        )
        self.ax_echo.set_ylim(0, 1)

        self.ax_echo.axvspan(
            self.ECHO_MIN, 5,
            color=_RED, alpha=0.10,
        )
        self.ax_echo.axvspan(
            5, 12,
            color=_AMBER, alpha=0.12,
        )
        self.ax_echo.axvspan(
            12, self.ECHO_MAX,
            color=_GREEN, alpha=0.10,
        )

        for s in range(
            self.ECHO_MIN,
            self.ECHO_MAX + 1,
            5,
        ):
            self.ax_echo.text(
                s, 0.09, str(s),
                ha="center",
                va="bottom",
                color=_MUTE,
                fontsize=7,
            )

        self.echo_marker, = self.ax_echo.plot(
            [self.ECHO_MIN, self.ECHO_MIN],
            [0.22, 0.86],
            color=_ACCENT,
            lw=3.0,
        )

        self.echo_val = self.ax_echo.text(
            self.ECHO_MAX - 0.7,
            0.58,
            "-- dB",
            ha="right",
            va="center",
            color=_BLUE_DARK,
            fontsize=13,
            fontweight="bold",
        )

        # 2. Estimator agreement / BER
        self.ax_spread = self.fig.add_axes(
            [0.625, 0.555, 0.335, 0.130]
        )

        self._panel(
            self.ax_spread,
            "ESTIMATOR AGREEMENT",
            "Spread and BER",
        )

        self.ax_spread.set_xlim(0, 30)
        self.ax_spread.set_ylim(0, 1)

        self.ax_spread.axvspan(
            0, SPREAD_GOOD_DEG,
            color=_GREEN, alpha=0.10,
        )
        self.ax_spread.axvspan(
            SPREAD_GOOD_DEG,
            SPREAD_BAD_DEG,
            color=_AMBER, alpha=0.12,
        )
        self.ax_spread.axvspan(
            SPREAD_BAD_DEG, 30,
            color=_RED, alpha=0.10,
        )

        for s in range(0, 31, 5):
            self.ax_spread.text(
                s, 0.09, str(s),
                ha="center",
                va="bottom",
                color=_MUTE,
                fontsize=7,
            )

        self.spread_marker, = self.ax_spread.plot(
            [0, 0],
            [0.22, 0.86],
            color=_ACCENT,
            lw=3.0,
        )

        self.ber_text = self.ax_spread.text(
            29,
            0.58,
            "BER --",
            ha="right",
            va="center",
            color=_MUTE,
            fontsize=11,
            fontweight="bold",
        )

        # 3. RX power
        self.ax_pwr = self.fig.add_axes(
            [0.625, 0.315, 0.335, 0.170]
        )

        self._panel(
            self.ax_pwr,
            "RX POWER",
            "dBFS",
        )

        self.ax_pwr.set_ylim(
            self.PMIN,
            self.PMAX,
        )
        self.ax_pwr.set_xlim(
            -0.6,
            N_RX - 0.4,
        )

        self.ax_pwr.set_yticks(
            [-80, -60, -40, -20, 0]
        )
        self.ax_pwr.set_xticks(
            range(N_RX)
        )
        self.ax_pwr.set_xticklabels(
            ["RX{}".format(i) for i in range(N_RX)]
        )

        self.ax_pwr.tick_params(
            colors=_MUTE,
            labelsize=8,
            pad=2,
        )

        self.ax_pwr.grid(
            axis="y",
            color=_GRID,
            linewidth=0.7,
            alpha=0.65,
        )

        self.ax_pwr.axhline(
            WARN_DBFS,
            color=_RED,
            lw=1.1,
            ls="--",
            alpha=0.8,
        )

        self.pwr_bars = self.ax_pwr.bar(
            range(N_RX),
            [0] * N_RX,
            bottom=self.PMIN,
            color=_ACCENT,
            width=0.52,
            edgecolor=_BLUE_DARK,
            linewidth=0.7,
        )

        self.pwr_labels = [
            self.ax_pwr.text(
                i,
                self.PMIN,
                "",
                ha="center",
                va="bottom",
                color=_FG,
                fontsize=8,
                fontweight="bold",
                clip_on=True,
            )
            for i in range(N_RX)
        ]

        # 4. Angle history - moved lower so it is completely separated
        # from RX0/RX1 labels.
        self.ax_spark = self.fig.add_axes(
            [0.625, 0.065, 0.335, 0.170]
        )

        self._panel(
            self.ax_spark,
            "ANGLE HISTORY",
            "Last 60 updates",
        )

        self.ax_spark.set_xlim(
            0,
            self.HMAX,
        )
        self.ax_spark.set_ylim(
            -90,
            90,
        )
        self.ax_spark.set_yticks(
            [-90, -45, 0, 45, 90]
        )

        self.ax_spark.tick_params(
            colors=_MUTE,
            labelsize=8,
            pad=2,
        )

        self.ax_spark.grid(
            axis="y",
            color=_GRID,
            linewidth=0.7,
            alpha=0.65,
        )

        self.spark, = self.ax_spark.plot(
            [],
            [],
            color=_ACCENT,
            lw=2.0,
        )

        self.spark_dot, = self.ax_spark.plot(
            [],
            [],
            "o",
            color=_RED,
            markersize=5,
        )

        self.spark_future, = self.ax_spark.plot(
            [], [], color=_AMBER, lw=2.0, ls="--",
        )

        self.future_angle_text = self.fig.text(
            0.795, 0.260, "FUTURE +1.0s: --",
            ha="center", va="center", color=_BLUE_DARK,
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=_PANEL_BLUE,
                      edgecolor=_BORDER, linewidth=0.8),
        )

        # Bottom status bar, deliberately below every chart.
        self.status_text = self.fig.text(
            0.312, 0.030,
            "Starting...",
            ha="center",
            va="center",
            color=_MUTE,
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=_PANEL_BLUE,
                edgecolor=_BORDER,
                linewidth=0.8,
            ),
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _panel(self, ax, title, subtitle=""):
        ax.set_facecolor(_PANEL)
        ax.set_xticks([])
        ax.set_yticks([])

        for sp in ax.spines.values():
            sp.set_color(_BORDER)
            sp.set_linewidth(1.0)

        # Titles stay inside their own card.
        ax.text(
            0.0,
            1.045,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=_BLUE_DARK,
            fontsize=9.5,
            fontweight="bold",
            clip_on=False,
        )

        if subtitle:
            ax.text(
                1.0,
                1.045,
                subtitle,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color=_MUTE,
                fontsize=7,
                clip_on=False,
            )

    def alive(self):
        return plt.fignum_exists(
            self.fig.number
        )

    def _draw(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _set_powers(self, powers):
        for i, b in enumerate(self.pwr_bars):
            if i >= len(powers):
                continue

            pv = float(
                np.clip(
                    powers[i],
                    self.PMIN,
                    self.PMAX,
                )
            )

            b.set_height(
                pv - self.PMIN
            )

            b.set_color(
                _pwr_color(powers[i])
            )

            self.pwr_labels[i].set_text(
                "{:.0f}".format(
                    powers[i]
                )
            )

            self.pwr_labels[i].set_y(
                min(
                    pv + 3,
                    self.PMAX - 3,
                )
            )

    def update_status(
        self,
        title=None,
        status=None,
        powers=None,
        frames=None,
    ):
        if not self.alive():
            return

        if title is not None:
            self.title_text.set_text(
                title
            )

        if powers is not None:
            self._set_powers(
                powers
            )

        parts = []

        if status is not None:
            parts.append(status)

        if frames is not None:
            parts.append(
                "frames={}".format(
                    frames
                )
            )

        if parts:
            self.status_text.set_text(
                "   •   ".join(parts)
            )

        self._draw()

    def update_angle(
        self,
        angle_deg,
        ratio_db=None,
        powers=None,
        frames=None,
        ber=None,
        methods=None,
        spread=None,
        filtered_angle_deg=None,
        future_times=None,
        future_angles=None,
    ):
        if not self.alive():
            return

        if filtered_angle_deg is None:
            filtered_angle_deg = angle_deg

        col = _combined_color(
            ratio_db,
            spread,
        )

        # Existing angle mapping is unchanged.
        a_clip = float(
            np.clip(
                filtered_angle_deg,
                -self.limit_deg,
                self.limit_deg,
            )
        )

        th = np.deg2rad(
            90 - a_clip
        )

        tx = 0.82 * np.cos(th)
        ty = 0.82 * np.sin(th)

        self.needle.set_data(
            [0, tx],
            [0, ty],
        )

        self.needle.set_color(
            col
        )

        self.tips.append(
            (tx, ty)
        )
        self.tips = self.tips[-10:]

        self.trail.set_data(
            [p[0] for p in self.tips],
            [p[1] for p in self.tips],
        )

        self.trail.set_color(
            col
        )

        self.angle_text.set_text(
            "{:+.1f}°".format(
                filtered_angle_deg
            )
        )

        self.angle_text.set_color(
            col
        )

        sub = []

        if ratio_db is not None:
            sub.append(
                "Echo {:.1f} dB".format(
                    ratio_db
                )
            )

        if spread is not None:
            sub.append(
                "Spread {:.1f}°".format(
                    spread
                )
            )

        self.sub_text.set_text(
            "     ".join(sub)
            if sub
            else "Waiting for measurement"
        )

        self.title_text.set_text(
            "LIVE AOA MEASUREMENT ({}-RX)".format(
                N_RX
            )
        )

        if methods is not None:

            def g(k):
                return methods.get(
                    k,
                    np.nan,
                )

            self.methods_text.set_text(
                " ESTIMATORS\n"
                " ───────────────\n"
                " MUSIC   {:+6.1f}°\n"
                " ESPRIT  {:+6.1f}°\n"
                " MVDR    {:+6.1f}°\n"
                " ML      {:+6.1f}°\n"
                " ───────────────\n"
                " FUSED   {:+6.1f}°\n"
                " KALMAN  {:+6.1f}°"
                .format(
                    g("MUSIC"),
                    g("ESPRIT"),
                    g("MVDR"),
                    g("ML"),
                    angle_deg,
                    filtered_angle_deg,
                )
            )

        if ratio_db is not None:
            v = float(
                np.clip(
                    ratio_db,
                    self.ECHO_MIN,
                    self.ECHO_MAX,
                )
            )

            self.echo_marker.set_xdata(
                [v, v]
            )
            self.echo_marker.set_color(
                col
            )

            self.echo_val.set_text(
                "{:+.1f} dB".format(
                    ratio_db
                )
            )

            self.echo_val.set_color(
                col
            )

        if spread is not None:
            sv = float(
                np.clip(
                    spread,
                    0,
                    30,
                )
            )

            self.spread_marker.set_xdata(
                [sv, sv]
            )

            self.spread_marker.set_color(
                col
            )

        if ber is not None:
            self.ber_text.set_text(
                "BER {:.3f}".format(
                    ber
                )
            )

            self.ber_text.set_color(
                _ber_color(ber)
            )

        if powers is not None:
            self._set_powers(
                powers
            )

        self.hist.append(
            filtered_angle_deg
        )

        self.hist = self.hist[-self.HMAX:]

        xs = np.arange(
            len(self.hist)
        )

        self.spark.set_data(
            xs,
            self.hist,
        )

        self.spark.set_color(
            col
        )

        if len(xs) > 0:
            self.spark_dot.set_data(
                [xs[-1]],
                [self.hist[-1]],
            )

            self.spark_dot.set_color(
                col
            )

        parts = []

        if frames is not None:
            parts.append(
                "Valid frames={}".format(
                    frames
                )
            )

        parts.append(
            "Status: {}".format(
                "GOOD"
                if col == _GREEN
                else "CHECK"
            )
        )

        self.status_text.set_text(
            "   •   ".join(parts)
        )

        if future_times is not None and future_angles is not None and len(future_times):
            future_x = len(self.hist) - 1 + np.arange(1, len(future_angles) + 1)
            self.spark_future.set_data(future_x, future_angles)
            pred_clip = float(np.clip(future_angles[-1], -self.limit_deg, self.limit_deg))
            pred_th = np.deg2rad(90.0 - pred_clip)
            self.future_needle.set_data(
                [0, 0.82 * np.cos(pred_th)],
                [0, 0.82 * np.sin(pred_th)],
            )
            self.future_angle_text.set_text(
                "FUTURE +{:.1f}s: {:+.1f}°".format(
                    float(future_times[-1]), float(future_angles[-1])
                )
            )
        self._draw()


# ============================================================
# DSP helpers
# ============================================================
def wrap180(x): return ((x + 180) % 360) - 180


def circular_mean_deg(a):
    z = np.mean(np.exp(1j * np.deg2rad(a)))
    return wrap180(np.rad2deg(np.angle(z)))


def sc_to_bin(sc): return sc + NFFT // 2
def rms_dbfs(x): return 20 * np.log10(np.sqrt(np.mean(np.abs(x) ** 2)) / 2048 + 1e-12)


def estimate_tone_bin(x):
    n = len(x); win = np.hamming(n)
    X = np.fft.fftshift(np.fft.fft(x * win))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1 / FS))
    k = np.argmin(np.abs(freqs - TONE_OFFSET))
    return X[k], freqs[k], 20 * np.log10(np.abs(X[k]) / (np.sum(win) * 2**11) + 1e-15)


def estimate_tone_nch(data_n):
    ph, po = [], []
    for ch in range(N_RX):
        a, _, p = estimate_tone_bin(data_n[ch]); ph.append(a); po.append(p)
    return np.array(ph), np.array(po)


def rel_phase_deg(ph):
    rel = ph * np.conj(ph[0]); out = [0.0]
    for k in range(1, len(ph)):
        out.append(wrap180(np.rad2deg(np.angle(rel[k]))))
    return np.array(out)


def qpsk_mod(bits):
    bits = bits.reshape(-1, 2)
    return ((1 - 2 * bits[:, 0]) + 1j * (1 - 2 * bits[:, 1])) / np.sqrt(2)


def qpsk_demod(s):
    return np.column_stack(((np.real(s) < 0).astype(int), (np.imag(s) < 0).astype(int))).reshape(-1)


def ofdm_mod_symbol(X):
    x = np.fft.ifft(np.fft.ifftshift(X)); return np.r_[x[-CP:], x]


def ofdm_demod_symbol(x): return np.fft.fftshift(np.fft.fft(x[CP:CP + NFFT]))


def make_ofdm_frame():
    rng = np.random.default_rng(1234)
    Xsync = np.zeros(NFFT, dtype=complex)
    sv = np.ones(len(active_sc), dtype=complex); sv[1::2] = -1
    for sc, val in zip(active_sc, sv): Xsync[sc_to_bin(sc)] = val
    Xref = np.zeros(NFFT, dtype=complex)
    for sc in active_sc: Xref[sc_to_bin(sc)] = 1 + 0j
    nb = len(data_sc) * NUM_DATA_SYMS * 2
    tx_bits = rng.integers(0, 2, nb); tx_syms = qpsk_mod(tx_bits)
    fd = [Xsync, Xref]; ptr = 0
    for _ in range(NUM_DATA_SYMS):
        X = np.zeros(NFFT, dtype=complex)
        for sc in pilot_sc: X[sc_to_bin(sc)] = 1 + 0j
        for sc in data_sc: X[sc_to_bin(sc)] = tx_syms[ptr]; ptr += 1
        fd.append(X)
    td = np.concatenate([ofdm_mod_symbol(X) for X in fd])
    td = td / np.max(np.abs(td)) * (2 ** 14)
    return td.astype(np.complex64), Xsync, Xref, tx_bits


def find_frame_start(rx, sync_td):
    c = np.abs(np.correlate(rx, sync_td, mode="valid"))
    return int(np.argmax(c)), float(np.max(c))


def array_relphase(H):
    out = [0.0]
    for k in range(1, N_ELEM):
        out.append(wrap180(np.rad2deg(np.angle(np.sum(H[k] * np.conj(H[0]))))))
    return np.array(out)


# ============================================================
# SDR TX helpers
# ============================================================
def stop_tx():
    try: sdr.tx_destroy_buffer()
    except Exception: pass
    try: sdr.dds_enabled = [0] * len(sdr.dds_enabled)
    except Exception: pass


def flush_rx(n=FLUSH_COUNT):
    for _ in range(n): sdr.rx()


def start_single_tx_ofdm(tx_frame, tx_index):
    stop_tx(); sdr.tx_enabled_channels = [0, 1]; sdr.tx_cyclic_buffer = True
    z = np.zeros_like(tx_frame)
    sdr.tx([tx_frame, z] if tx_index == 0 else [z, tx_frame])


def start_weighted_dual_tx_ofdm(tx_frame, w):
    stop_tx(); sdr.tx_enabled_channels = [0, 1]; sdr.tx_cyclic_buffer = True
    x0 = (TX_SCALE * w[0] * tx_frame).astype(np.complex64)
    x1 = (TX_SCALE * w[1] * tx_frame).astype(np.complex64)
    peak = max(np.max(np.abs(x0)), np.max(np.abs(x1)), 1.0)
    if peak > (2 ** 14):
        x0 = x0 / peak * (2 ** 14); x1 = x1 / peak * (2 ** 14)
    sdr.tx([x0.astype(np.complex64), x1.astype(np.complex64)])


# ============================================================
# OFDM capture
# ============================================================
def estimate_H_one_frame(rxf, Xref):
    H = np.zeros((N_ELEM, len(active_sc)), dtype=complex)
    for ch in range(N_ELEM):
        Y = ofdm_demod_symbol(rxf[ch][SYM_LEN:2 * SYM_LEN])
        for i, sc in enumerate(active_sc):
            H[ch, i] = Y[sc_to_bin(sc)] / (Xref[sc_to_bin(sc)] + 1e-12)
    return H


def capture_H(n_frames, gui=None, title="CAPTURE"):
    Hsum = np.zeros((N_ELEM, len(active_sc)), dtype=complex)
    href = None; powers = np.zeros(N_ELEM); got = 0
    for _ in range(n_frames):
        data = sdr.rx(); rx = [np.array(data[i]) for i in range(N_RX)]
        start, _ = find_frame_start(rx[0], sync_td)
        if start + frame_len > len(rx[0]):
            if gui: gui.update_status(title=title, status="frame cut", frames="{}/{}".format(got, n_frames))
            continue
        rxf = [r[start:start + frame_len] for r in rx]
        H = estimate_H_one_frame(rxf, Xref)
        if href is None: href = H.copy(); Hsum += H
        else:
            ph = np.angle(np.sum(H * np.conj(href)) + 1e-12); Hsum += H * np.exp(-1j * ph)
        pnow = np.array([rms_dbfs(r) for r in rxf]); powers += pnow; got += 1
        if gui: gui.update_status(title=title, status="capturing", powers=pnow, frames="{}/{}".format(got, n_frames))
    if got == 0: return None, None, 0
    return Hsum / got, powers / got, got


def capture_H_and_decode(n_frames, gui=None, title="SCENE"):
    Hsum = np.zeros((N_ELEM, len(active_sc)), dtype=complex)
    href = None; powers = np.zeros(N_ELEM); got = 0; ber_list = []
    ref_bins = [sc_to_bin(sc) for sc in active_sc]
    for _ in range(n_frames):
        data = sdr.rx(); rx = [np.array(data[i]) for i in range(N_RX)]
        start, _ = find_frame_start(rx[0], sync_td)
        if start + frame_len > len(rx[0]):
            if gui: gui.update_status(title=title, status="frame cut", frames="{}/{}".format(got, n_frames))
            continue
        rxf = [r[start:start + frame_len] for r in rx]
        H = estimate_H_one_frame(rxf, Xref)
        if href is None: href = H.copy(); Hsum += H
        else:
            ph = np.angle(np.sum(H * np.conj(href)) + 1e-12); Hsum += H * np.exp(-1j * ph)
        Y0 = np.zeros((NUM_DATA_SYMS + 2, NFFT), dtype=complex)
        for s in range(NUM_DATA_SYMS + 2):
            Y0[s, :] = ofdm_demod_symbol(rxf[0][s * SYM_LEN:(s + 1) * SYM_LEN])
        H0 = np.zeros(NFFT, dtype=complex)
        for b in ref_bins: H0[b] = Y0[1, b] / (Xref[b] + 1e-12)
        rx_bits = []
        for s in range(2, NUM_DATA_SYMS + 2):
            Xeq = np.zeros(NFFT, dtype=complex)
            for b in ref_bins: Xeq[b] = Y0[s, b] / (H0[b] + 1e-12)
            rx_bits.extend(qpsk_demod(np.array([Xeq[sc_to_bin(sc)] for sc in data_sc])))
        rx_bits = np.array(rx_bits[:len(tx_bits)])
        if len(rx_bits) == len(tx_bits): ber_list.append(np.mean(rx_bits != tx_bits))
        pnow = np.array([rms_dbfs(r) for r in rxf]); powers += pnow; got += 1
        if gui: gui.update_status(title=title, status="capturing", powers=pnow, frames="{}/{}".format(got, n_frames))
    if got == 0: return None, None, 0, None
    ber = float(np.mean(ber_list)) if ber_list else None
    return Hsum / got, powers / got, got, ber


# ============================================================
# Dual-TX leakage cancellation
# ============================================================
def estimate_coupling_column(tx_frame_cyclic, tx_index, gui=None):
    label = "COUPLING TX{}".format(tx_index)
    print("\n--- {} ---".format(label))
    if gui: gui.update_status(title=label, status="TX{} only".format(tx_index))
    start_single_tx_ofdm(tx_frame_cyclic, tx_index)
    time.sleep(SETTLE_TIME_SEC); flush_rx()
    H, p, n = capture_H(COUPLING_FRAMES, gui=gui, title=label)
    if H is None: raise RuntimeError("No frames for TX{} coupling".format(tx_index))
    h = np.mean(H, axis=1)
    print("{} powers: {} dBFS | |h|={}".format(label, fmt_vec(p), np.array2string(np.abs(h), precision=2)))
    return h, p


def compute_tx_weights_from_coupling(Hc):
    U, S, Vh = np.linalg.svd(Hc, full_matrices=True)
    w = Vh.conj().T[:, -1]; w = w / (np.linalg.norm(w) + 1e-12)
    w = w * np.exp(-1j * np.angle(w[0]))
    pb = np.linalg.norm(Hc @ np.array([1.0 + 0j, 0.0 + 0j])) ** 2
    pa = np.linalg.norm(Hc @ w) ** 2
    return w, S, 10 * np.log10((pb + 1e-12) / (pa + 1e-12))


# ============================================================
# Covariance + FOUR DOA estimators
# ============================================================
def _steer(theta_deg):
    k = np.arange(N_ELEM)
    return np.exp(1j * 2 * np.pi * (d / wavelength) * k * np.sin(np.deg2rad(theta_deg)))


def _covariance(H_t):
    c = np.exp(-1j * np.deg2rad(phase_cal))          # apply phase calibration
    X = H_t * c[:, None]
    R = (X @ X.conj().T) / X.shape[1]
    J = np.fliplr(np.eye(N_ELEM))                    # forward-backward averaging
    R = 0.5 * (R + J @ R.conj() @ J)
    R = R + LOADING_FRAC * (np.real(np.trace(R)) / N_ELEM) * np.eye(N_ELEM)   # diagonal loading
    return R


def doa_music(R, K=1):
    K = min(K, N_ELEM - 1)
    _, evec = np.linalg.eigh(R)
    En = evec[:, :N_ELEM - K]
    Cn = En @ En.conj().T
    M = N_ELEM
    coeff = np.array([np.trace(Cn, offset=l) for l in range(-(M - 1), M)])
    roots = np.roots(coeff[::-1])
    inside = roots[np.abs(roots) < 1.0]
    if inside.size == 0: return np.nan
    z = inside[np.argmax(np.abs(inside))]
    s = np.clip(np.angle(z) / (2 * np.pi * (d / wavelength)), -1, 1)
    return AOA_SIGN * np.rad2deg(np.arcsin(s))


def doa_esprit(R, K=1):
    K = min(K, N_ELEM - 1)
    _, evec = np.linalg.eigh(R)
    Es = evec[:, N_ELEM - K:]                # signal subspace (largest K)
    Es1, Es2 = Es[:-1, :], Es[1:, :]
    Psi = np.linalg.pinv(Es1) @ Es2
    eig = np.linalg.eigvals(Psi)
    z = eig[np.argmax(np.abs(eig))]
    s = np.clip(np.angle(z) / (2 * np.pi * (d / wavelength)), -1, 1)
    return AOA_SIGN * np.rad2deg(np.arcsin(s))


def doa_mvdr(R, K=1):
    Rinv = np.linalg.inv(R)
    P = np.array([1.0 / np.real(np.conj(_steer(t)) @ Rinv @ _steer(t)) for t in SCAN_DEG])
    return AOA_SIGN * SCAN_DEG[int(np.argmax(P))]


def doa_ml(R, K=1):
    # deterministic single-source ML == conventional (Bartlett) beamformer peak
    P = np.array([np.real(np.conj(_steer(t)) @ R @ _steer(t)) for t in SCAN_DEG])
    return AOA_SIGN * SCAN_DEG[int(np.argmax(P))]


def fuse(angles):
    vals = np.array([v for v in angles.values() if np.isfinite(v)])
    if vals.size == 0:
        return np.nan, np.nan
    med = np.median(vals)
    spread = float(np.max(vals) - np.min(vals))
    if DISPLAY_SCHEME == "median":
        return float(med), spread
    if DISPLAY_SCHEME == "music":
        m = angles.get("MUSIC", med)
        return float(m if np.isfinite(m) else med), spread
    keep = vals[np.abs(vals - med) <= CONSENSUS_TOL_DEG]    # consensus
    if keep.size == 0: keep = vals
    return float(np.mean(keep)), spread


def estimate_all_doa(H_t, K=1):
    R = _covariance(H_t)
    angles = {}
    for name, fn in (("MUSIC", doa_music), ("ESPRIT", doa_esprit),
                     ("MVDR", doa_mvdr), ("ML", doa_ml)):
        try:
            angles[name] = float(fn(R, K))
        except Exception:
            angles[name] = np.nan
    fused, spread = fuse(angles)
    return angles, fused, spread


def reflector_residual(H_scene, H_bg):
    ph = np.angle(np.sum(H_scene * np.conj(H_bg)) + 1e-12)
    H_t = H_scene * np.exp(-1j * ph) - H_bg
    ratio_db = 20 * np.log10(np.linalg.norm(H_t) / (np.linalg.norm(H_bg) + 1e-12) + 1e-12)
    return H_t, ratio_db


# ============================================================
# SDR setup
# ============================================================
print("Connecting to FMCOMMS5/ZC702:", URI, "| N_RX =", N_RX)
gui = ReflectorAoAGUI(limit_deg=90)
gui.update_status(title="Connecting", status="Connecting ({}-RX)".format(N_RX))

sdr = adi.FMComms5(uri=URI)
sdr.rx_enabled_channels = RX_CHANNELS
sdr.sample_rate = FS
sdr.rx_lo = FC; sdr.rx_lo_chip_b = FC; sdr.tx_lo = FC; sdr.tx_lo_chip_b = FC
sdr.gain_control_mode_chan0 = "manual"; sdr.gain_control_mode_chan1 = "manual"
sdr.gain_control_mode_chip_b_chan0 = "manual"; sdr.gain_control_mode_chip_b_chan1 = "manual"
sdr.rx_hardwaregain_chan0 = RX_GAIN_A_CH0; sdr.rx_hardwaregain_chan1 = RX_GAIN_A_CH1
sdr.rx_hardwaregain_chip_b_chan0 = RX_GAIN_B_CH0; sdr.rx_hardwaregain_chip_b_chan1 = RX_GAIN_B_CH1
sdr.tx_hardwaregain_chan0 = TX_GAIN_A_CH0; sdr.tx_hardwaregain_chan1 = TX_GAIN_A_CH1
sdr.tx_hardwaregain_chip_b_chan0 = TX_GAIN_B_CH0; sdr.tx_hardwaregain_chip_b_chan1 = TX_GAIN_B_CH1
try: sdr._rxadc.set_kernel_buffers_count(1)
except Exception: pass
gui.update_status(title="Connected", status="ready ({}-RX)".format(N_RX))


# ============================================================
# STAGE 1: Tone phase calibration
# ============================================================
print("\n===== STAGE 1: TONE PHASE CALIBRATION =====")
sdr.rx_buffer_size = TONE_NUM_SAMPLES
sdr.rx_rf_bandwidth = TONE_RX_BW; sdr.rx_rf_bandwidth_chip_b = TONE_RX_BW
sdr.tx_rf_bandwidth = TONE_TX_BW; sdr.tx_rf_bandwidth_chip_b = TONE_TX_BW
stop_tx(); sdr.tx_enabled_channels = [0]; sdr.dds_single_tone(TONE_OFFSET, 0.9)
print("Array: {} elements (RX {}) | spacing {:.2f} mm | AOA_SIGN {}".format(N_ELEM, RX_CHANNELS, d * 1000, AOA_SIGN))
print("Place TX source at BROADSIDE. Do not move RX cables after this.")
gui.update_status(title="Stage 1: Calibration", status="TX at broadside, press Enter")
input("Press Enter when ready...")
time.sleep(SETTLE_TIME_SEC); flush_rx()
cal_relph = []; cal_powers = []
for idx in range(CAL_AVG_COUNT):
    data = sdr.rx(); data = np.array([data[i] for i in range(N_RX)])
    phasors, p_dbfs = estimate_tone_nch(data)
    cal_relph.append(rel_phase_deg(phasors)); cal_powers.append(p_dbfs)
    gui.update_status(title="Stage 1: Calibration", status="calibrating", powers=p_dbfs,
                      frames="{}/{}".format(idx + 1, CAL_AVG_COUNT))
    time.sleep(0.03)
cal_relph = np.array(cal_relph); cal_powers = np.array(cal_powers)
phase_cal = np.array([circular_mean_deg(cal_relph[:, k]) for k in range(N_ELEM)])
mean_cal_p = np.mean(cal_powers, axis=0)
print("phase_cal:", fmt_vec(phase_cal), "| powers:", fmt_vec(mean_cal_p), "dBFS")
gui.update_status(title="Stage 1 done", status="calibration ready", powers=mean_cal_p)
stop_tx(); time.sleep(0.5)


# ============================================================
# Build OFDM waveform
# ============================================================
tx_frame, Xsync, Xref, tx_bits = make_ofdm_frame()
sync_td = ofdm_mod_symbol(Xsync); frame_len = len(tx_frame)
tx_frame_cyclic = np.tile(tx_frame, 4).astype(np.complex64)
sdr.rx_buffer_size = OFDM_RX_BUFFER_SIZE
sdr.rx_rf_bandwidth = OFDM_RX_BW; sdr.rx_rf_bandwidth_chip_b = OFDM_RX_BW
sdr.tx_rf_bandwidth = OFDM_TX_BW; sdr.tx_rf_bandwidth_chip_b = OFDM_TX_BW


# ============================================================
# STAGE 2: Coupling matrix
# ============================================================
print("\n===== STAGE 2: TX->RX COUPLING =====")
print("Clear the scene (no reflector).")
gui.update_status(title="Stage 2: Coupling", status="clear scene, press Enter")
input("Press Enter to estimate coupling...")
time.sleep(SETTLE_TIME_SEC); flush_rx()
h0, _ = estimate_coupling_column(tx_frame_cyclic, 0, gui=gui)
h1, _ = estimate_coupling_column(tx_frame_cyclic, 1, gui=gui)
Hc = np.column_stack([h0, h1])


# ============================================================
# STAGE 3: SVD weights
# ============================================================
print("\n===== STAGE 3: SVD TX WEIGHTS =====")
w, S, predicted_supp_db = compute_tx_weights_from_coupling(Hc)
print("Singular values:", np.array2string(S, precision=3),
      "| predicted suppression {:.1f} dB".format(predicted_supp_db))
gui.update_status(title="Stage 3 done", status="weights ready ({:.1f} dB)".format(predicted_supp_db))
start_weighted_dual_tx_ofdm(tx_frame_cyclic, w)
time.sleep(SETTLE_TIME_SEC); flush_rx()
_, p_check, _ = capture_H(10, gui=gui, title="Weighted Check")
if p_check is not None: print("Weighted-TX RX power:", fmt_vec(p_check), "dBFS")


# ============================================================
# STAGE 4: Weighted OFDM TX
# ============================================================
print("\n===== STAGE 4: WEIGHTED OFDM TX =====")
start_weighted_dual_tx_ofdm(tx_frame_cyclic, w)
gui.update_status(title="Stage 4", status="weighted OFDM running")
time.sleep(SETTLE_TIME_SEC); flush_rx()


# ============================================================
# STAGE 5: Background capture (scene already clear from Stage 2)
# ============================================================
print("\n===== STAGE 5: BACKGROUND CAPTURE =====")
gui.update_status(title="Stage 5: Background", status="capturing background")
flush_rx()
H_bg, bg_p, bg_n = capture_H(BG_FRAMES, gui=gui, title="Background")
if H_bg is None: raise RuntimeError("No frames during background capture.")
print("Background from {} frames | power {} dBFS".format(bg_n, fmt_vec(bg_p)))
gui.update_status(title="Stage 5 done", status="background ready", powers=bg_p, frames=bg_n)


# ============================================================
# STAGE 6-8: Live AoA with all four estimators
# ============================================================
print("\n===== LIVE: 4-estimator reflector AoA (Ctrl+C to stop) =====")
print("Place the reflector. Display scheme: {}\n".format(DISPLAY_SCHEME))
gui.update_status(title="Live AoA", status="place reflector")
angle_kalman = AngleKalmanFilter()
kalman_previous_time = time.monotonic()
try:
    while True:
        H_scene, sc_p, sc_n, ber = capture_H_and_decode(SCENE_FRAMES, gui=gui, title="Live Scene")
        if H_scene is None:
            gui.update_status(title="Live AoA", status="no frames, skipping"); continue

        H_t, ratio_db = reflector_residual(H_scene, H_bg)
        methods, fused, spread = estimate_all_doa(H_t, K=DOA_K)

        print("AoA({}-RX) fused={:7.2f} | M={:7.2f} E={:7.2f} V={:7.2f} L={:7.2f} | "
              "spread={:5.1f} | echo/bg={:6.1f}dB | BER={} | P={}"
              .format(N_RX, fused, methods["MUSIC"], methods["ESPRIT"],
                      methods["MVDR"], methods["ML"], spread, ratio_db,
                      "{:.4f}".format(ber) if ber is not None else "NA", fmt_vec(sc_p)))

        now_kalman = time.monotonic()
        dt_kalman = max(now_kalman - kalman_previous_time, 1e-3)
        kalman_previous_time = now_kalman
        filtered_angle, angular_rate = angle_kalman.update(fused, dt_kalman)
        future_t, future_angle = angle_kalman.predict_future()

        print("KALMAN | filtered={:+7.2f} deg | angular rate={:+7.2f} deg/s | "
              "future +{:.1f}s={:+7.2f} deg"
              .format(filtered_angle, angular_rate, future_t[-1], future_angle[-1]))

        gui.update_angle(angle_deg=fused, ratio_db=ratio_db, powers=sc_p, frames=sc_n,
                         ber=ber, methods=methods, spread=spread,
                         filtered_angle_deg=filtered_angle,
                         future_times=future_t, future_angles=future_angle)

        if ratio_db < -40:
            print("  weak echo: stronger/closer reflector or recapture background.")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    stop_tx()
    gui.update_status(title="Stopped", status="TX stopped")
    try:
        plt.ioff(); plt.show()
    except Exception:
        pass