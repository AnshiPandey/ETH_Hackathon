# Intelligent ISAC-Based Drone Detection & Tracking

> From radar signals to real-time drone intelligence — detect, localize, track, and predict.

## 🚀 Overview

This project is an **Integrated Sensing and Communication (ISAC)-based radar system** for real-time drone detection, localization, tracking, and motion prediction.

The system combines **OFDM-based Angle of Arrival (AoA)** with **two-tone Range-Doppler sensing** to estimate the three key parameters of a target:

**Angle • Distance • Velocity**

Because real-world radar measurements are noisy and fluctuate over time, **Kalman filtering** is used to stabilize the measurements and estimate the target's motion state. The system further uses the filtered state for **short-term trajectory prediction** and displays the results through a real-time dashboard.

---

## 🎯 What the System Does

```text
        Radar / SDR Signals
                │
        ┌───────┴────────┐
        ▼                ▼
   OFDM Waveform    Two-Tone Waveform
        │                │
        ▼                ▼
   AoA Estimation   Range + Doppler
        │                │
        │                ▼
        │          Velocity Estimation
        │                │
        └───────┬────────┘
                ▼
         Kalman Filtering
                │
                ▼
       Target State Estimate
                │
                ▼
       Motion Prediction
                │
                ▼
       Real-Time Dashboard
```

---

## 🔬 Core Technologies

| Parameter       | Method                      |
| --------------- | --------------------------- |
| Angle           | OFDM-based AoA Estimation   |
| Distance        | Two-Tone Range Estimation   |
| Velocity        | Doppler Processing          |
| Noise Reduction | Kalman Filter               |
| Tracking        | Sequential State Estimation |
| Prediction      | Future Motion Estimation    |
| Visualization   | Real-Time Dashboard         |

---

## 💡 Why It Matters

Traditional detection can answer **"Is there a drone?"**

Our approach aims to answer:

**"Where is it, how fast is it moving, and where is it likely to go next?"**

Combining **Angle + Distance + Velocity** provides a richer understanding of target motion, while Kalman filtering improves stability in noisy sensing environments.

### Potential Applications

* 🛡️ Airspace & perimeter security
* 🚁 Drone detection and monitoring
* 🏭 Critical infrastructure protection
* 👁️ Autonomous surveillance
* 🤖 Robotic and intelligent sensing systems

---

## 📊 Real-Time Monitoring

The dashboard provides an intuitive view of:

* Current and filtered angle
* Distance and filtered distance
* Velocity and filtered velocity
* Measurement/signal quality
* Target tracking
* Future-state prediction
* Real-time graphs

---

## 📁 Project Structure

```text
ETH_Hackathon/
│
├── main.kalman.py     # Main launcher
├── aoa_kalman.py      # AoA estimation & Kalman tracking
├── dv_kalman.py       # Distance/Velocity & Kalman tracking
└── README.md
```

### Running the Project

```bash
python main.kalman.py
```

Select:

```text
1 → Angle of Arrival
2 → Distance & Velocity
```

---

## 🚀 Future Scope

The current system establishes the foundation for a complete **multi-parameter drone tracking platform**.

Future development includes:

* Unified AoA + Range + Velocity sensor fusion
* 2D/3D drone trajectory visualization
* Multi-target tracking
* Advanced filtering using EKF/UKF
* Adaptive target detection
* Predictive threat analysis
* Real-time hardware/SDR deployment

---

## 🏆 Project Vision

The ultimate goal is to transform raw radar measurements into **reliable and predictive aerial situational awareness**:

```text
DETECT → LOCALIZE → TRACK → FILTER → PREDICT
```

A step toward intelligent radar systems that don't just detect drones — **they understand and anticipate their movement.**
