# 🚁 Intelligent ISAC-Based Drone Detection & Tracking

> **From wireless signals to real-time aerial intelligence — Detect • Localize • Track • Predict**

## 🌐 Overview

An **Integrated Sensing and Communication (ISAC)** based system that transforms modern wireless signals into a real-time **drone detection and tracking platform**.

Inspired by emerging **5G wireless sensing technologies**, the system combines **OFDM-based sensing, Doppler processing, signal analysis, and intelligent state estimation** to determine:

**📐 Angle • 📏 Distance • ⚡ Velocity**

The processed measurements are continuously analyzed to track target motion, reduce measurement fluctuations, and provide short-term trajectory prediction through an interactive real-time dashboard.

---

## ⚙️ How It Works

```text
              📡 Wireless / SDR Signals
                       │
              ┌────────┴────────┐
              ▼                 ▼
        OFDM-Based Sensing   Doppler Sensing
              │                 │
              ▼                 ▼
        Angle Estimation    Range & Velocity
              │                 │
              └────────┬────────┘
                       ▼
              Intelligent Tracking
                       │
                       ▼
              Motion Prediction
                       │
                       ▼
              📊 Real-Time Dashboard
```

---

## 🔥 Key Features

* 📡 5G-inspired ISAC sensing
* 📐 Real-time **Angle of Arrival (AoA)** estimation
* 📏 **Distance / range** estimation
* ⚡ **Velocity / Doppler** estimation
* 🧠 Intelligent measurement stabilization
* 🎯 Continuous target tracking
* 🔮 Short-term trajectory prediction
* 📊 Interactive real-time visualization
* 💻 SDR-compatible sensing architecture

---

## 🧩 Technology Stack

**Wireless Sensing** → OFDM • ISAC • 5G Concepts
**Signal Processing** → Doppler • Phase • Frequency Analysis
**Tracking** → State Estimation • Motion Modeling • Prediction
**Hardware** → SDR / Radar Sensing
**Visualization** → Python • Real-Time Dashboard

---

## 🎯 Why ISAC?

Traditional wireless communication primarily focuses on connecting devices.

**ISAC adds another dimension — sensing the environment using wireless signals.**

This enables a single wireless infrastructure to potentially support both:

```text
📶 COMMUNICATION  +  📡 SENSING
         ↓
     🧠 INTELLIGENCE
```

Making ISAC a promising technology for **next-generation 5G networks, autonomous systems, smart infrastructure, and aerial monitoring**.

---

## 📊 Real-Time Intelligence

The dashboard provides a live view of:

**Angle | Distance | Velocity | Target Status | Tracking | Prediction | Signal Information**

allowing raw sensing data to be converted into an intuitive picture of target behavior and movement.

---

## 📁 Project Structure

```text
ETH_Hackathon/
│
├── main.kalman.py      # Main launcher
├── aoa_kalman.py       # Angle sensing & tracking
├── dv_kalman.py        # Distance/velocity sensing & tracking
└── README.md
```

### ▶️ Run

```bash
python main.kalman.py
```

```text
1 → Angle of Arrival
2 → Distance & Velocity
```

---

## 🚀 Future Scope

* 🔗 Multi-parameter sensor fusion
* 🌐 2D/3D drone trajectory visualization
* 🎯 Multi-target tracking
* 🧠 Advanced adaptive tracking
* 📡 Real-time SDR deployment
* 🔮 Predictive aerial threat analysis
* 📶 Integration with future 5G/6G ISAC networks

---

## 🏆 Vision

> **Turn wireless signals into actionable aerial intelligence.**

```text
DETECT → LOCALIZE → TRACK → ANALYZE → PREDICT
```

**Building toward intelligent, real-time and next-generation wireless sensing systems.**
tems that don't just detect drones — **they understand and anticipate their movement.**
