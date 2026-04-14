# 🌍 EnviroSense AI

**A Hybrid Deep Learning PWA for Proactive Air Quality Forecasting & Anomaly Detection**

EnviroSense AI is an end-to-end environmental intelligence ecosystem. By bridging the gap between raw IoT sensor data and advanced Deep Learning, the system transforms air quality monitoring from a reactive "status check" into a proactive "early warning system."

## 🚀 Key Features

* **Real-Time IoT Ingestion:** High-fidelity data streaming from **NextPM sensors** via ESP32 and MQTT over WebSockets.
* **Predictive Intelligence:** **LSTM (Long Short-Term Memory)** neural networks that forecast  and  levels for a 6-hour horizon.
* **Automated Fault Detection:** **Unsupervised Autoencoder** models that identify sensor drift and hardware anomalies in real-time.
* **Installable PWA:** A mobile-native dashboard built with **React** and **Vite**, featuring offline caching and "Add to Home Screen" functionality.
* **Proactive Alerting:** Integrated notification system (Telegram/Email) triggered by **AI-predicted** hazardous conditions.

---

## 🏗️ System Architecture

The platform follows a modular 4-layer architecture designed for high availability and analytical depth:

* **Edge Layer:** ESP32 nodes capturing high-fidelity particulate matter (PM1.0, PM2.5, PM10) and particle bin counts (0.3μm–10μm) via UART/Modbus protocols.
* **Persistence Layer:** TimescaleDB (PostgreSQL) optimized for high-velocity time-series storage and hypertable partitioning.
* **Inference Layer:** FastAPI serving pre-trained Deep Learning models (LSTM/Autoencoders) for real-time forecasting and anomaly detection.
* **Presentation Layer:** React PWA utilizing WebSockets for instant data updates and cross-platform accessibility.

---

## 📊 Data Analytics & AI/ML Methodology

This project implements rigorous data science workflows to ensure accuracy and reliability:

### Data Analytics Focus

* **Time-Series Decomposition:** Statistical separation of raw data into Trend, Seasonality, and Residuals to validate environmental cycles.
<<<<<<< HEAD
* **Size-Distribution Profiling:** Analyzing shifts in particle bin counts (0.3μm to 10μm) to identify pollutant sources (e.g., smoke vs. dust).
* **Environmental Correlation:** Quantifying the impact of Temperature and Humidity on Particulate Matter concentrations through multivariate regression.

### AI/ML Focus

* **Sequence Modeling (LSTM):** Implementing Long Short-Term Memory neural networks to capture temporal dependencies and predict PM levels for a 6-hour horizon.
* **Unsupervised Anomaly Detection:** Using Reconstruction Error thresholds from trained Autoencoders to flag sensor hardware failures or localized extreme events.
=======
* **Multivariate Correlation:** Analyzing the impact of Temperature and Humidity on Particulate Matter concentration.

### AI/ML Focus

* **Sequence Modeling:** Implementing an LSTM architecture to capture temporal dependencies in air quality patterns.
* **Anomaly Detection:** Using a Reconstruction Error threshold from a trained Autoencoder to flag hardware failures or localized smoke events.
>>>>>>> master

---

## 🛠️ Tech Stack

* **Hardware:** ESP32, Next-PM Optical Sensor, DHT22.
* **Backend:** Python 3.10+, FastAPI, SQLAlchemy, Paho-MQTT.
* **Frontend:** React.js, Vite, Tailwind CSS, Recharts.
* **Database:** PostgreSQL with TimescaleDB extension.
* **ML/DS Libraries:** TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Plotly.

---

## 🖥️ Remote Development & VPS Workflow

To maintain a production-ready environment, development is conducted via **Remote-SSH**, ensuring that high-compute tasks remain on the server while code is authored in a local IDE.

### 1. Remote Connection

Standardize connections via a non-root user to maintain security and system integrity.

**SSH Configuration (`~/.ssh/config`):**

```yaml
Host envirosense-vps
    HostName <YOUR_VPS_IP>
    User <YOUR_USERNAME>
    IdentityFile ~/.ssh/id_ed25519

```

### 2. Isolated Execution

All analytics and subscriber logic should run within a dedicated Python Virtual Environment to prevent dependency drift.

```bash
# Activation and Verification
source .venv/bin/activate
python -c "import sys; print(sys.prefix)" # Confirms isolated environment

```

### 3. Port Forwarding for Visualization

When running analytical dashboards (e.g., Plotly Dash or Streamlit) on the VPS, use VS Code Port Forwarding to tunnel the web interface to your local browser:

* **Remote Port:** `8050` (or your app's port)
* **Local Address:** `localhost:8050`

---

## 📦 Getting Started

<<<<<<< HEAD
1. **Clone the Repository:**
```bash
git clone https://github.com/YourUsername/EnviroSense-AI.git
cd EnviroSense-AI
=======
### Prerequisites

* Python 3.10+
* Node.js & npm
* Mosquitto MQTT Broker (configured on VPS)
* PostgreSQL with TimescaleDB

### Installation

1. **Clone the Repo**
```bash
git clone https://github.com/yourusername/envirosense-ai.git
cd envirosense-ai
>>>>>>> master

```


2. **Environment Setup:**
Install dependencies from `requirements.txt` into your virtual environment.
3. **Database Configuration:**
Ensure the PostgreSQL `pg_hba.conf` allows connections for your specific database user.
4. **Service Deployment:**
Configure the `subscriber.py` as a `systemd` service for 24/7 data persistence.
