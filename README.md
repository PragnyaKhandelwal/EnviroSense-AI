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

The system follows a modular 4-layer architecture designed for high availability and analytical depth:

1. **Edge Layer:** ESP32 nodes capturing particulate matter and weather data.
2. **Persistence Layer:** **TimescaleDB** (PostgreSQL) optimized for high-velocity time-series storage.
3. **Inference Layer:** **FastAPI** serving pre-trained Deep Learning models for real-time predictions.
4. **Presentation Layer:** **React PWA** utilizing WebSockets for instant data updates without refresh.

---

## 📊 Data Analytics & AI/ML Methodology

This project serves as a showcase for two distinct academic disciplines:

### Data Analytics Focus

* **Time-Series Decomposition:** Statistical separation of raw data into Trend, Seasonality, and Residuals to validate environmental cycles.
* **Multivariate Correlation:** Analyzing the impact of Temperature and Humidity on Particulate Matter concentration.

### AI/ML Focus

* **Sequence Modeling:** Implementing an LSTM architecture to capture temporal dependencies in air quality patterns.
* **Anomaly Detection:** Using a Reconstruction Error threshold from a trained Autoencoder to flag hardware failures or localized smoke events.

---

## 🛠️ Tech Stack

* **Hardware:** ESP32, NextPM Optical Sensor, DHT22.
* **Backend:** Python, FastAPI, SQLAlchemy, Paho-MQTT.
* **Frontend:** React.js, Vite, Tailwind CSS, Recharts.
* **Database:** PostgreSQL with TimescaleDB extension.
* **ML/DS Libraries:** TensorFlow/Keras, Scikit-learn, Pandas, Numpy.

---

## 📦 Getting Started

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

```


2. **Backend Setup**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

```


3. **Frontend Setup**
```bash
cd frontend
npm install
npm run dev

```



---

## 📈 Future Roadmap

* [ ] Integration of OpenWeatherMap API for outdoor baseline comparison.
* [ ] Support for multi-node sensor mesh networking.
* [ ] Deployment of Edge-AI (TensorFlow Lite) directly on ESP32.

