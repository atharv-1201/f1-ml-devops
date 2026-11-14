# 🏎️ End-to-End DevOps Pipeline for ML Model Inference on AWS EC2

[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-Jenkins%20%7C%20GitHub%20Actions-blue)](Jenkinsfile)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2%20Ubuntu-FF9900?logo=amazon-aws&logoColor=white)](#aws-infrastructure)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20Inference-009688?logo=fastapi&logoColor=white)](app/main.py)
[![Pytest](https://img.shields.io/badge/Testing-Pytest%20Passing-success)](tests/)

An end-to-end DevOps project that automates the testing, containerization, and deployment of a Machine Learning race pace prediction model onto an **AWS EC2** instance using **Docker** and a declarative **CI/CD Pipeline (Jenkins & GitHub Actions)**.

---

## 🏗️ Architecture Overview

```text
 ┌───────────────┐
 │   Developer   │
 └───────┬───────┘
         │  git push
         ▼
 ┌───────────────┐
 │ GitHub Repo   │
 └───────┬───────┘
         │  Webhook / Event Trigger
         ▼
 ┌────────────────────────────────────────────────────────┐
 │           CI/CD Pipeline (Jenkins / Actions)           │
 │                                                        │
 │  1. Checkout ──► 2. Pytest Unit Tests ──► 3. Docker    │
 │     Code            & Quality Gate           Build     │
 └──────────────────────────┬─────────────────────────────┘
                            │  SSH Deploy
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │                   AWS EC2 Instance                     │
 │                 (Public IP: 18.61.27.74)               │
 │                                                        │
 │   ┌────────────────────────────────────────────────┐   │
 │   │           Docker Container (:8000)             │   │
 │   │  • FastAPI REST API   • GBR ML Engine          │   │
 │   │  • Interactive UI     • Healthcheck Monitor    │   │
 │   └────────────────────────────────────────────────┘   │
 └──────────────────────────▲─────────────────────────────┘
                            │  HTTP Requests
 ┌──────────────────────────┴─────────────────────────────┐
 │       Evaluators / Users (Browser / Postman / cURL)     │
 └────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Tool / Technology | Purpose |
| :--- | :--- | :--- |
| **Source Control** | **Git & GitHub** | Source code management, version control, PR reviews |
| **CI/CD Engine** | **Jenkins & GitHub Actions** | Automated testing, build triggers, and remote deployment |
| **Containerization** | **Docker** | Consistent, isolated runtime for the ML inference service |
| **Cloud Hosting** | **AWS EC2 (Ubuntu 24.04 LTS)** | Cloud virtual server hosting the live Dockerized API |
| **ML Engine** | **Scikit-Learn (Gradient Boosting)** | Race pace and podium prediction based on F1 data |
| **Web Service** | **FastAPI & Uvicorn** | High-performance asynchronous REST API & Web UI |
| **Testing** | **Pytest & TestClient** | Automated unit tests and API quality gate |

---

## 📂 Repository Structure

```text
devops/
├── .github/
│   └── workflows/
│       └── deploy.yml        # Automated CI/CD workflow (GitHub Actions)
├── app/
│   ├── data/
│   │   └── model.joblib      # Serialized ML model artifact
│   ├── static/
│   │   └── index.html        # Interactive F1 race dashboard UI
│   ├── __init__.py
│   ├── main.py               # FastAPI application & route handlers
│   ├── model.py              # ML inference engine & feature preprocessors
│   ├── schemas.py            # Pydantic input/output validation schemas
│   └── train.py              # Model training script for CI pipeline
├── tests/
│   ├── __init__.py
│   └── test_api.py           # Automated unit test suite
├── 2025_f1_predictions/      # Historical F1 analysis scripts
├── Dockerfile                # Production-grade multi-layer Dockerfile
├── .dockerignore             # Optimized build context filter
├── .gitignore                # Git exclusions
├── deploy.sh                 # Self-contained deployment script
├── Jenkinsfile               # Declarative 4-stage Jenkins pipeline
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚦 CI/CD Pipeline Stages

Our pipeline enforces rigorous DevOps best practices across 4 automated stages:

1. **Stage 1: Code Checkout**
   * Pulls the latest commits from the `main` branch.
2. **Stage 2: Automated Testing & Quality Gate**
   * Installs dependencies and runs `pytest tests/ -v`.
   * Verifies `/health`, `/model-info`, single-driver prediction, and grid simulation.
   * If any test fails, the build is immediately aborted, preventing broken code from reaching EC2.
3. **Stage 3: Docker Container Build**
   * Builds the Docker image `f1-predictor:latest` using layer caching.
   * Enforces non-root container security permissions (`appuser`).
4. **Stage 4: Automated Deployment to AWS EC2**
   * Securely SSHs into the target EC2 instance (`18.61.27.74`).
   * Fetches latest code, restarts the container on port `8000`, and verifies the `/health` endpoint.

---

## 🌐 Live API Endpoints

Once deployed on AWS EC2, the service exposes the following endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Interactive Web Dashboard (Sliders for weather, race simulation) |
| `GET` | `/health` | Healthcheck endpoint (`{"status": "healthy"}`) |
| `GET` | `/model-info` | Returns model algorithm, training samples, and features |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `POST` | `/predict` | Single driver lap-time inference |
| `POST` | `/predict/grid` | Simulates full 2025 Monaco GP grid and calculates podium |

### Sample cURL Request: Single Driver Inference

```bash
curl -X POST "http://18.61.27.74:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "driver": "VER",
       "qualifying_time": 70.669,
       "rain_probability": 0.0,
       "temperature": 22.0
     }'
```

### Response:
```json
{
  "driver": "VER",
  "team": "Red Bull",
  "qualifying_time": 70.669,
  "rain_probability": 0.0,
  "temperature": 22.0,
  "predicted_lap_time": 79.347,
  "unit": "seconds"
}
```

---

## 💻 Local Quickstart

To run the project locally on your machine:

```bash
# 1. Clone repository
git clone <YOUR_REPO_URL>
cd devops

# 2. Run unit tests
pip install -r requirements.txt
pytest tests/ -v

# 3. Build & Run Docker container
docker build -t f1-predictor:latest .
docker run -d -p 8000:8000 --name f1-test f1-predictor:latest

# 4. Access Dashboard
open http://localhost:8000
```
