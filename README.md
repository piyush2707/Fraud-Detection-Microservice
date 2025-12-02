# 🚀 Fraud Detection Microservice (End-to-End MLOps Pipeline)

## 👤 AI Engineer & Contact

| Role | Name | LinkedIn |
| :--- | :--- | :--- |
| **AI Engineer** | Piyush Joshi | [@piyush2707](https://www.linkedin.com/in/piyush2707) |

---

## ✨ Project Summary

This project demonstrates a **full-cycle MLOps pipeline** built to deploy a **real-time credit card fraud detection model**. It showcases expertise in taking a machine learning model from a training script to a scalable, containerized microservice ready for production deployment.

### Key Capabilities Demonstrated:

* **Production Readiness:** Focus on reliability, low-latency prediction, and automated deployment.
* **Engineering Focus:** Emphasis on clean code, containerization, and workflow automation.

---

## 🛠️ Architecture and Technologies

This system is built using modern, cloud-native principles.

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Model** | XGBoost Classifier | Optimized for imbalanced classification and high-speed inference. |
| **Serving API** | **FastAPI** | Creates a robust, low-latency prediction endpoint. |
| **Containerization** | **Docker** | Packages the entire environment (model, code, dependencies). |
| **Automation (CI/CD)**| **GitHub Actions** | Automates the build and push of the Docker image upon every code update. |
| **Workflow** | **n8n Principles** | Shows ability to integrate the API output (e.g., triggering alerts, routing data). |

---

## 🔗 Live Documentation & Code

You can access the full codebase here.

* **GitHub Repository:** [Fraud-Detection-Microservice](https://github.com/piyush2707/Fraud-Detection-Microservice)
* **Live API Documentation (Swagger UI):** `[Update with Live Cloud URL if deployed]`

---

## 💻 Repository Structure

| File / Folder | Purpose |
| :--- | :--- |
| `train.py` | Model training, SMOTE handling, and asset serialization (`.joblib`). |
| `api.py` | FastAPI server code for loading the model and serving the `/predict` endpoint. |
| `Dockerfile` | Defines the container environment for the FastAPI service. |
| `.github/workflows/main.yml` | The **CI/CD pipeline** configuration for automated Docker builds. |
| `model/` | Stores the trained ML model and `StandardScaler` preprocessor (crucial assets). |

---

## ✅ Local Quickstart

To run the full microservice locally, you need Docker installed:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/piyush2707/Fraud-Detection-Microservice.git](https://github.com/piyush2707/Fraud-Detection-Microservice.git)
    cd Fraud-Detection-Microservice
    ```
2.  **Build the Docker Image:**
    ```bash
    docker build -t fraud-api:latest .
    ```
3.  **Run the Container:**
    ```bash
    docker run -d -p 8000:8000 fraud-api:latest
    ```
4.  **Verify:** The API documentation is available at `http://localhost:8000/docs`.
5.  
