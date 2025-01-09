# Assignment-1: MLOps Foundations 
# Group : 120

## Overview
This project focuses on understanding the basics of MLOps by implementing a CI/CD pipeline, using popular MLOps tools, and gaining hands-on experience in model experimentation, packaging, and deployment.



## Objectives
- Build a CI/CD pipeline for a machine learning project.
- Use version control effectively .
- Track experiments and version data.
- Perform hyperparameter tuning and package models for deployment.
- (Optional) Deploy and orchestrate a model using Kubernetes.

---

## Modules and Tasks

### **M1: CI/CD Pipeline**
**Objective**: Understand the basics of CI/CD.

#### Tasks:
1. **Set Up a CI/CD Pipeline**:
   - Use GitHub Actions or GitLab CI.
   - Include stages for linting, testing, and deploying a sample ML model.

2. **Version Control**:
   - Implement Git for version control.
   - Demonstrate branching, merging, and pull requests.

#### Deliverables:
- Report detailing the CI/CD pipeline stages.
- Screenshots/logs of successful pipeline runs.
- Git repository link with branch and merge history.

---

### **M2: Process and Tooling**
**Objective**: Gain hands-on experience with MLOps tools.

#### Tasks:
1. **Experiment Tracking**:
   - Use MLflow to track metrics, parameters, and results for at least three training runs.

2. **Data Versioning**:
   - Use DVC to version control a dataset.
   - Demonstrate reverting to a previous version.

#### Deliverables:
- MLflow logs with experiment results.
- DVC repository showcasing dataset versioning.

---

### **M3: Model Experimentation and Packaging**
**Objective**: Train and tune a model, then package it for deployment.

#### Tasks:
1. **Hyperparameter Tuning**:
   - Use Optuna or GridSearchCV for tuning.
   - Document the process and optimal parameters.

2. **Model Packaging**:
   - Use Docker and Flask to package the best-performing model.
   - Provide a Dockerfile and a simple Flask application.

#### Deliverables:
- Report on tuning results.
- Dockerfile and Flask application code.
- Screenshots of the Dockerized model running.

---

### **M4: Model Deployment & Orchestration (Optional)**
**Objective**: Deploy and orchestrate a model using Kubernetes.

#### Tasks:
1. **Model Deployment**:
   - Deploy the Dockerized model on AWS, Azure, or GCP.
   - Use AWS ECS, Azure AKS, or GKE.

2. **Orchestration**:
   - Set up a Kubernetes cluster.
   - Deploy using Kubernetes and create a Helm chart.

#### Deliverables:
- Link to the deployed model endpoint.
- Kubernetes configuration files and Helm chart.
- Report on deployment and orchestration.

---

### **M5: Final Deliverables**
**Deliver a zip file containing**:
- Code, Data, and Model.
- One-page summary including:
  - Description of the work completed.
  - Justification for choices made.
- Screen recording (max 5 minutes):
  - Explains the work done.
  - Shows results.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-link>

   after running app.py use below command to test:-
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"features\": [5.1, 3.5, 1.4, 0.2]}"

   ```

2. Follow instructions in the `README` file of each module directory to replicate results.

3. For deployment:
   - Navigate to the `deployment` folder.
   - Build and run the Docker container:
     ```bash
     docker build -t ml-model .
     docker run -p 5000:5000 ml-model
     ```

4. (Optional) Deploy the model using Kubernetes:
   - Apply Kubernetes configurations:
     ```bash
     kubectl apply -f kubernetes/
     ```

---

## Author
Prepared by [Your Name].

## License
This project is licensed under the MIT License.

