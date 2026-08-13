# Assignment: Migrate the SIRA Machine Learning Application to AWS

## Smart Incident Report Analyzer (SIRA)

### Assignment Level
Intermediate — AWS + Machine Learning Engineering

### Project Context

You have been developing the **Smart Incident Report Analyzer (SIRA)** locally using Python, Machine Learning, NLP, and Object-Oriented Programming.

You have now been introduced to the AWS environment and the following services:

- AWS IAM
- Amazon S3
- Amazon EBS
- Amazon EC2
- Amazon SageMaker AI
- SageMaker JupyterLab

Your next task is to take what you have already built locally and **recreate the development environment in AWS**.

The objective is not simply to create an AWS account or launch a notebook.

The objective is to understand how a real ML Engineering project can move from a local development environment into the cloud.

---

# 1. Objective

By completing this assignment, you should be able to:

1. Launch and access a cloud-based development environment.
2. Work with SageMaker JupyterLab.
3. Recreate the SIRA project structure in the cloud.
4. Upload and retrieve datasets using Amazon S3.
5. Install and manage Python dependencies.
6. Execute your existing Python modules in the cloud.
7. Run your preprocessing and feature-engineering pipeline.
8. Understand the relationship between local storage, EBS, and S3.
9. Apply basic AWS security practices using IAM.
10. Document your cloud architecture and development process.

---

# 2. Starting Point

You are NOT starting the project from scratch.

Your existing SIRA application should already contain the work completed during the previous phases.

Your local project should be similar to:

```text
smart_incident_report_analyzer/

│
├── data/
│   ├── incidents_report.csv
│   ├── incident_reports_1000.csv
│   └── incident_reports_clean.csv
│
├── notebooks/
│   ├── 01_python_to_ml_engineering.ipynb
│   └── 02_data_loading.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── incident.py
│   ├── preprocessing.py
│   └── feature_engineering.py
│
├── models/
│
├── main.py
├── clean_data.py
├── requirements.txt
├── README.md
└── .gitignore
```

Your task is to reproduce an appropriate version of this environment in AWS.

---

# 3. Part A — AWS Environment

## Task 1: Access SageMaker JupyterLab

Using the AWS environment provided during training:

1. Access Amazon SageMaker AI.
2. Launch the JupyterLab development environment.
3. Confirm that you can access the JupyterLab interface.
4. Identify where your project files will be stored.
5. Identify the compute environment being used.

Take screenshots showing:

- SageMaker environment
- JupyterLab environment
- Running development environment

Do NOT expose passwords, access keys, secret keys, or other credentials in your screenshots.

---

# 4. Part B — Create the SIRA Project Structure

Inside your SageMaker JupyterLab environment, recreate the project.

Your structure should be similar to:

```text
smart_incident_report_analyzer/

│
├── data/
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── incident.py
│   ├── preprocessing.py
│   └── feature_engineering.py
│
├── models/
│
├── main.py
├── clean_data.py
├── requirements.txt
└── README.md
```

You are expected to understand the purpose of each directory and module.

You should NOT simply create empty folders without understanding what belongs inside them.

---

# 5. Part C — Move Your Dataset to Amazon S3

Create an appropriate S3 location for the SIRA project.

Organize your data logically.

For example:

```text
SIRA S3 Bucket
│
├── raw/
│   └── incident_reports_1000.csv
│
├── processed/
│   └── incident_reports_clean.csv
│
└── models/
```

Upload the appropriate dataset to S3.

You should be able to explain:

### Why are we using S3?

### Why shouldn't the SIRA application depend entirely on a local CSV file?

### What is the difference between storing a dataset locally and storing it in S3?

### Why should raw and processed datasets be separated?

---

# 6. Part D — Connect SIRA to S3

Your application should be capable of retrieving the dataset from S3.

You may use the AWS SDK for Python (**Boto3**) or another appropriate method introduced during training.

Your workflow should become:

```text
Amazon S3
    │
    ▼
Dataset
    │
    ▼
SIRA Application
    │
    ▼
DataLoader
    │
    ▼
Preprocessing
```

The application should NOT require you to manually download the CSV every time you want to process the dataset.

---

# 7. Part E — Install Project Dependencies

Your project already uses Python libraries.

Review your:

```text
requirements.txt
```

Install the dependencies required by SIRA inside your cloud environment.

At minimum, identify the libraries required for:

- Data manipulation
- Machine Learning
- NLP/vectorization
- AWS interaction
- Notebook development

Verify that your application can import its required packages successfully.

---

# 8. Part F — Run Your Existing Preprocessing Pipeline

Your existing preprocessing module should now operate inside the AWS environment.

Run your preprocessing pipeline against the dataset stored in S3.

The objective is to reproduce the process you previously performed locally:

```text
Raw Dataset
     ↓
Data Quality Analysis
     ↓
Preprocessing
     ↓
Clean Dataset
     ↓
Feature Engineering
```

Your final cleaned dataset should be saved appropriately.

Consider whether the processed dataset should also be stored in S3.

---

# 9. Part G — Run Feature Engineering

Use your existing:

```text
feature_engineering.py
```

module.

Your application should be able to perform the vectorization process in the cloud.

For example:

```text
Clean Incident Reports
          ↓
FeatureEngineer
          ↓
TF-IDF / Bag of Words
          ↓
Numerical Feature Matrix
```

You should verify that the feature engineering process works successfully in SageMaker JupyterLab.

---

# 10. Part H — Run the Machine Learning Pipeline

If your ML model has already been developed, execute the existing training/evaluation workflow in the AWS environment.

Your workflow should resemble:

```text
S3 Dataset
    ↓
SageMaker JupyterLab
    ↓
Data Loading
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
Train/Test Split
    ↓
Model Training
    ↓
Evaluation
    ↓
Model Artifact
```

Record the model performance.

Compare the result with the model performance you obtained locally.

---

# 11. Part I — Investigate the AWS Storage Architecture

You have learned about:

- EBS
- S3

Now demonstrate that you understand the difference.

In your README, explain:

### Amazon EBS

What problem does EBS solve?

Where is EBS attached?

What happens to files stored on the development environment's attached storage?

### Amazon S3

What problem does S3 solve?

Why is S3 suitable for datasets and model artifacts?

Why might S3 be preferred as a central storage layer for an ML project?

Do not simply define the services. Relate your explanation to SIRA.

---

# 12. Part J — IAM and Security

Review the IAM configuration used for your development environment.

Answer:

1. What identity is your SageMaker environment using?
2. What permissions does it require?
3. Why shouldn't you use the AWS root account for normal development?
4. What is the Principle of Least Privilege?
5. Why should AWS credentials never be placed inside Python files?
6. Why should credentials never be committed to GitHub?

Your explanation should demonstrate understanding rather than copied definitions.

---

# 13. Part K — Cloud Architecture Diagram

Create an architecture diagram for your SIRA application.

Your diagram should show at least:

```text
                 AWS
                  │
          ┌───────┴────────┐
          │                │
         S3            SageMaker
          │            JupyterLab
          │                │
          ▼                ▼
      Dataset ───────► SIRA
                           │
                           ▼
                    Preprocessing
                           │
                           ▼
                  Feature Engineering
                           │
                           ▼
                     ML Training
                           │
                           ▼
                       Model
```

Your architecture does not have to look exactly like this.

You are expected to design an architecture that makes sense for your implementation.

---

# 14. Part L — Compare Local vs Cloud Development

Create a section in your README called:

```text
Local vs AWS Development
```

Compare:

| Area | Local Environment | AWS Environment |
|---|---|---|
| Compute | | |
| Storage | | |
| Dataset access | | |
| Scalability | | |
| Collaboration | | |
| Security | | |
| Cost | | |
| ML development | | |

Explain your answers specifically in the context of SIRA.

---

# 15. Part M — Cost Awareness

Before creating AWS resources, consider:

- What resources are you using?
- Are they running continuously?
- Which resources could generate charges?
- When should they be stopped?
- When should temporary resources be deleted?

Document the resources you created and explain how you would prevent unnecessary costs.

> Never leave chargeable resources running unnecessarily.

---

# 16. Challenge Task ⭐

For students who want an additional challenge:

Modify your SIRA application so that the following workflow works:

```text
                    S3
                     │
                     ▼
              Raw Dataset
                     │
                     ▼
              DataLoader
                     │
                     ▼
             Preprocessing
                     │
                     ▼
             FeatureEngineer
                     │
                     ▼
              ML Pipeline
                     │
                     ▼
                Prediction
```

The goal is to reduce manual intervention.

The application should be able to retrieve the required dataset from S3 and execute the processing pipeline from the cloud environment.

---

# 17. Important Restrictions

You MUST:

- Use your own SIRA project.
- Use the AWS environment introduced during training.
- Use your existing Python modules where appropriate.
- Keep the project organized.
- Follow good software engineering practices.
- Document your work.
- Protect AWS credentials.

You MUST NOT:

- Hard-code AWS credentials.
- Upload secret keys to GitHub.
- Make S3 buckets publicly accessible unnecessarily.
- Use the root account for routine development.
- Delete or modify another student's resources.
- Leave unnecessary chargeable resources running.

---

# 18. Required Deliverables

Submit the following:

### 1. GitHub Repository

Your repository should contain the updated SIRA project.

### 2. AWS Project

Your SIRA project should be successfully running in SageMaker JupyterLab.

### 3. S3 Storage

Your dataset should be appropriately stored in S3.

### 4. Working Python Pipeline

Your application should demonstrate:

```text
S3
 ↓
DataLoader
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
ML Pipeline
```

### 5. Architecture Diagram

Include your AWS/SIRA architecture diagram.

### 6. Updated README

The README should document:

- Project overview
- AWS services used
- AWS architecture
- Project structure
- Dataset location
- How the application is executed
- Local vs AWS comparison
- IAM/security considerations
- Cost considerations
- Challenges encountered
- How you solved those challenges

### 7. Screenshots

Include appropriate screenshots demonstrating:

- SageMaker JupyterLab
- Project structure
- S3 dataset
- Successful execution
- Model/ML results

Do not include credentials or sensitive information.

---

# 19. Reflection Questions

Answer these questions in your README:

### Question 1

Why did we move SIRA from a local environment to AWS?

### Question 2

What role does S3 play in the SIRA architecture?

### Question 3

What role does SageMaker JupyterLab play?

### Question 4

What is the difference between S3 and EBS?

### Question 5

Why do we need IAM?

### Question 6

What would happen if you deleted your local dataset but the dataset was still available in S3?

### Question 7

What challenges did you encounter during migration?

### Question 8

If SIRA had 1 million incident reports instead of 1,000, what aspects of the architecture would need to change?

### Question 9

If multiple developers were working on SIRA, what AWS services or practices could help with collaboration and security?

### Question 10

What part of the SIRA application would you move to a managed ML service such as SageMaker, and why?

---

# 20. Success Criteria

You have successfully completed the assignment when:

- [ ] You can access SageMaker JupyterLab.
- [ ] You recreated the SIRA project structure.
- [ ] Your project dependencies are installed.
- [ ] Your dataset is stored in S3.
- [ ] Your application can access the dataset.
- [ ] Your preprocessing pipeline runs successfully.
- [ ] Your feature engineering pipeline runs successfully.
- [ ] Your ML workflow executes successfully.
- [ ] You understand the role of IAM.
- [ ] You understand the difference between S3 and EBS.
- [ ] You documented your architecture.
- [ ] You documented your challenges and solutions.
- [ ] Your GitHub repository is updated.
- [ ] No AWS credentials are exposed.

---

# Final Engineering Challenge

Do not approach this assignment as:

> "I need to upload my project to AWS."

Approach it as:

> **"I am an ML Engineer taking an existing machine learning system from local development into a cloud environment."**

Your objective is not merely to make SIRA run on AWS.

Your objective is to understand **why each AWS service is being used, how the services interact, and how the architecture could evolve into a production ML system.**

Good engineering requires more than making the code run.

It requires understanding the **data, infrastructure, security, scalability, cost, and deployment decisions** behind the system.
