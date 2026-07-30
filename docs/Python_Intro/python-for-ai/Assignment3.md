## Assignment 3: Data Quality Investigation & Preprocessing

## Smart Incident Report Analyzer (SIRA)

### Background

You have joined the AI Engineering team at **SIRA Technologies**. The team has received a dataset containing incident reports collected from various Oil & Gas facilities. Before any Machine Learning model can be developed, the quality of the dataset must be assessed and improved. Your responsibility is to inspect the dataset, identify all quality issues, clean the data, and produce a final dataset that is suitable for machine learning. No additional guidance will be provided regarding the types of issues present in the dataset. It is your responsibility as a Machine Learning Engineer to investigate and justify every modification you make.

---

### Objective

Using the `Sira` project structure developed during the class, build a reusable preprocessing pipeline capable of cleaning the dataset.

Your solution should demonstrate proper use of:

* Object-Oriented Programming
* Python Modules
* Functions
* Data Analysis
* Problem Solving
* Software Engineering best practices

---

### Requirements

Using your existing project:

```text
smart_incident_report_analyzer/

│
├── data/
│
├── notebooks/
│
├── src/
│   ├── __init__.py
│   ├── incident.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── utils.py
│
├── clean_data.py     # New module to be created
│
├── README.md
└── ...*
```

Perform a complete data quality assessment on the provided dataset. URL here: https://github.com/devmab24/3Logy-NDI-AI-ML-Crash/blob/main/datasets/incident_reports_1000.csv

Your preprocessing pipeline should clean the dataset and produce a new file named:

```text
incident_reports_clean.csv
```

NOTE: The file should be saved in the appropriate `dir`.

---

### Deliverables

Create your own github repository commit and submit the following:

### 1. Updated project structure/folder

Including all source code.

---

### 2. Updated `preprocessing.py`

Implement any methods you consider necessary to clean the dataset. If none you may leave the `preprocessing.py` as it is.

---

### 3. `clean_data.py`

A script that executes the complete preprocessing pipeline from start to finish.

Running

```bash
python clean_data.py
```

should generate the cleaned dataset automatically.

---

### 4. Clean Dataset

Save the final cleaned dataset as

```text
incident_reports_clean.csv
```

inside the `data/` directory.

---

### 5. README.md

Your README should briefly answer the following:

* About the `Sira` Application
* What data quality issues did you discover?
* How did you identify them?
* What steps did you take to clean the data?
* Why did you choose those approaches?

---

### Expected Outcome

When your program finishes executing, it should successfully produce a cleaned dataset that is ready for machine learning which we'll use in phase 3-4.

---

### Rules

* Do not manually edit the CSV file.
* Your solution must be fully automated.
* Use Python only.
* Follow the project structure introduced during class.
* Write clean, modular, and reusable code.
* Ensure your program can be executed multiple times without errors.

---

### Submission

Submit:

* GitHub repository link here: https://github.com/devmab24/3Logy-NDI-AI-ML-Crash/blob/main/submissions/Sira_URL_Submissions.md
* Deadline: On or Before 1:00PM Monday, 3rd August 2026
---

### Evaluation Rubric (100 Marks)

| Criteria                             | Marks |
| ------------------------------------ | ----: |
| Project structure and organization   |    10 |
| Investigation of data quality issues |    20 |
| Design of preprocessing pipeline     |    25 |
| Code quality and modularity          |    20 |
| Correctness of cleaned dataset       |    15 |
| Documentation (README)               |    10 |

---

### Instructor's Note

> In Machine Learning Engineering, one of the most valuable skills is not writing models, it is understanding data. This assignment is designed to evaluate your ability to investigate, reason about, and improve the quality of real-world data. There is no single correct solution. Your grade will reflect the quality of your analysis, the robustness of your implementation, and the engineering practices you apply. Wish you the best.
