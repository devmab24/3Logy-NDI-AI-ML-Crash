# Assignment: News Headline Dataset Collection and Preparation

## Scenario

You have just been hired as a Machine Learning Engineer by a media intelligence company. The company wants to build a Machine Learning model that can automatically classify news headlines into categories such as:

* Technology
* Health
* Business
* Sports
* Politics
* Entertainment

Before a model can be trained, the company needs a dataset. Your responsibility is to collect the data, organize it, and prepare it for Machine Learning.

---

# Learning Objectives

By completing this assignment, students should be able to:

- ✅ Use APIs to collect data
- ✅ Perform basic web scraping
- ✅ Extract useful information
- ✅ Create Pandas DataFrames
- ✅ Export data to CSV files
- ✅ Understand the Data Collection stage of the ML lifecycle

---

# Task 1: Find a News Source

Students may use:

```python
requests
BeautifulSoup
```

Sources:

* BBC
* CNN
* Reuters
* TechCrunch
* Al Jazeera

(Only scrape publicly available pages and respect site terms.)

---

# Expected Dataset

The final dataset should look like:

| headline                   | category   |
| -------------------------- | ---------- |
| Apple launches new AI chip | Technology |
| WHO releases health report | Health     |
| Stock market rises today   | Business   |

---

# Minimum Requirements

Each student must collect:

### At least

```text
200 headlines
```

from

```text
4 categories
```

Minimum:

```text
50 headlines per category
```

---

Example:

| Category   | Records |
| ---------- | ------- |
| Technology | 50      |
| Health     | 50      |
| Business   | 50      |
| Sports     | 50      |

Total:

```text
200 records
```

---

# Deliverables

Students must submit:

## 1. Python Script

Example:

```text
news_collection.py
```

---

## 2. CSV Dataset

Example:

```text
news_dataset.csv
```

---

## 3. Short Report

One page explaining:

### Data Source

Where did the data come from?

---

### Collection Method

API or Web Scraping?

---

### Challenges

What difficulties were encountered?

---

### Dataset Size

How many records were collected?

---

# Suggested Folder Structure

```text
news_classification_project/

│
├── data/
│   └── news_dataset.csv
│
├── scripts/
│   └── collect_news.py
│
├── report/
│   └── report.md
│
└── README.md
```

---

# Bonus Challenge

Add the following columns:

| headline | category | source | date |
| -------- | -------- | ------ | ---- |

Example:

| headline                 | category   | source | date       |
| ------------------------ | ---------- | ------ | ---------- |
| AI transforms healthcare | Technology | BBC    | 2025-07-08 |

---

# Evaluation Rubric (100 Marks)

| Criteria                         | Marks |
| -------------------------------- | ----- |
| Data Collection                  | 20    |
| Code Quality                     | 20    |
| DataFrame Creation               | 20    |
| CSV Export                       | 15    |
| Documentation                    | 15    |
| Creativity / Additional Features | 10    |

Total:

```text
100 Marks
```
# Submission
On or before Thursday 9th July, 2025
