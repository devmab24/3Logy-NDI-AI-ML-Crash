# Assignment 2: Object-Oriented News Dataset Builder

## Scenario

Last week, you collected news headlines using web scraping.

Your manager was impressed with the dataset but has raised a concern:

> "What if we want to collect data from multiple websites? What if we want to add more categories in the future? What if another engineer wants to reuse your code?"

The company has asked you to redesign your solution using Object-Oriented Programming (OOP).

---

# Learning Objectives

By completing this assignment, students should be able to:

* Create classes
* Use constructors (`__init__`)
* Use instance attributes
* Use methods
* Organize code using OOP principles
* Understand code reusability
* Think like software and ML engineers

---

# Task

Convert your previous news scraping project into an object-oriented application.

---

# Step 1: Create a NewsScraper Class

The class should contain:

### Attributes

```python
url
category
headlines
```

Example:

```python
scraper = NewsScraper(
    url,
    category
)
```

---

### Methods

```python
fetch_page()

parse_headlines()

save_to_csv()

display_summary()
```

---

# Example usage

```python
scraper = NewsScraper(
    "https://example.com",
    "Technology"
)

scraper.fetch_page()

scraper.parse_headlines()

scraper.display_summary()

scraper.save_to_csv()
```

---

# Step 2: Create a DatasetManager Class

Purpose:

Manage all scraped categories.

Attributes:

```python
all_data
```

Methods:

```python
add_records()

create_dataframe()

export_dataset()
```

---

Example

```python
manager = DatasetManager()

manager.add_records(data)

manager.create_dataframe()

manager.export_dataset()
```

---

# Step 3: Generate final dataset

The output should still be:

| headline                 | category   |
| ------------------------ | ---------- |
| AI transforms healthcare | Technology |
| New vaccine approved     | Health     |

---

# Bonus challenge (recommended)

Create a class called:

```python
NewsArticle
```

Attributes:

```python
headline
category
source
```

Example:

```python
article = NewsArticle(
    headline,
    category,
    source
)
```

Store NewsArticle objects inside your scraper.
---

# Grading Rubric (100 Marks)

| Criteria                 | Marks |
| ------------------------ | ----- |
| Correct Class Design     | 25    |
| Proper Use of `__init__` | 10    |
| Methods Implementation   | 20    |
| Successful Scraping      | 15    |
| DataFrame Creation       | 10    |
| CSV Export               | 10    |
| Code Organization        | 10    |


That is exactly the bridge between **Python scripting** and **software engineering for AI/ML systems**.
---
# NOTE
This assignment should introduce to you the idea of a **data pipeline**, which is exactly what ML Engineers build in industry.
