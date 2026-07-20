import requests
import pandas as pd
import os
from datetime import date

NEWS_API_KEY = "b1885c8d5a69490eb6c0e8610f3fb609"

CATEGORIES = ["technology", "health"]

records = []

for category in CATEGORIES:
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "category": category,
        "language": "en",
        "pageSize": 100,  # max allowed per request
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)
    print(f"{category}: status {response.status_code}")

    if response.status_code != 200:
        print(f"  Error: {response.json().get('message')}")
        continue

    data = response.json()
    articles = data.get("articles", [])
    print(f"  Retrieved {len(articles)} articles")

    for article in articles:
        headline = article.get("title")
        source = article.get("source", {}).get("name")
        published = article.get("publishedAt", "")[:10]  # just the date part

        if headline:
            records.append({
                "headline": headline,
                "category": category.capitalize(),
                "source": source,
                "date": published if published else date.today().isoformat()
            })

# Build DataFrame
df = pd.DataFrame(records)
df.drop_duplicates(subset="headline", inplace=True)

print("\nHeadlines per category:")
print(df["category"].value_counts())

# # Save to your project folder
# output_dir = "news_classification_project/data"
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir, "tech_health_headlines.csv")
# df.to_csv(output_path, index=False)
# print(f"\nSaved to: {os.path.abspath(output_path)}")