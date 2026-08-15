import os
from datetime import date

import pandas as pd
import requests


API_KEY = "b1885c8d5a69490eb6c0e8610f3fb609"

NEWS_GROUPS = ["technology", "health"]
API_ENDPOINT = "https://newsapi.org/v2/top-headlines"


def retrieve_articles(news_group):
    """Fetch headlines for a specified news category."""

    request_data = {
        "category": news_group,
        "language": "en",
        "pageSize": 100,
        "apiKey": API_KEY
    }

    api_result = requests.get(API_ENDPOINT, params=request_data)

    print(f"{news_group}: HTTP Status = {api_result.status_code}")

    if api_result.status_code != 200:
        print(f"Request Failed: {api_result.json().get('message')}")
        return []

    json_result = api_result.json()
    article_collection = json_result.get("articles", [])

    print(f"Downloaded {len(article_collection)} articles")

    processed_articles = []

    for news in article_collection:
        article_name = news.get("title")
        publisher_name = news.get("source", {}).get("name")
        publish_day = news.get("publishedAt", "")[:10]

        if article_name:
            processed_articles.append({
                "headline": article_name,
                "category": news_group.title(),
                "source": publisher_name,
                "date": publish_day or date.today().isoformat()
            })

    return processed_articles


def create_dataset():
    """Combine all categories into one dataset."""

    news_dataset = []

    for group in NEWS_GROUPS:
        news_dataset.extend(retrieve_articles(group))

    headlines_df = pd.DataFrame(news_dataset)
    headlines_df.drop_duplicates(subset="headline", inplace=True)

    return headlines_df


def main():
    final_dataframe = create_dataset()

    print("\nHeadline Distribution:")
    print(final_dataframe["category"].value_counts())

    # Uncomment to save the dataset
    # save_directory = "news_classification_project/data"
    # os.makedirs(save_directory, exist_ok=True)
    #
    # csv_file = os.path.join(save_directory, "tech_health_headlines.csv")
    # final_dataframe.to_csv(csv_file, index=False)
    #
    # print(f"\nDataset saved to: {os.path.abspath(csv_file)}")


if __name__ == "__main__":
    main()