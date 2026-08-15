import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date


SPORT_FEED = "https://feeds.bbci.co.uk/sport/rss.xml"

HTTP_CONFIG = {
    "User-Agent": "Mozilla/5.0"
}


def fetch_feed(feed_url):
    """Download the sports RSS feed."""

    feed_response = requests.get(feed_url, headers=HTTP_CONFIG)
    print(f"HTTP Status: {feed_response.status_code}")

    if feed_response.status_code != 200:
        print("Retrying request...")
        feed_response = requests.get(feed_url, headers=HTTP_CONFIG)
        print(f"Retry Status: {feed_response.status_code}")

    return feed_response


def extract_articles(feed_response, feed_url):
    """Extract sports headlines from the RSS feed."""

    xml_document = BeautifulSoup(feed_response.content, "xml")
    article_items = xml_document.find_all("item")

    print(f"Total Articles Found: {len(article_items)}")

    sports_data = []

    for news_item in article_items:
        article_title = news_item.title.get_text(strip=True)

        sports_data.append({
            "headline": article_title,
            "category": "Sports",
            "source": "Sofascore" if "sofascore" in feed_url.lower() else "BBC",
            "date": date.today().isoformat()
        })

    return sports_data


def prepare_dataset():
    """Create and clean the sports dataset."""

    rss_response = fetch_feed(SPORT_FEED)

    headline_data = extract_articles(rss_response, SPORT_FEED)

    sports_dataset = pd.DataFrame(headline_data)
    sports_dataset.drop_duplicates(subset="headline", inplace=True)

    return sports_dataset


def main():
    final_dataset = prepare_dataset()

    print(final_dataset.head())
    print(f"\nTotal Sports Headlines: {len(final_dataset)}")

    # Uncomment to save the dataset
    # import os
    #
    # project_directory = os.path.dirname(os.path.abspath(__file__))
    # csv_output = os.path.join(project_directory, "..", "data", "sports_headlines.csv")
    #
    # os.makedirs(os.path.dirname(csv_output), exist_ok=True)
    # final_dataset.to_csv(csv_output, index=False)


if __name__ == "__main__":
    main()