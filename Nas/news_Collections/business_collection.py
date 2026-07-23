import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date


RSS_LINK = "https://feeds.bbci.co.uk/news/business/rss.xml"

REQUEST_INFO = {
    "User-Agent": "Mozilla/5.0"
}


def download_rss(feed_address):
    """Download the RSS feed."""

    server_response = requests.get(feed_address, headers=REQUEST_INFO)
    print(f"HTTP Response: {server_response.status_code}")

    if server_response.status_code != 200:
        print("Trying the feed again...")
        server_response = requests.get(feed_address, headers=REQUEST_INFO)
        print(f"Retry Response: {server_response.status_code}")

    return server_response


def get_business_headlines(server_response, feed_address):
    """Extract headlines from the RSS feed."""

    parsed_xml = BeautifulSoup(server_response.content, "xml")
    article_list = parsed_xml.find_all("item")

    print(f"Articles Retrieved: {len(article_list)}")

    business_news = []

    for article in article_list:
        article_heading = article.title.get_text(strip=True)

        business_news.append({
            "headline": article_heading,
            "category": "Business",
            "source": "Sofascore" if "sofascore" in feed_address.lower() else "BBC",
            "date": date.today().isoformat()
        })

    return business_news


def build_dataframe():
    """Create a DataFrame of business headlines."""

    feed_data = download_rss(RSS_LINK)

    extracted_news = get_business_headlines(feed_data, RSS_LINK)

    business_df = pd.DataFrame(extracted_news)
    business_df.drop_duplicates(subset="headline", inplace=True)

    return business_df


def main():
    final_data = build_dataframe()

    print(final_data.head())
    print(f"\nTotal Business Headlines: {len(final_data)}")

    # Uncomment to save the data
    # import os
    #
    # current_path = os.path.dirname(os.path.abspath(__file__))
    # save_file = os.path.join(current_path, "..", "data", "business_headlines.csv")
    #
    # os.makedirs(os.path.dirname(save_file), exist_ok=True)
    # final_data.to_csv(save_file, index=False)


if __name__ == "__main__":
    main()