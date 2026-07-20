import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date

url = "import requests"


url = "https://feeds.bbci.co.uk/sport/rss.xml"
headers = {"User-Agent": "Mozilla/5.0"}  # some servers block requests without this

response = requests.get(url, headers=headers)
print("Status code:", response.status_code)

if response.status_code != 200:
    # fallback to BBC sport feed if CNN fails
    url = "https://feeds.bbci.co.uk/sport/rss.xml"
    response = requests.get(url, headers=headers)
    print("Fallback status code:", response.status_code)

soup = BeautifulSoup(response.content, 'xml')  # 'xml' not 'html.parser'

items = soup.find_all('item')
print(f"Found {len(items)} items")

records = []
for item in items:
    headline = item.title.get_text(strip=True)
    records.append({
        "headline": headline,
        "category": "Sports",
        "source": "Sofascore" if "sofascore" in url else "BBC",
        "date": date.today().isoformat()
    })

df = pd.DataFrame(records)
df.drop_duplicates(subset="headline", inplace=True)
print(df.head())
print(f"\nTotal sports headlines collected: {len(df)}")

# import os

# script_dir = os.path.dirname(os.path.abspath(__file__))
# output_path = os.path.join(script_dir, "..", "data", "sports_headlines.csv")
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# df.to_csv(output_path, index=False)



