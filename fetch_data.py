import requests
import os
import json
import time
from dotenv import load_dotenv

# Load hidden variables from the .env file
load_dotenv()
# API Setup
API_KEY = os.getenv("FOOTBALL_API_KEY")
URL = "https://v3.football.api-sports.io/fixtures"

HEADERS = {
    "x-apisports-key": API_KEY
}
SEASONS = ["2020","2021","2022","2023","2024","2025"]
# Prem api is 39

my_dict = {}
for year in SEASONS:
    PARAMS = {
    "league": "39",
    "season": year
    }
    response = requests.get(URL, headers=HEADERS, params=PARAMS)
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Retrieved {len(data['response'])} matches.")
        my_dict[year] = data['response']
    else:
        try:
            error_message = response.json().get('message', 'No additional details provided.')
            print(f"Error {response.status_code}: {error_message}")
        except Exception:
            print(f"Error {response.status_code}: {response.text}")
    time.sleep(1)
with open("Premier_League_Seasons_Data.json", "w") as f:
    json.dump(my_dict, f, indent=4)