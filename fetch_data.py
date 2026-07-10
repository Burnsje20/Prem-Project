import requests
import os
import json
from dotenv import load_dotenv

# Load hidden variables from the .env file
load_dotenv()
# API Setup
API_KEY = os.getenv("FOOTBALL_API_KEY")
URL = "https://v3.football.api-sports.io/fixtures"

HEADERS = {
    "x-apisports-key": API_KEY
}

# Prem api is 39
PARAMS = {
    "league": "39",
    "season": "2023" #2023-2024 season
    }

response = requests.get(URL, headers=HEADERS, params=PARAMS)
if response.status_code == 200:
    data = response.json()
    print(f"Success! Retrieved {len(data['response'])} matches.")
    # Let's inspect the first match to see what the data looks like
    if data['response']:
        print(json.dumps(data['response'][0], indent=2))
else:
    try:
        error_message = response.json().get('message', 'No additional details provided.')
        print(f"Error {response.status_code}: {error_message}")
    except Exception:
        print(f"Error {response.status_code}: {response.text}")