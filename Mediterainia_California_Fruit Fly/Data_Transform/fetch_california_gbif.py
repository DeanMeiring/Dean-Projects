import os
import requests
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "ca_gbif_occurrences.csv")

GBIF_URL = "https://api.gbif.org/v1/occurrence/search"

params = {
    "scientificName": "Ceratitis capitata",
    "country": "US",
    "hasCoordinate": "true",
    "hasGeospatialIssue": "false",
    "limit": 300,
    "offset": 0
}

all_records = []
print("Fetching California Medfly occurrences from GBIF...")

while True:
    response = requests.get(GBIF_URL, params=params)
    data = response.json()
    results = data.get("results", [])
    
    if not results:
        break
        
    for item in results:
        lat = item.get("decimalLatitude")
        lon = item.get("decimalLongitude")
        event_date = item.get("eventDate") or item.get("year")
        
        # California Bounding Box Check
        if lat and lon and event_date:
            if (32.5 <= lat <= 42.0) and (-124.5 <= lon <= -114.1):
                all_records.append({
                    "gbif_id": item.get("key"),
                    "event_date": event_date,
                    "latitude": lat,
                    "longitude": lon,
                    "state_province": item.get("stateProvince", "California")
                })
                
    params["offset"] += params["limit"]
    if params["offset"] >= data.get("count", 0):
        break

df = pd.DataFrame(all_records)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Fetch complete! Found {len(df)} occurrences in California.")
print(f"Saved to: '{OUTPUT_FILE}'")