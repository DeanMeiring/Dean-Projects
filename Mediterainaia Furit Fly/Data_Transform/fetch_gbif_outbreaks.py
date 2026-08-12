import os
import requests
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def fetch_gbif_outbreaks(country="ZA", scientific_name="Ceratitis capitata", max_records=1000):
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "scientificName": scientific_name,
        "country": country,
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "limit": max_records
    }
    
    print(f"Querying GBIF API for {scientific_name} in {country}...")
    response = requests.get(url, params=params)
    results = response.json().get("results", [])
    
    records = []
    for r in results:
        records.append({
            "gbif_id": r.get("key"),
            "event_date": r.get("eventDate"),
            "year": r.get("year"),
            "month": r.get("month"),
            "day": r.get("day"),
            "latitude": r.get("decimalLatitude"),
            "longitude": r.get("decimalLongitude"),
            "state_province": r.get("stateProvince"),
            "locality": r.get("locality")
        })
        
    df = pd.DataFrame(records)
    
    # Filter for Western Cape bounding box if state_province is missing
    # Western Cape approx bounds: Lat (-34.8 to -30.5), Lon (17.8 to 24.2)
    df = df[(df['latitude'] >= -34.8) & (df['latitude'] <= -30.5) & 
            (df['longitude'] >= 17.8) & (df['longitude'] <= 24.2)]
    
    output_path = os.path.join(SCRIPT_DIR, "gbif_medfly_occurrences_wc.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} Western Cape outbreak occurrences to '{output_path}'.")

if __name__ == "__main__":
    fetch_gbif_outbreaks()