import json
import os
import time

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "wc_v2_region_coords.json")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Query candidates per region, tried in order until one returns a result.
# These are farming districts/valleys, not always incorporated towns, so a
# couple of fall back to the nearest named town as a centroid proxy.
REGION_QUERIES = {
    "Warm Bokkeveld": ["Warm Bokkeveld", "Op-die-Berg", "Ceres"],
    "Wolseley": ["Wolseley"],
    "Tulbagh": ["Tulbagh"],
    "Elgin & Grabouw": ["Grabouw", "Elgin, Western Cape"],
    "Vyeboom": ["Vyeboom", "Villiersdorp"],
}


def geocode(query):
    resp = requests.get(
        GEOCODE_URL,
        params={"name": query, "count": 5, "language": "en", "format": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []

    # Prefer a South African, Western Cape result if one is present.
    for r in results:
        if r.get("country_code") == "ZA" and r.get("admin1", "").lower().startswith("western cape"):
            return r
    return results[0] if results else None


def main():
    coords = {}
    for region, queries in REGION_QUERIES.items():
        found = None
        used_query = None
        for query in queries:
            found = geocode(query)
            used_query = query
            if found:
                break
            time.sleep(0.2)

        if not found:
            print(f"WARNING: could not geocode '{region}' with any of {queries}. Skipping — "
                  f"add it to '{OUTPUT_PATH}' manually before training.")
            continue

        coords[region] = {
            "lat": found["latitude"],
            "lon": found["longitude"],
            "matched_name": found.get("name"),
            "admin1": found.get("admin1"),
            "query_used": used_query,
        }
        print(f"{region}: matched '{found.get('name')}' ({found.get('admin1')}) "
              f"at {found['latitude']:.4f}, {found['longitude']:.4f} via query '{used_query}'")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(coords, f, indent=2)

    missing = set(REGION_QUERIES) - set(coords)
    if missing:
        raise RuntimeError(
            f"Failed to geocode: {sorted(missing)}. Fix '{OUTPUT_PATH}' by hand (or adjust "
            "REGION_QUERIES) before running fetch_weather_history_v2.py."
        )

    print(f"Saved coordinates for {len(coords)} regions to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    main()
