import json
import os
import sys

import requests
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    api_key = os.getenv("CRUX_API_KEY", "")
    if not api_key:
        print("Missing CRUX_API_KEY in .env")
        sys.exit(1)

    origin = "https://www.solmarvillas.com"
    if len(sys.argv) > 1 and sys.argv[1].strip():
        origin = sys.argv[1].strip()
    if origin.endswith("/"):
        origin = origin[:-1]

    payload = {
        "origin": origin,
        "metrics": [
            "largest_contentful_paint",
            "interaction_to_next_paint",
            "cumulative_layout_shift",
        ],
    }

    url = f"https://chromeuxreport.googleapis.com/v1/records:queryHistoryRecord?key={api_key}"
    resp = requests.post(url, json=payload, timeout=30)

    print(f"Origin: {origin}")
    print(f"HTTP: {resp.status_code}")

    content_type = resp.headers.get("Content-Type", "")
    print(f"Content-Type: {content_type}")

    try:
        data = resp.json()
    except ValueError:
        print("Non-JSON response:")
        print(resp.text[:2000])
        return

    print(json.dumps(data, indent=2)[:4000])


if __name__ == "__main__":
    main()
