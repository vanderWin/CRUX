import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv


CRUX_ENDPOINT = "https://chromeuxreport.googleapis.com/v1/records:queryRecord"
PSI_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"

METRIC_DEFS = {
    "lcp": {
        "crux_keys": ["largest_contentful_paint"],
        "thresholds": (2500, 4000),
        "label": "Largest Contentful Paint (ms)",
    },
    "inp": {
        "crux_keys": ["interaction_to_next_paint"],
        "thresholds": (200, 500),
        "label": "Interaction to Next Paint (ms)",
    },
    "cls": {
        "crux_keys": ["cumulative_layout_shift"],
        "thresholds": (0.1, 0.25),
        "label": "Cumulative Layout Shift",
    },
    "fcp": {
        "crux_keys": ["first_contentful_paint"],
        "thresholds": (1800, 3000),
        "label": "First Contentful Paint (ms)",
    },
    "ttfb": {
        "crux_keys": ["experimental_time_to_first_byte", "time_to_first_byte"],
        "thresholds": (800, 1800),
        "label": "Time to First Byte (ms)",
    },
}

PSI_AUDIT_KEYS = {
    "lcp": ["largest-contentful-paint"],
    "inp": ["interaction-to-next-paint", "experimental-interaction-to-next-paint"],
    "cls": ["cumulative-layout-shift"],
    "fcp": ["first-contentful-paint"],
    "ttfb": ["server-response-time"],
}


def unique_preserve(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def read_urls_from_csv(path: str) -> List[str]:
    try:
        df = pd.read_csv(path, header=None)
    except FileNotFoundError:
        return []
    except Exception:
        return []

    urls = []
    for raw in df.iloc[:, 0].astype(str).tolist():
        value = raw.strip()
        if value.startswith(("http://", "https://")):
            urls.append(value)
    return unique_preserve(urls)


def read_urls_from_upload(upload) -> List[str]:
    if upload is None:
        return []
    df = pd.read_csv(upload, header=None)
    urls = []
    for raw in df.iloc[:, 0].astype(str).tolist():
        value = raw.strip()
        if value.startswith(("http://", "https://")):
            urls.append(value)
    return unique_preserve(urls)


def grade_value(metric_key: str, value: Optional[float]) -> str:
    if value is None:
        return ""
    thresholds = METRIC_DEFS[metric_key]["thresholds"]
    if value <= thresholds[0]:
        return "Good"
    if value <= thresholds[1]:
        return "Needs Improvement"
    return "Poor"


def format_collection_period(record: Dict) -> str:
    period = record.get("collectionPeriod", {})
    first = period.get("firstDate", {})
    last = period.get("lastDate", {})
    if not first or not last:
        return ""
    try:
        first_year = int(first.get("year"))
        first_month = int(first.get("month"))
        last_year = int(last.get("year"))
        last_month = int(last.get("month"))
    except (TypeError, ValueError):
        return ""
    return f"{first_year}-{first_month:02d} to {last_year}-{last_month:02d}"


def get_metric_record(metrics: Dict, keys: List[str]) -> Tuple[Optional[str], Optional[Dict]]:
    for key in keys:
        if key in metrics:
            return key, metrics[key]
    return None, None


def get_histogram_density(metric: Dict, index: int) -> Optional[float]:
    try:
        return metric.get("histogram", [])[index].get("density")
    except (IndexError, AttributeError):
        return None


def parse_crux_response(data: Dict) -> Dict:
    record = data.get("record", {})
    metrics = record.get("metrics", {})

    row = {
        "collection_period": format_collection_period(record),
        "form_factor": record.get("key", {}).get("formFactor", ""),
    }

    for short_name, meta in METRIC_DEFS.items():
        _, metric = get_metric_record(metrics, meta["crux_keys"])
        p75 = None
        if metric:
            p75 = metric.get("percentiles", {}).get("p75")
        row[f"{short_name}_p75"] = p75
        row[f"{short_name}_rating"] = grade_value(short_name, p75)
        row[f"{short_name}_good"] = get_histogram_density(metric, 0) if metric else None
        row[f"{short_name}_ni"] = get_histogram_density(metric, 1) if metric else None
        row[f"{short_name}_poor"] = get_histogram_density(metric, 2) if metric else None

    return row


def parse_error_message(resp: requests.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}"
    error = data.get("error", {})
    return error.get("message", f"HTTP {resp.status_code}")


def query_crux(api_key: str, url: str, form_factor: str) -> Tuple[Optional[Dict], Optional[str]]:
    payload = {"url": url}
    if form_factor != "ALL":
        payload["formFactor"] = form_factor

    resp = requests.post(
        f"{CRUX_ENDPOINT}?key={api_key}",
        json=payload,
        timeout=30,
    )

    if resp.status_code != 200:
        return None, parse_error_message(resp)

    data = resp.json()
    if "record" not in data:
        error = data.get("error", {}).get("message", "No record returned")
        return None, error
    return data, None


def extract_audit(audits: Dict, keys: List[str]) -> Optional[Dict]:
    for key in keys:
        if key in audits:
            return audits[key]
    return None


def query_pagespeed(api_key: str, url: str, strategy: str) -> Tuple[Optional[Dict], Optional[str]]:
    params = {"url": url, "strategy": strategy, "key": api_key}
    resp = requests.get(PSI_ENDPOINT, params=params, timeout=60)
    if resp.status_code != 200:
        return None, parse_error_message(resp)
    return resp.json(), None


def parse_pagespeed_response(data: Dict) -> Dict:
    audits = data.get("lighthouseResult", {}).get("audits", {})
    row = {}
    for short_name, keys in PSI_AUDIT_KEYS.items():
        audit = extract_audit(audits, keys)
        if not audit:
            row[f"psi_{short_name}_display"] = None
            row[f"psi_{short_name}_score"] = None
            continue
        row[f"psi_{short_name}_display"] = audit.get("displayValue")
        row[f"psi_{short_name}_score"] = audit.get("score")
    return row


def get_default_api_key() -> str:
    secret = st.secrets.get("CRUX_API_KEY") if hasattr(st, "secrets") else None
    if secret:
        return secret
    return os.getenv("CRUX_API_KEY", "")


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="CrUX CWV Fetcher", layout="wide")
    st.title("CrUX CWV Fetcher")
    st.caption("Load URLs from a CSV and pull Core Web Vitals field data from the CrUX API.")

    with st.sidebar:
        api_key = st.text_input(
            "CrUX API key",
            type="password",
            value=get_default_api_key(),
        )
        form_factor = st.selectbox("Form factor", ["ALL", "PHONE", "DESKTOP", "TABLET"])
        delay = st.slider("Delay between requests (seconds)", 0.0, 2.0, 0.4, 0.1)
        include_psi = st.checkbox("Also fetch PageSpeed Insights (lab data)")
        psi_strategy = st.selectbox("PSI strategy", ["mobile", "desktop"])

    st.subheader("Input")
    upload = st.file_uploader("Upload CSV", type=["csv"])
    csv_path = st.text_input("Or read from CSV path", value="pages.csv")

    urls = read_urls_from_upload(upload) if upload else read_urls_from_csv(csv_path)
    st.write(f"Found {len(urls)} URL(s).")

    if st.button("Run"):
        if not api_key:
            st.error("Enter a CrUX API key to continue.")
            return
        if not urls:
            st.error("No valid URLs found in the CSV.")
            return

        results = []
        progress = st.progress(0)
        status = st.empty()

        for index, url in enumerate(urls, start=1):
            status.write(f"Fetching {index}/{len(urls)}: {url}")
            row = {"url": url}

            crux_data, crux_error = query_crux(api_key, url, form_factor)
            if crux_error:
                row["crux_error"] = crux_error
            else:
                row.update(parse_crux_response(crux_data))

            if include_psi:
                psi_data, psi_error = query_pagespeed(api_key, url, psi_strategy)
                if psi_error:
                    row["psi_error"] = psi_error
                else:
                    row.update(parse_pagespeed_response(psi_data))

            results.append(row)
            progress.progress(index / len(urls))
            if delay:
                time.sleep(delay)

        status.write("Done.")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download results CSV",
            data=df.to_csv(index=False),
            file_name="crux_results.csv",
            mime="text/csv",
        )

    st.subheader("Metric notes")
    notes = [
        f"{key.upper()}: {meta['label']}"
        for key, meta in METRIC_DEFS.items()
    ]
    st.write(", ".join(notes))


if __name__ == "__main__":
    main()
