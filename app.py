import gzip
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError


CRUX_ENDPOINT = "https://chromeuxreport.googleapis.com/v1/records:queryRecord"
HISTORY_ENDPOINT = "https://chromeuxreport.googleapis.com/v1/records:queryHistoryRecord"

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

HISTORY_METRICS = {
    "lcp": "largest_contentful_paint",
    "inp": "interaction_to_next_paint",
    "cls": "cumulative_layout_shift",
}

TREND_TOLERANCE = {
    "lcp": 100.0,
    "inp": 50.0,
    "cls": 0.02,
}

STATUS_COLORS = {
    "Good": "#0B8043",
    "Needs Improvement": "#F09300",
    "Poor": "#C53929",
}


def lighten_hex(hex_color: str, factor: float = 0.8) -> str:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return hex_color
    try:
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
    except ValueError:
        return hex_color
    red = round(red + (255 - red) * factor)
    green = round(green + (255 - green) * factor)
    blue = round(blue + (255 - blue) * factor)
    return f"#{red:02X}{green:02X}{blue:02X}"


STATUS_COLORS_LIGHT = {
    key: lighten_hex(value, 0.82) for key, value in STATUS_COLORS.items()
}

TREND_COLORS = {
    "improving": "#188038",
    "regressing": "#d93025",
    "stable": "#5f6368",
}

METRIC_VIS = {
    "lcp": {"label": "LCP", "shape": "triangle-up"},
    "inp": {"label": "INP", "shape": "circle"},
    "cls": {"label": "CLS", "shape": "square"},
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


def read_urls_from_text(text: str) -> List[str]:
    if not text:
        return []
    urls: List[str] = []
    for line in text.splitlines():
        for raw in line.replace(",", " ").split():
            value = raw.strip()
            if value.startswith(("http://", "https://")):
                urls.append(value)
    return unique_preserve(urls)


def normalize_site_input(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if "://" not in text:
        text = f"https://{text}"
    parsed = urlparse(text)
    if not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def normalize_origin_input(value: str) -> str:
    return normalize_site_input(value)


def reset_sitemap_state() -> None:
    st.session_state["sitemap_urls"] = []
    st.session_state["sitemap_counts"] = {}
    st.session_state["sitemap_errors"] = []


def fetch_robots_sitemaps(site_url: str) -> List[str]:
    robots_url = urljoin(site_url, "/robots.txt")
    try:
        resp = requests.get(robots_url, timeout=20)
    except requests.RequestException:
        return []
    if resp.status_code != 200:
        return []

    sitemaps = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("sitemap:"):
            value = line.split(":", 1)[1].strip()
            if not value:
                continue
            if not value.startswith(("http://", "https://")):
                value = urljoin(site_url, value)
            sitemaps.append(value)
    return unique_preserve(sitemaps)


def parse_sitemap_xml(xml_text: str) -> Tuple[List[str], List[str]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return [], []

    def strip_ns(tag: str) -> str:
        return tag.split("}", 1)[1] if "}" in tag else tag

    root_tag = strip_ns(root.tag)
    if root_tag == "sitemapindex":
        sitemap_urls = []
        for sitemap in root.findall(".//{*}sitemap"):
            loc = sitemap.find("{*}loc")
            if loc is not None and loc.text:
                sitemap_urls.append(loc.text.strip())
        return sitemap_urls, []
    if root_tag == "urlset":
        urls = []
        for url in root.findall(".//{*}url"):
            loc = url.find("{*}loc")
            if loc is not None and loc.text:
                urls.append(loc.text.strip())
        return [], urls

    urls = [
        loc.text.strip()
        for loc in root.findall(".//{*}loc")
        if loc.text
    ]
    return [], urls


def fetch_sitemap_xml(sitemap_url: str) -> Optional[str]:
    try:
        resp = requests.get(sitemap_url, timeout=30)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    content = resp.content
    if sitemap_url.endswith(".gz"):
        try:
            content = gzip.decompress(content)
        except OSError:
            pass
    try:
        return content.decode(resp.encoding or "utf-8", errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def discover_urls_from_sitemaps(
    site_input: str,
    max_sitemaps: int,
    max_urls: int,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    site_url = normalize_site_input(site_input)
    if not site_url:
        return [], ["Invalid site input."], {}

    seed_sitemaps = fetch_robots_sitemaps(site_url)
    if not seed_sitemaps:
        seed_sitemaps = [urljoin(site_url, "/sitemap.xml")]

    queue = list(seed_sitemaps)
    seen_sitemaps = set()
    seen_urls = set()
    urls = []
    errors = []
    sitemap_counts: Dict[str, int] = {}

    while queue and len(seen_sitemaps) < max_sitemaps and len(urls) < max_urls:
        sitemap_url = queue.pop(0)
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)

        xml_text = fetch_sitemap_xml(sitemap_url)
        if not xml_text:
            errors.append(f"Failed to fetch {sitemap_url}")
            continue

        sitemap_urls, page_urls = parse_sitemap_xml(xml_text)
        normalized_pages = []
        for page in page_urls:
            if not page:
                continue
            if page.startswith(("http://", "https://")):
                normalized_pages.append(page)
            else:
                normalized_pages.append(urljoin(site_url, page))
        sitemap_counts[sitemap_url] = len(normalized_pages)

        for page in normalized_pages:
            if page in seen_urls:
                continue
            seen_urls.add(page)
            urls.append(page)
            if len(urls) >= max_urls:
                break

        for child in sitemap_urls:
            if child in seen_sitemaps or child in queue:
                continue
            queue.append(child)
            if len(seen_sitemaps) + len(queue) >= max_sitemaps:
                break

    return urls, errors, sitemap_counts


def format_period_end(period: Dict) -> str:
    last = period.get("lastDate", {})
    try:
        year = int(last.get("year"))
        month = int(last.get("month"))
        day = int(last.get("day", 1))
    except (TypeError, ValueError):
        return ""
    return f"{year}-{month:02d}-{day:02d}"


def extract_collection_periods(record: Dict) -> List[str]:
    periods = record.get("collectionPeriods")
    if isinstance(periods, list) and periods:
        return [format_period_end(period) for period in periods]
    return []


def extract_metric_timeseries(metric: Dict) -> List[Optional[float]]:
    percentiles = metric.get("percentilesTimeseries", {})
    values = None
    if isinstance(percentiles, dict):
        if "p75s" in percentiles:
            values = percentiles.get("p75s")
        elif "p75" in percentiles:
            values = percentiles.get("p75")
    elif isinstance(percentiles, list):
        values = [item.get("p75") for item in percentiles if isinstance(item, dict)]

    if not values:
        return []
    return [coerce_float(value) for value in values]


def latest_non_null(values: List[Optional[float]]) -> Optional[float]:
    for value in reversed(values):
        numeric = coerce_float(value)
        if numeric is not None:
            return numeric
    return None


def trend_label(metric_key: str, values: List[Optional[float]]) -> str:
    numeric = [coerce_float(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if len(numeric) < 2:
        return "stable"
    first = numeric[0]
    last = numeric[-1]
    tolerance = TREND_TOLERANCE.get(metric_key, 0.0)
    diff = last - first
    if abs(diff) <= tolerance:
        return "stable"
    return "improving" if diff < 0 else "regressing"


def query_history(api_key: str, origin: str, form_factor: str) -> Tuple[Optional[Dict], Optional[str]]:
    payload = {"origin": origin, "metrics": list(HISTORY_METRICS.values())}
    if form_factor != "ALL":
        payload["formFactor"] = form_factor

    resp = requests.post(
        f"{HISTORY_ENDPOINT}?key={api_key}",
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


def build_history_frame(record: Dict) -> Tuple[pd.DataFrame, str]:
    periods = extract_collection_periods(record)
    metric_series = {
        key: extract_metric_timeseries(record.get("metrics", {}).get(metric, {}))
        for key, metric in HISTORY_METRICS.items()
    }

    lengths = []
    if periods:
        lengths.append(len(periods))
    for values in metric_series.values():
        if values:
            lengths.append(len(values))

    if not lengths:
        return pd.DataFrame(), ""

    target_len = min(lengths)
    if periods:
        periods = periods[-target_len:]
    else:
        periods = [str(index + 1) for index in range(target_len)]

    data = {"period": periods}
    for key, values in metric_series.items():
        if not values:
            data[key.upper()] = [None] * target_len
            continue
        data[key.upper()] = values[-target_len:]

    last_period = periods[-1] if periods else ""
    return pd.DataFrame(data), last_period


def format_metric_value(metric_key: str, value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if metric_key == "cls":
        return f"{value:.3f}"
    return f"{value:.0f} ms"


def summary_statement(label: str, metric_key: str, value: Optional[float], trend: str) -> str:
    status = grade_value(metric_key, value)
    status_text = status.lower() if status else "unknown"
    trend_text = trend or "stable"
    joiner = "and" if trend_text == "stable" else "but"
    tooltip_value = format_metric_value(metric_key, value)
    status_color = STATUS_COLORS.get(status, "#5f6368")
    trend_color = TREND_COLORS.get(trend_text, "#5f6368")
    return (
        f"<div class='summary-row' title='Current p75: {tooltip_value}'>"
        f"{label} is <span style='color:{status_color}; font-weight:600;'>"
        f"{status_text}</span> {joiner} "
        f"<span style='color:{trend_color}; font-weight:600;'>"
        f"{trend_text}</span>"
        f"</div>"
    )


def build_cwv_chart(history_df: pd.DataFrame) -> alt.Chart:
    if history_df.empty:
        return alt.Chart(pd.DataFrame())

    history_df = history_df.copy()
    history_df["period_dt"] = pd.to_datetime(history_df["period"], errors="coerce")
    if history_df["period_dt"].isna().all():
        history_df["period_dt"] = pd.to_datetime(range(len(history_df)), unit="D")

    rows = []
    for index, row in history_df.iterrows():
        for metric_key in ["lcp", "inp", "cls"]:
            metric_label = METRIC_VIS[metric_key]["label"]
            value = row.get(metric_label)
            rows.append(
                {
                    "period": row.get("period"),
                    "period_dt": row.get("period_dt"),
                    "period_index": index,
                    "metric": metric_label,
                    "metric_key": metric_key,
                    "value": value,
                    "value_display": format_metric_value(metric_key, coerce_float(value)),
                    "norm": normalize_metric_value(metric_key, value),
                    "rating": grade_value(metric_key, value) or "Unknown",
                    "shape": METRIC_VIS[metric_key]["shape"],
                }
            )

    chart_df = pd.DataFrame(rows)
    periods = history_df["period_dt"].tolist()
    if not periods:
        return alt.Chart(pd.DataFrame())
    start_period = periods[0]
    end_period = periods[-1]

    band_df = pd.DataFrame(
        [
            {
                "x0": start_period,
                "x1": end_period,
                "y0": 0.0,
                "y1": 1.0,
                "color": "#e6f4ea",
            },
            {
                "x0": start_period,
                "x1": end_period,
                "y0": 1.0,
                "y1": 2.0,
                "color": "#fef7e0",
            },
            {
                "x0": start_period,
                "x1": end_period,
                "y0": 2.0,
                "y1": 3.0,
                "color": "#fce8e6",
            },
        ]
    )

    base = alt.Chart(chart_df).encode(
        x=alt.X("period_dt:T", axis=alt.Axis(title=None, labelAngle=-35)),
        y=alt.Y("norm:Q", scale=alt.Scale(domain=[0, 3]), axis=None),
    )

    band_layer = alt.Chart(band_df).mark_rect(opacity=1).encode(
        x=alt.X("x0:T", axis=None, scale=alt.Scale(domain=[start_period, end_period])),
        x2=alt.X2("x1:T"),
        y=alt.Y("y0:Q", axis=None, scale=alt.Scale(domain=[0, 3])),
        y2="y1:Q",
        color=alt.Color("color:N", scale=None, legend=None),
    )

    line_layer = base.mark_line(color="#5f6368", opacity=0.5).encode(
        detail="metric:N"
    )

    point_layer = base.mark_point(size=70, filled=True).encode(
        color=alt.Color(
            "rating:N",
            scale=alt.Scale(
                domain=["Good", "Needs Improvement", "Poor", "Unknown"],
                range=[
                    STATUS_COLORS["Good"],
                    STATUS_COLORS["Needs Improvement"],
                    STATUS_COLORS["Poor"],
                    "#9aa0a6",
                ],
            ),
            legend=None,
        ),
        shape=alt.Shape("shape:N", scale=None),
    )

    wide_df = history_df.copy()
    wide_df["LCP_display"] = wide_df["LCP"].map(lambda v: format_metric_value("lcp", coerce_float(v)))
    wide_df["INP_display"] = wide_df["INP"].map(lambda v: format_metric_value("inp", coerce_float(v)))
    wide_df["CLS_display"] = wide_df["CLS"].map(lambda v: format_metric_value("cls", coerce_float(v)))
    wide_df["period_label"] = wide_df["period_dt"].dt.strftime("%b %d, %Y")
    wide_df["period_label"] = wide_df["period_label"].fillna(wide_df["period"])

    hover_df = wide_df[
        ["period_dt", "period_label", "LCP_display", "INP_display", "CLS_display"]
    ].copy()
    hover_df = hover_df.sort_values("period_dt").reset_index(drop=True)

    if len(hover_df) == 1:
        half_span = pd.Timedelta(days=14)
        hover_df["x0"] = hover_df["period_dt"] - half_span
        hover_df["x1"] = hover_df["period_dt"] + half_span
    else:
        periods = hover_df["period_dt"].tolist()
        midpoints = []
        for index in range(len(periods) - 1):
            midpoints.append(periods[index] + (periods[index + 1] - periods[index]) / 2)
        edges = (
            [periods[0] - (midpoints[0] - periods[0])]
            + midpoints
            + [periods[-1] + (periods[-1] - midpoints[-1])]
        )
        hover_df["x0"] = edges[:-1]
        hover_df["x1"] = edges[1:]

    hover_df["y0"] = 0.0
    hover_df["y1"] = 3.0

    hover_layer = alt.Chart(hover_df).mark_rect(opacity=0).encode(
        x=alt.X("x0:T", axis=None, scale=alt.Scale(domain=[start_period, end_period])),
        x2=alt.X2("x1:T"),
        y=alt.Y("y0:Q", axis=None, scale=alt.Scale(domain=[0, 3])),
        y2="y1:Q",
        tooltip=[
            alt.Tooltip("period_label:N", title="Period end"),
            alt.Tooltip("LCP_display:N", title="LCP"),
            alt.Tooltip("INP_display:N", title="INP"),
            alt.Tooltip("CLS_display:N", title="CLS"),
        ],
    )

    chart = (
        alt.layer(
            band_layer,
            line_layer,
            point_layer,
            hover_layer,
        )
        .resolve_axis(x="shared", y="shared")
        .resolve_scale(x="shared", y="shared")
        .properties(height=320)
    )
    return chart


def build_device_metric_chart(
    device_df: pd.DataFrame,
    metric_key: str,
) -> Optional[alt.Chart]:
    if device_df.empty:
        return None

    value_col = f"{metric_key}_p75"
    if value_col not in device_df.columns:
        return None

    chart_df = device_df[["device", value_col]].copy()
    chart_df["value"] = chart_df[value_col].map(coerce_float)
    chart_df = chart_df.dropna(subset=["value"])
    if chart_df.empty:
        return None

    chart_df["value_display"] = chart_df["value"].map(
        lambda v: format_metric_value(metric_key, v)
    )
    chart_df["rating"] = chart_df["value"].map(
        lambda v: grade_value(metric_key, v) or "Unknown"
    )

    status_domain = ["Good", "Needs Improvement", "Poor", "Unknown"]
    status_range = [
        STATUS_COLORS["Good"],
        STATUS_COLORS["Needs Improvement"],
        STATUS_COLORS["Poor"],
        "#9aa0a6",
    ]
    metric_label = METRIC_VIS[metric_key]["label"]
    axis_title = metric_label
    if metric_key in ("lcp", "inp"):
        axis_title = f"{metric_label} (ms)"
    good_threshold = METRIC_DEFS[metric_key]["thresholds"][0]

    base = alt.Chart(chart_df).encode(
        x=alt.X(
            "device:N",
            title=None,
            sort=["Phone", "Desktop", "Tablet"],
        )
    )

    bar_layer = base.mark_bar().encode(
        y=alt.Y(
            "value:Q",
            title=axis_title,
            axis=alt.Axis(titlePadding=12),
        ),
        color=alt.Color(
            "rating:N",
            scale=alt.Scale(domain=status_domain, range=status_range),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("device:N", title="Device"),
            alt.Tooltip("value_display:N", title=METRIC_VIS[metric_key]["label"]),
            alt.Tooltip("rating:N", title="Status"),
        ],
    )

    rule_df = pd.DataFrame({"threshold": [good_threshold]})
    rule_layer = alt.Chart(rule_df).mark_rule(
        strokeDash=[6, 4],
        color="#5f6368",
    ).encode(
        y=alt.Y("threshold:Q"),
    )

    return (bar_layer + rule_layer).properties(
        height=280,
        title=alt.TitleParams(text=metric_label, offset=18),
    )


def update_origin_summary() -> None:
    origin_input = st.session_state.get("origin_input", "")
    show_devices = st.session_state.get("show_device_split", True)
    st.session_state["origin_summary"] = None
    st.session_state["origin_error"] = ""
    reset_sitemap_state()

    if not origin_input.strip():
        return

    normalized_origin = normalize_origin_input(origin_input)
    if not normalized_origin:
        st.session_state["origin_error"] = "Enter a valid origin like https://www.example.com."
        return

    api_key = get_default_api_key()
    if not api_key:
        st.session_state["origin_error"] = "Set CRUX_API_KEY in .env or Streamlit secrets."
        return

    history_data, history_error = query_history(api_key, normalized_origin, "ALL")
    if history_error:
        st.session_state["origin_error"] = history_error
        return

    record = history_data.get("record", {})
    history_df, last_period = build_history_frame(record)
    summary = {
        "origin": normalized_origin,
        "history_df": history_df,
        "last_period": last_period,
        "latest_values": {
            key: latest_non_null(history_df[key.upper()].tolist())
            for key in HISTORY_METRICS.keys()
        }
        if not history_df.empty
        else {},
        "trends": {
            key: trend_label(key, history_df[key.upper()].tolist())
            for key in HISTORY_METRICS.keys()
        }
        if not history_df.empty
        else {},
    }

    if show_devices:
        device_rows = []
        for device in ["PHONE", "DESKTOP", "TABLET"]:
            device_data, device_error = query_history(api_key, normalized_origin, device)
            if device_error:
                continue
            device_record = device_data.get("record", {})
            device_series = {
                key: extract_metric_timeseries(
                    device_record.get("metrics", {}).get(metric, {})
                )
                for key, metric in HISTORY_METRICS.items()
            }
            device_rows.append(
                {
                    "device": device.title(),
                    "lcp_p75": latest_non_null(device_series["lcp"]),
                    "inp_p75": latest_non_null(device_series["inp"]),
                    "cls_p75": latest_non_null(device_series["cls"]),
                }
            )
        summary["device_rows"] = device_rows

    st.session_state["origin_summary"] = summary


def render_origin_summary(container, summary: Dict, show_device_split: bool) -> None:
    container.subheader(f"Core Web Vitals summary for {summary.get('origin', '')}")
    history_df = summary.get("history_df")
    last_period = summary.get("last_period", "")
    if not isinstance(history_df, pd.DataFrame) or history_df.empty:
        container.warning("No history data returned for this origin.")
        return

    if last_period:
        container.caption(f"Latest period end: {last_period}")

    latest_values = summary.get("latest_values", {})
    trends = summary.get("trends", {})
    summary_html = "\n".join(
        [
            summary_statement(
                "Loading Performance",
                "lcp",
                latest_values.get("lcp"),
                trends.get("lcp", "stable"),
            ),
            summary_statement(
                "Interactivity",
                "inp",
                latest_values.get("inp"),
                trends.get("inp", "stable"),
            ),
            summary_statement(
                "Visual Stability",
                "cls",
                latest_values.get("cls"),
                trends.get("cls", "stable"),
            ),
        ]
    )
    container.markdown(summary_html, unsafe_allow_html=True)

    chart = build_cwv_chart(history_df)
    container.altair_chart(chart, use_container_width=True)

    if show_device_split:
        device_rows = summary.get("device_rows", [])
        if device_rows:
            container.subheader("Device split (latest p75 per form factor)")
            container.markdown(
                "<div class='device-split-spacer'></div>",
                unsafe_allow_html=True,
            )
            device_df = pd.DataFrame(device_rows)
            chart_cols = container.columns(3)
            for col, metric_key in zip(chart_cols, ["lcp", "inp", "cls"]):
                chart = build_device_metric_chart(device_df, metric_key)
                if chart is None:
                    col.info(f"No {METRIC_VIS[metric_key]['label']} data available.")
                    continue
                col.altair_chart(chart, use_container_width=True)
        else:
            container.info("Device split data unavailable for this origin.")


def coerce_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def format_share_percent(value: Optional[object]) -> Optional[float]:
    numeric = coerce_float(value)
    if numeric is None:
        return None
    return round(numeric * 100, 1)


def format_share_display(value: Optional[object]) -> str:
    numeric = coerce_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.1f}%"


def style_status_cell(value: object) -> str:
    if not isinstance(value, str):
        return ""
    color = STATUS_COLORS_LIGHT.get(value)
    if not color:
        return ""
    return f"background-color: {color}; font-weight: 600;"


def normalize_metric_value(metric_key: str, value: Optional[object]) -> Optional[float]:
    numeric = coerce_float(value)
    if numeric is None:
        return None
    good, ni = METRIC_DEFS[metric_key]["thresholds"]
    if good <= 0:
        return None
    if numeric <= good:
        return numeric / good
    if numeric <= ni:
        return 1.0 + (numeric - good) / max(ni - good, 1e-9)
    return min(2.0 + (numeric - ni) / max(ni, 1e-9), 3.0)


def grade_value(metric_key: str, value: Optional[object]) -> str:
    numeric = coerce_float(value)
    if numeric is None:
        return ""
    thresholds = METRIC_DEFS[metric_key]["thresholds"]
    if numeric <= thresholds[0]:
        return "Good"
    if numeric <= thresholds[1]:
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


def build_export_headers() -> Dict[str, str]:
    headers = {
        "url": "URL",
        "collection_period": "Collection period",
        "form_factor": "Form factor",
    }
    for metric_key in METRIC_DEFS.keys():
        label = metric_key.upper()
        unit = " (ms)" if metric_key in ("lcp", "inp", "fcp", "ttfb") else ""
        headers[f"{metric_key}_p75"] = f"{label} p75{unit}"
        headers[f"{metric_key}_rating"] = f"{label} status"
        headers[f"{metric_key}_good"] = f"{label} good share (%)"
        headers[f"{metric_key}_ni"] = f"{label} needs improvement share (%)"
        headers[f"{metric_key}_poor"] = f"{label} poor share (%)"
    return headers


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


def get_default_api_key() -> str:
    if hasattr(st, "secrets"):
        try:
            secret = st.secrets.get("CRUX_API_KEY")
        except StreamlitSecretNotFoundError:
            secret = None
        if secret:
            return secret
    return os.getenv("CRUX_API_KEY", "")


def main() -> None:
    load_dotenv()
    st.set_page_config(
        page_title="Bulk CrUX Fetcher | Journey Further",
        page_icon="favicon.png",
        layout="wide",
    )
    st.title("Bulk CrUX Fetcher")
    st.caption(
        "Paste URL lists, upload a CSV, or fetch from sitemaps to pull Core Web Vitals field data from the CrUX API."
    )
    st.markdown(
        "<style>"
        ".summary-row{margin:0.2rem 0 0.6rem;font-size:1.05rem;}"
        ".device-split-spacer{height:0.9rem;display:block;}"
        "</style>",
        unsafe_allow_html=True,
    )

    st.subheader("Origin summary")
    origin_input = st.text_input(
        "Enter a site to fetch a summary. (Unlocks sitemap crawling)",
        value="",
        placeholder="https://www.example.com",
        help="You can paste a domain like example.com; https:// will be assumed.",
        key="origin_input",
        on_change=update_origin_summary,
    )
    st.caption("Press Enter in the origin field to fetch summary data.")
    show_device_split = st.checkbox(
        "Show device split (per form factor)",
        value=True,
        key="show_device_split",
        on_change=update_origin_summary,
    )
    origin_summary = st.container()

    with st.sidebar:
        form_factor = st.selectbox("Form factor", ["ALL", "PHONE", "DESKTOP", "TABLET"])
        delay = st.slider("Delay between requests (seconds)", 0.0, 2.0, 0.5, 0.1)

    st.subheader("Choose an Input")
    source_choice = st.radio(
        "URL source",
        ["Paste URLs", "CSV upload", "Sitemaps"],
        horizontal=True,
        index=0,
    )
    urls_preview: List[str] = []
    upload = None
    pasted_urls = ""

    if "sitemap_urls" not in st.session_state:
        st.session_state["sitemap_urls"] = []
    if "sitemap_counts" not in st.session_state:
        st.session_state["sitemap_counts"] = {}
    if "sitemap_errors" not in st.session_state:
        st.session_state["sitemap_errors"] = []
    if "origin_summary" not in st.session_state:
        st.session_state["origin_summary"] = None
    if "origin_error" not in st.session_state:
        st.session_state["origin_error"] = ""

    origin_error = st.session_state.get("origin_error", "")
    summary_data = st.session_state.get("origin_summary")
    if origin_error:
        origin_summary.error(origin_error)
    elif summary_data:
        render_origin_summary(origin_summary, summary_data, show_device_split)

    if source_choice == "Paste URLs":
        pasted_urls = st.text_area(
            "Paste URLs (one per line)",
            height=160,
            placeholder="https://www.example.com/page\nhttps://www.example.com/other",
        )
        urls_preview = read_urls_from_text(pasted_urls)
        st.write(f"Found {len(urls_preview)} URL(s).")
    elif source_choice == "CSV upload":
        upload = st.file_uploader("Upload CSV", type=["csv"])
        urls_preview = read_urls_from_upload(upload)
        st.write(f"Found {len(urls_preview)} URL(s).")
    else:
        max_sitemaps = st.number_input("Max sitemaps to crawl", 1, 200, 20, 1)
        max_urls = st.number_input("Max URLs to fetch", 1, 10000, 500, 50)
        st.caption("Sitemaps are discovered via the robots.txt of the origin specified above. Subject to firewalls.")

        if st.button("Fetch sitemaps"):
            normalized_origin = normalize_origin_input(origin_input)
            if not normalized_origin:
                st.error("Enter a valid origin above (e.g. https://www.example.com).")
            else:
                status = st.empty()
                status.write("Discovering sitemaps...")
                urls, errors, sitemap_counts = discover_urls_from_sitemaps(
                    normalized_origin,
                    int(max_sitemaps),
                    int(max_urls),
                )
                st.session_state["sitemap_urls"] = urls
                st.session_state["sitemap_counts"] = sitemap_counts
                st.session_state["sitemap_errors"] = errors
                status.write("Sitemap fetch complete.")

        if st.session_state["sitemap_urls"]:
            st.write(f"Fetched {len(st.session_state['sitemap_urls'])} URL(s) from sitemaps.")
        if st.session_state["sitemap_counts"]:
            summary_rows = [
                {"Sitemap": key, "Pages detected": count}
                for key, count in st.session_state["sitemap_counts"].items()
            ]
            summary_df = pd.DataFrame(summary_rows).sort_values("Pages detected", ascending=False)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        if st.session_state["sitemap_errors"]:
            st.warning(f"{len(st.session_state['sitemap_errors'])} sitemap fetch issue(s).")

    if st.button("Run"):
        api_key = get_default_api_key()
        if not api_key:
            st.error("Set CRUX_API_KEY in .env or Streamlit secrets to continue.")
            return

        urls: List[str] = []
        if source_choice == "Paste URLs":
            urls = read_urls_from_text(pasted_urls)
            if not urls:
                st.error("No valid URLs found in the pasted list.")
                return
        elif source_choice == "CSV upload":
            urls = read_urls_from_upload(upload)
            if not urls:
                st.error("No valid URLs found in the CSV.")
                return
        else:
            urls = st.session_state.get("sitemap_urls", [])
            if not urls:
                st.error("Fetch sitemaps first or switch to Paste URLs or CSV upload.")
                return

        if urls:
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

                results.append(row)
                progress.progress(index / len(urls))
                if delay:
                    time.sleep(delay)

            status.write("Done.")
            df = pd.DataFrame(results).rename(columns=build_export_headers())
            share_cols = [col for col in df.columns if col.endswith("share (%)")]
            for col in share_cols:
                df[col] = df[col].map(format_share_percent)
            status_cols = [col for col in df.columns if col.endswith(" status")]
            styled_df = df.style
            if share_cols:
                styled_df = styled_df.format({col: format_share_display for col in share_cols})
            if status_cols:
                styled_df = styled_df.applymap(style_status_cell, subset=status_cols)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
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

# ===== Centered footer (replaces sidebar logo & copyright) =====
import base64, mimetypes, datetime, os
from pathlib import Path

def _data_uri_for_logo():
    # try common locations / names
    candidates = [
        Path("logo.svg"), Path("logo.png"),
        Path("static/logo.svg"), Path("static/logo.png")
    ]
    for p in candidates:
        if p.exists():
            mime = mimetypes.guess_type(p.name)[0] or "image/png"
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
    return None  # fallback: no logo found

def render_footer(bg="#2b0573", max_h=70):
    current_year = datetime.datetime.now().year
    logo_uri = _data_uri_for_logo()

    img_html = (
        f'<img src="{logo_uri}" style="max-height:{max_h}px; width:auto; margin-bottom:0px;" />'
        if logo_uri else ""
    )

    st.markdown(
        f"""
        <div style="background:{bg}; padding:5px; text-align:center; margin-top:40px; border-radius:10px; ">
            {img_html}
            <div style="color:#fff; font-size:0.9em;">
                &copy; {current_year}
                <a href="https://www.journeyfurther.com/?utm_source=bulk-crux-tool&utm_medium=footer&utm_campaign=bulk-crux-tool"
                   target="_blank" style="color:#fff; text-decoration:none;">Journey Further</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

render_footer()  # call at the very end of the script
