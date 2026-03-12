"""
Chicago Booth Course Recommender
Streamlit application for course selection, bid prediction, and degree tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from pathlib import Path
import sys

# Add Data directory to path
DATA_DIR = Path(__file__).parent / "Data"
sys.path.insert(0, str(DATA_DIR))

from data_processing import (
    load_bid_data, load_evaluation_data, build_course_features, load_course_list,
)
from degree_data import (
    DEGREE_REQUIREMENTS, CONCENTRATIONS,
    check_course_requirements, get_all_concentration_courses,
)
from bid_model_v2 import predict_two_stage

# ============================================================
# Configuration
# ============================================================
MAROON = "#800000"
TEAL = "#7EBEC5"
DARK_GRAY = "#58595B"
LIGHT_GRAY = "#D6D6CE"
BODY_TEXT = "#333333"

st.set_page_config(
    page_title="Booth Course Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown(f"""
<style>
    .stApp {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    h1, h2, h3 {{
        color: {MAROON};
    }}
    .metric-card {{
        background: white;
        border: 1px solid {LIGHT_GRAY};
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: bold;
        color: {MAROON};
    }}
    .metric-label {{
        font-size: 13px;
        color: {DARK_GRAY};
    }}
    .bid-free {{
        color: #2ecc71;
        font-weight: bold;
    }}
    .bid-low {{
        color: {TEAL};
        font-weight: bold;
    }}
    .bid-moderate {{
        color: #f39c12;
        font-weight: bold;
    }}
    .bid-high {{
        color: #e74c3c;
        font-weight: bold;
    }}
    .bid-extreme {{
        color: {MAROON};
        font-weight: bold;
    }}
    .header-banner {{
        background: linear-gradient(135deg, {MAROON}, #5a0000);
        color: white;
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 20px;
    }}
    .header-banner h1 {{
        color: white !important;
        margin: 0;
    }}
    .header-banner p {{
        color: {LIGHT_GRAY};
        margin: 5px 0 0 0;
    }}
    div[data-testid="stMetricValue"] {{
        color: {MAROON};
    }}
    /* Navigation radio styling */
    div[data-testid="stRadio"] > div {{
        gap: 0px;
        justify-content: center;
    }}
    div[data-testid="stRadio"] label {{
        font-size: 15px;
        font-weight: 600;
        padding: 10px 24px;
        border-bottom: 2px solid transparent;
        cursor: pointer;
    }}
    div[data-testid="stRadio"] label[data-checked="true"] {{
        border-bottom: 3px solid {MAROON};
        color: {MAROON};
    }}
    /* Hide sidebar */
    section[data-testid="stSidebar"] {{
        display: none;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data Loading (cached)
# ============================================================
@st.cache_data(ttl=300)
def load_all_data():
    features, bid_df, eval_df = build_course_features()
    course_list = load_course_list()
    return features, bid_df, eval_df, course_list


@st.cache_resource
def load_model():
    model_path = DATA_DIR / "bid_model_v2.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


# ============================================================
# Helper Functions
# ============================================================
CONC_COLORS = {
    "Accounting": "#8B4513",
    "Applied Artificial Intelligence": "#1a73e8",
    "Behavioral Science": "#9C27B0",
    "Business Analytics": "#0097A7",
    "Business, Society and Sustainability": "#2E7D32",
    "Econometrics and Statistics": "#5C6BC0",
    "Economics": "#E65100",
    "Entrepreneurship": "#C62828",
    "Finance": "#1565C0",
    "Analytic Finance": "#0D47A1",
    "General Management": "#4E342E",
    "Healthcare": "#00838F",
    "International Business": "#AD1457",
    "Marketing Management": "#6A1B9A",
    "Operations Management": "#33691E",
    "Strategic Management": "#BF360C",
}


def _build_req_tags(req_info):
    """Build HTML badge tags for Foundation/FLMBE/Concentration."""
    tags = ""
    if req_info["foundations"]:
        tags += f"<span style='background:{MAROON};color:white;padding:2px 8px;border-radius:10px;font-size:11px;margin-right:4px;'>Foundation: {', '.join(req_info['foundations'])}</span>"
    if req_info["flmbe"]:
        flmbe_names = ', '.join(f.split(' > ')[1] for f in req_info['flmbe'])
        tags += f"<span style='background:{TEAL};color:white;padding:2px 8px;border-radius:10px;font-size:11px;margin-right:4px;'>FLMBE: {flmbe_names}</span>"
    if req_info["concentrations"]:
        for conc in req_info["concentrations"]:
            c_color = CONC_COLORS.get(conc, DARK_GRAY)
            tags += f"<span style='background:{c_color};color:white;padding:2px 8px;border-radius:10px;font-size:11px;margin-right:4px;margin-top:2px;display:inline-block;'>{conc}</span>"
    if not tags:
        tags = f"<span style='background:{LIGHT_GRAY};color:{DARK_GRAY};padding:2px 8px;border-radius:10px;font-size:11px;'>Elective</span>"
    return tags


def get_bid_level(price):
    """Classify bid price into demand level."""
    if price == 0 or pd.isna(price):
        return "Free", "bid-free"
    elif price < 2000:
        return "Low", "bid-low"
    elif price < 6000:
        return "Moderate", "bid-moderate"
    elif price < 12000:
        return "High", "bid-high"
    else:
        return "Very High", "bid-extreme"


def format_price(price):
    if pd.isna(price) or price == 0:
        return "Free"
    return f"{int(price):,} pts"


def compute_degree_progress(taken_courses):
    """Compute degree requirement completion given taken course codes."""
    # LEAD (31001) is 0-credit — exclude from unit count
    lead_courses = set(DEGREE_REQUIREMENTS["leadership"].get("courses", []))
    credit_courses = [c for c in taken_courses if c not in lead_courses]
    progress = {
        "leadership": bool(lead_courses & set(taken_courses)),
        "foundations": {},
        "flmbe": {},
        "total_units": len(credit_courses) * 100,
    }

    for area_name, area in DEGREE_REQUIREMENTS["foundations"]["areas"].items():
        all_valid = set(area["basic"]) | set(area["substitutes"])
        fulfilled = bool(all_valid & set(taken_courses))
        progress["foundations"][area_name] = fulfilled

    for cat_name, category in DEGREE_REQUIREMENTS["flmbe"]["categories"].items():
        for area_name, area in category.items():
            all_valid = set(area["basic"]) | set(area["substitutes"])
            fulfilled = bool(all_valid & set(taken_courses))
            progress["flmbe"][f"{area_name}"] = fulfilled

    progress["foundations_complete"] = sum(progress["foundations"].values())
    progress["flmbe_complete"] = sum(progress["flmbe"].values())
    progress["units_remaining"] = max(0, 2000 - progress["total_units"])

    return progress


def get_requirement_filter_codes(filter_type):
    """Get course codes for a requirement filter."""
    codes = set()
    if filter_type == "Foundation":
        for area in DEGREE_REQUIREMENTS["foundations"]["areas"].values():
            codes.update(area["basic"])
            codes.update(area["substitutes"])
    elif filter_type.startswith("FLMBE"):
        # Extract area name after "FLMBE: "
        area_name = filter_type.replace("FLMBE: ", "")
        for cat in DEGREE_REQUIREMENTS["flmbe"]["categories"].values():
            for name, area in cat.items():
                if name == area_name:
                    codes.update(area["basic"])
                    codes.update(area["substitutes"])
    return codes


# ============================================================
# LLM-powered Natural Language Search
# ============================================================
def llm_search(query, section_data):
    """
    Use Claude API to convert a natural language query into structured filters,
    then apply them to section_data and return filtered results.
    """
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None, "Set ANTHROPIC_API_KEY in .streamlit/secrets.toml to enable AI Search."
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        return None, f"API error: {e}"

    concentration_list = sorted(CONCENTRATIONS.keys())
    foundation_list = list(DEGREE_REQUIREMENTS["foundations"]["areas"].keys())
    flmbe_list = [area for cat in DEGREE_REQUIREMENTS["flmbe"]["categories"].values() for area in cat]

    system_prompt = f"""You are a course search assistant for Chicago Booth MBA. Convert the user's natural language query into structured JSON filters.

Available filters (all optional — only include filters mentioned or implied by the query):
- "search": free-text keyword to match against course code, title, or instructor name
- "quarters": list from ["Autumn 2025", "Winter 2026", "Spring 2026", "Summer 2026"]
- "days": list from ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
- "time_slots": list from ["Morning", "Afternoon", "Evening 1 (5 PM)", "Evening 2 (6 PM)"]
- "foundations": list from {json.dumps(foundation_list)}
- "flmbe": list from {json.dumps(flmbe_list)}
- "concentrations": list from {json.dumps(concentration_list)}
- "max_bid": maximum Phase 1 bid price (integer). Use if user says "cheap", "low bid", "affordable", "under X points", etc.
- "min_rating": minimum course rating 1-5 (float). Use if user says "highly rated", "good reviews", "above X rating", etc.
- "sort": one of ["price_asc", "price_desc", "rating_desc", "hours_asc", "code"]

Respond with ONLY valid JSON. No explanation. Examples:
- "Tuesday evening finance courses" → {{"days": ["Tuesday"], "time_slots": ["Evening 1 (5 PM)", "Evening 2 (6 PM)"], "concentrations": ["Finance"]}}
- "cheap courses with high ratings" → {{"max_bid": 2000, "min_rating": 4.0, "sort": "price_asc"}}
- "spring strategy courses" → {{"quarters": ["Spring 2026"], "concentrations": ["Strategic Management"]}}
- "Professor Rajan" → {{"search": "Rajan"}}"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
        )
        raw = response.content[0].text.strip()
        # Parse JSON (handle markdown code blocks if any)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        filters = json.loads(raw)
    except json.JSONDecodeError:
        return None, f"Could not parse AI response: {raw}"
    except Exception as e:
        return None, f"API error: {e}"

    # Apply filters to section_data
    df = section_data.copy()
    df["display_price"] = df["ci_p1_mean"].fillna(df["p1_price_mean"])
    df["display_rating"] = df["instr_rating"].fillna(df["eval_overall"])
    df["display_hours"] = df["instr_hours"].fillna(df["eval_hours"])
    df["_day"] = df["schedule"].apply(_parse_day_from_schedule)
    df["_time_slot"] = df["schedule"].apply(_parse_time_slot)

    if filters.get("search"):
        s = filters["search"]
        df = df[
            df["dept_code"].str.contains(s, case=False, na=False) |
            df["title"].str.contains(s, case=False, na=False) |
            df["faculty"].str.contains(s, case=False, na=False)
        ]

    if filters.get("quarters"):
        df = df[df["quarter"].isin(filters["quarters"])]

    if filters.get("days"):
        df = df[df["_day"].isin(filters["days"])]

    if filters.get("time_slots"):
        df = df[df["_time_slot"].isin(filters["time_slots"])]

    if filters.get("foundations") or filters.get("flmbe") or filters.get("concentrations"):
        df = _apply_requirement_filter(
            df,
            filters.get("foundations", []),
            filters.get("flmbe", []),
            filters.get("concentrations", []),
        )

    if filters.get("max_bid") is not None:
        df = df[df["display_price"].fillna(0) <= filters["max_bid"]]

    if filters.get("min_rating") is not None:
        df = df[df["display_rating"].fillna(0) >= filters["min_rating"]]

    # Sort
    sort_map = {
        "price_asc": ("display_price", True),
        "price_desc": ("display_price", False),
        "rating_desc": ("display_rating", False),
        "hours_asc": ("display_hours", True),
        "code": ("dept_code", True),
    }
    sort_key = filters.get("sort", "code")
    if sort_key in sort_map:
        col, asc = sort_map[sort_key]
        df = df.sort_values(col, ascending=asc, na_position="last")

    return df, filters


# ============================================================
# Build current-year section-level data
# ============================================================
def build_section_data(course_list, features, bid_df, eval_df):
    """
    Build section-level data for current-year courses.
    Each row = one section (course_code + instructor combination).
    Merges bid history and eval data per instructor.
    """
    cl = course_list.copy()

    # Merge course-level features (bid stats, eval stats by dept_code)
    cl = cl.merge(
        features[["dept_code", "p1_price_mean", "p1_price_max", "p1_price_median",
                   "p2_price_mean", "p2_price_max",
                   "eval_overall", "eval_clarity", "eval_interesting",
                   "eval_useful", "eval_got_out", "eval_recommend",
                   "eval_hours", "est_capacity"]],
        on="dept_code", how="left",
    )

    # Per-instructor bid stats: avg price for this instructor across all courses
    instr_bid = bid_df[bid_df["phase"] == "Phase 1"].copy()
    instr_bid_stats = instr_bid.groupby("instructor").agg(
        instr_p1_mean=("clearing_price", "mean"),
        instr_p1_max=("clearing_price", "max"),
    ).reset_index()
    cl = cl.merge(instr_bid_stats, on="instructor", how="left")

    # Per-instructor eval stats
    eval_df_copy = eval_df.copy()
    eval_df_copy["instructor"] = eval_df_copy["instructor_last"] + ", " + eval_df_copy["instructor_first"]
    instr_eval = eval_df_copy.groupby("instructor").agg(
        instr_rating=("overall_score", "mean"),
        instr_recommend=("recommend", "mean"),
        instr_hours=("hours_per_week", "mean"),
        instr_clarity=("clarity", "mean"),
        instr_interesting=("interesting", "mean"),
        instr_useful=("useful_tools", "mean"),
        instr_got_out=("how_much_got", "mean"),
    ).reset_index()
    cl = cl.merge(instr_eval, on="instructor", how="left")

    # Per course+instructor bid history
    ci_bid = bid_df[bid_df["phase"] == "Phase 1"].copy()
    ci_bid_stats = ci_bid.groupby(["dept_code", "instructor"]).agg(
        ci_p1_mean=("clearing_price", "mean"),
        ci_p1_max=("clearing_price", "max"),
    ).reset_index()
    cl = cl.merge(ci_bid_stats, on=["dept_code", "instructor"], how="left")

    # Deduplicate: merge sections with same dept_code + faculty + quarter
    # (e.g. 30000-01 and 30000-02 both taught by Nikolaev in Autumn)
    # Keep first schedule, combine section numbers
    def _agg_sections(group):
        row = group.iloc[0].copy()
        sections = group["section"].tolist()
        row["sections"] = ", ".join(sorted(sections))
        row["course_code"] = row["dept_code"] + "-" + "/".join(sorted(sections))
        schedules = group["schedule"].unique()
        row["schedule"] = " / ".join(schedules[:3])
        return row

    cl = cl.groupby(["dept_code", "faculty", "quarter"], group_keys=False).apply(_agg_sections).reset_index(drop=True)

    return cl


# ============================================================
# Main App
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="header-banner">
        <h1>Chicago Booth Course Recommender</h1>
        <p>AI-powered course selection, bid prediction & degree tracking · 2025–26 Academic Year</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    features, bid_df, eval_df, course_list = load_all_data()
    model_artifacts = load_model()

    # Build section-level data for current year
    section_data = build_section_data(course_list, features, bid_df, eval_df)

    # Top navigation (radio-based for stable tab persistence)
    NAV_OPTIONS = [
        "🔍 Course Explorer",
        "📊 Bid Predictor",
        "🎓 Degree Tracker",
        "🗓️ AI Course Planner",
    ]
    active_page = st.radio(
        "Navigation",
        NAV_OPTIONS,
        horizontal=True,
        label_visibility="collapsed",
        key="nav_page",
    )

    st.markdown("---")

    if active_page == NAV_OPTIONS[0]:
        page_explorer(features, bid_df, eval_df, section_data)
    elif active_page == NAV_OPTIONS[1]:
        page_bid_predictor(features, bid_df, eval_df, model_artifacts, section_data)
    elif active_page == NAV_OPTIONS[2]:
        page_degree_tracker(features, bid_df, eval_df, section_data)
    elif active_page == NAV_OPTIONS[3]:
        page_course_planner(features, bid_df, eval_df, section_data)

    # Footer
    st.markdown("---")
    st.markdown(
        f'<div style="text-align:center; padding:16px 0; color:{DARK_GRAY}; font-size:12px;">'
        f'Built with Claude Code & Streamlit  ·  Data: Autumn 2023 – Spring 2026  ·  '
        f'{len(features)} courses  ·  Chicago Booth MBA 2025–26'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Page: Course Explorer
# ============================================================
def _apply_requirement_filter(df, sel_foundation, sel_flmbe, sel_conc):
    """Apply Foundation / FLMBE / Concentration filters to a DataFrame with dept_code column."""
    matching_codes = set()
    has_filter = False

    if sel_foundation:
        has_filter = True
        for area_name in sel_foundation:
            area = DEGREE_REQUIREMENTS["foundations"]["areas"].get(area_name, {})
            matching_codes.update(area.get("basic", []))
            matching_codes.update(area.get("substitutes", []))

    if sel_flmbe:
        has_filter = True
        for area in sel_flmbe:
            matching_codes |= get_requirement_filter_codes(f"FLMBE: {area}")

    if sel_conc:
        has_filter = True
        conc_all = get_all_concentration_courses()
        for name in sel_conc:
            if name in conc_all:
                matching_codes |= conc_all[name]

    if has_filter:
        df = df[df["dept_code"].isin(matching_codes)]
    return df


def _parse_day_from_schedule(schedule):
    """Extract day of week from schedule string."""
    if pd.isna(schedule):
        return ""
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        if day in schedule:
            return day
    return ""


def _parse_time_slot(schedule):
    """Categorize schedule into time slot."""
    if pd.isna(schedule):
        return ""
    import re
    m = re.search(r'(\d{1,2}):(\d{2})\s*([APap][Mm])', str(schedule))
    if not m:
        return "Other"
    hour = int(m.group(1))
    ampm = m.group(3).upper()
    if ampm == "PM" and hour != 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0

    if hour < 12:
        return "Morning"
    elif hour < 17:
        return "Afternoon"
    elif hour < 18:
        return "Evening 1 (5 PM)"
    else:
        return "Evening 2 (6 PM)"


# Quarter sort order
QUARTER_SORT = {"Autumn 2025": 0, "Winter 2026": 1, "Spring 2026": 2, "Summer 2026": 3}


def page_explorer(features, bid_df, eval_df, section_data):
    st.header("Course Explorer")

    # --- AI-powered Natural Language Search (top) ---
    st.markdown(
        f'<div style="background:linear-gradient(135deg, {MAROON}11, {TEAL}18); border:1px solid {TEAL}55; '
        f'border-radius:10px; padding:16px 20px 8px 20px; margin-bottom:16px;">'
        f'<div style="font-size:14px; font-weight:700; color:{MAROON}; margin-bottom:6px;">✨ AI Search</div>'
        f'<div style="font-size:12px; color:{DARK_GRAY}; margin-bottom:8px;">'
        f'Describe what you\'re looking for in natural language — AI converts it to filters automatically.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    ai_query = st.text_area(
        "AI Search",
        placeholder='e.g. "Tuesday evening Finance courses with low bid" or "highly rated strategy courses in Spring"',
        height=80,
        label_visibility="collapsed",
        key="ai_search_query",
    )
    _, btn_col, clear_col, _ = st.columns([2, 1, 1, 2])
    with btn_col:
        ai_search_clicked = st.button("🔍 Search with AI", use_container_width=True)
    with clear_col:
        ai_clear_clicked = st.button("✕ Clear", use_container_width=True)

    # Persist AI results in session_state
    if ai_clear_clicked:
        st.session_state.pop("ai_results_df", None)
        st.session_state.pop("ai_filters", None)

    if ai_search_clicked and ai_query.strip():
        with st.spinner("Searching with AI..."):
            res, filt = llm_search(ai_query, section_data)
        if res is None:
            st.error(f"AI Search error: {filt}")
        elif isinstance(filt, dict):
            st.session_state["ai_results_df"] = res
            st.session_state["ai_filters"] = filt

    # Read from session_state
    ai_results = st.session_state.get("ai_results_df", None)
    ai_filters = st.session_state.get("ai_filters", None)

    if ai_results is not None and isinstance(ai_filters, dict):
        filter_tags = " · ".join(f"**{k}**: {v}" for k, v in ai_filters.items() if v)
        st.caption(f"AI filters active: {filter_tags}")

    # --- Manual Filters (collapsible) ---
    with st.expander("🔧 Manual Filters", expanded=not (ai_results is not None and isinstance(ai_filters, dict))):
        # View mode toggle
        view_mode = st.radio(
            "View by",
            ["By Section (per instructor)", "By Course (aggregated)"],
            horizontal=True,
            key="explorer_view_mode",
        )

        # Row 1: Search & Sort
        col1, col2 = st.columns([3, 2])
        with col1:
            search = st.text_input("Keyword search", placeholder="e.g. Finance, 35200, Pricing", key="explorer_search")
        with col2:
            sort_by = st.selectbox("Sort by", [
                "Course Code", "Bid Price (High→Low)", "Bid Price (Low→High)",
                "Rating (High→Low)", "Workload (Low→High)",
            ], key="explorer_sort")

        # Row 2: Requirements
        col_f, col_fl, col_c = st.columns(3)
        with col_f:
            foundation_areas = list(DEGREE_REQUIREMENTS["foundations"]["areas"].keys())
            sel_foundation = st.multiselect("Foundation", foundation_areas, placeholder="All", key="explorer_foundation")
        with col_fl:
            flmbe_areas = [area for cat in DEGREE_REQUIREMENTS["flmbe"]["categories"].values() for area in cat]
            sel_flmbe = st.multiselect("FLMBE", flmbe_areas, placeholder="All", key="explorer_flmbe")
        with col_c:
            sel_conc = st.multiselect("Concentration", sorted(CONCENTRATIONS.keys()), placeholder="All", key="explorer_concentration")

        # Row 3: Schedule
        col_q, col_day, col_time = st.columns(3)
        with col_q:
            quarter_options = sorted(section_data["quarter"].unique(), key=lambda q: QUARTER_SORT.get(q, 99))
            selected_quarters = st.multiselect("Quarter", quarter_options, placeholder="All quarters", key="explorer_quarters")
        with col_day:
            day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            sel_days = st.multiselect("Day", day_options, placeholder="All days", key="explorer_days")
        with col_time:
            time_options = ["Morning", "Afternoon", "Evening 1 (5 PM)", "Evening 2 (6 PM)", "Other"]
            sel_times = st.multiselect("Time", time_options, placeholder="All times", key="explorer_times")

    # ---- Build filtered data ----
    if view_mode == "By Section (per instructor)":
        # If AI search returned results, use them directly
        if ai_results is not None and isinstance(ai_filters, dict):
            df = ai_results.copy()
            if "display_price" not in df.columns:
                df["display_price"] = df["ci_p1_mean"].fillna(df["p1_price_mean"]) if "ci_p1_mean" in df.columns else df.get("p1_price_mean", 0)
            if "display_rating" not in df.columns:
                df["display_rating"] = df["instr_rating"].fillna(df["eval_overall"]) if "instr_rating" in df.columns else df.get("eval_overall", np.nan)
            if "display_hours" not in df.columns:
                df["display_hours"] = df["instr_hours"].fillna(df["eval_hours"]) if "instr_hours" in df.columns else df.get("eval_hours", np.nan)
            if "display_recommend" not in df.columns:
                df["display_recommend"] = df["instr_recommend"].fillna(df["eval_recommend"]) if "instr_recommend" in df.columns else df.get("eval_recommend", np.nan)
        else:
            df = section_data.copy()

            # Use ci_p1_mean (course+instructor specific) as primary price
            df["display_price"] = df["ci_p1_mean"].fillna(df["p1_price_mean"])
            df["display_rating"] = df["instr_rating"].fillna(df["eval_overall"])
            df["display_hours"] = df["instr_hours"].fillna(df["eval_hours"])
            df["display_recommend"] = df["instr_recommend"].fillna(df["eval_recommend"])

            # Parse schedule for filtering
            df["_day"] = df["schedule"].apply(_parse_day_from_schedule)
            df["_time_slot"] = df["schedule"].apply(_parse_time_slot)

        if search and not (ai_results is not None and isinstance(ai_filters, dict)):
            mask = (
                df["dept_code"].str.contains(search, case=False, na=False) |
                df["title"].str.contains(search, case=False, na=False) |
                df["faculty"].str.contains(search, case=False, na=False)
            )
            df = df[mask]

        if not (ai_results is not None and isinstance(ai_filters, dict)):
            if selected_quarters:
                df = df[df["quarter"].isin(selected_quarters)]

            df = _apply_requirement_filter(df, sel_foundation, sel_flmbe, sel_conc)

            if sel_days:
                df = df[df["_day"].isin(sel_days)]

            if sel_times:
                df = df[df["_time_slot"].isin(sel_times)]

        # Sort
        sort_map = {
            "Course Code": ("dept_code", True),
            "Bid Price (High→Low)": ("display_price", False),
            "Bid Price (Low→High)": ("display_price", True),
            "Rating (High→Low)": ("display_rating", False),
            "Workload (Low→High)": ("display_hours", True),
        }
        sort_col, sort_asc = sort_map[sort_by]
        df = df.sort_values(sort_col, ascending=sort_asc, na_position="last")

        # Summary
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sections Found", len(df))
        m2.metric("Avg Bid (P1)", format_price(df["display_price"].mean()))
        m3.metric("Avg Rating", f"{df['display_rating'].mean():.2f}/5" if df["display_rating"].notna().any() else "N/A")
        m4.metric("Avg Hours/Week", f"{df['display_hours'].mean():.2f}" if df["display_hours"].notna().any() else "N/A")

        # Section list — card UI
        st.markdown("---")
        if len(df) > 50:
            st.caption(f"Showing top 50 of {len(df)} sections. Refine filters to narrow results.")

        for _, row in df.head(50).iterrows():
            price = row.get("display_price", 0)
            if pd.isna(price):
                price = 0
            level, css_class = get_bid_level(price)
            rating = row.get("display_rating", np.nan)
            hours = row.get("display_hours", np.nan)

            level_colors = {
                "Free": "#2ecc71", "Low": TEAL,
                "Moderate": "#f39c12", "High": "#e74c3c", "Very High": MAROON,
            }
            level_color = level_colors.get(level, DARK_GRAY)

            req_info = check_course_requirements(row["dept_code"])
            req_tags = _build_req_tags(req_info)

            with st.container(border=True):
                # Row 1: Title + metrics
                c_info, c_price, c_demand, c_rating, c_hours = st.columns([3.5, 0.8, 0.7, 0.5, 0.5])
                with c_info:
                    title_text = row.get("title", "") if pd.notna(row.get("title", "")) else ""
                    st.markdown(f'<div style="font-size:15px; font-weight:bold; color:{MAROON};">{row.get("course_code", "")} — {title_text}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:13px; color:{DARK_GRAY}; margin-top:-8px;">{row.get("faculty", "")}  ·  {row.get("quarter", "")}  ·  {row.get("schedule", "")}</div>', unsafe_allow_html=True)
                with c_price:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:18px; font-weight:bold; color:{MAROON};">{format_price(price)}</div><div style="font-size:11px; color:{DARK_GRAY};">Bid P1</div></div>', unsafe_allow_html=True)
                with c_demand:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:14px; font-weight:bold; color:{level_color};">{level}</div><div style="font-size:11px; color:{DARK_GRAY};">Demand</div></div>', unsafe_allow_html=True)
                with c_rating:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:14px;">{f"{rating:.2f}" if pd.notna(rating) else "—"}</div><div style="font-size:11px; color:{DARK_GRAY};">Rating</div></div>', unsafe_allow_html=True)
                with c_hours:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:14px;">{f"{hours:.1f}" if pd.notna(hours) else "—"}</div><div style="font-size:11px; color:{DARK_GRAY};">Hrs/Wk</div></div>', unsafe_allow_html=True)
                # Row 2: Req tags (left) + Price History expander label (right)
                c_tags, c_spacer = st.columns([5, 1])
                with c_tags:
                    st.markdown(f'<div style="margin-top:-8px; margin-bottom:8px;">{req_tags}</div>', unsafe_allow_html=True)
                # Price History + Instructor Radar inside the card
                with st.expander("📈 Price History & Instructor Rating", expanded=False):
                    chart_col, radar_col = st.columns(2)
                    with chart_col:
                        course_bid = bid_df[
                            (bid_df["dept_code"] == row["dept_code"]) &
                            (bid_df["instructor"] == row["instructor"]) &
                            (bid_df["phase"].isin(["Phase 1", "Phase 2"]))
                        ].copy()
                        if len(course_bid) == 0:
                            course_bid = bid_df[
                                (bid_df["dept_code"] == row["dept_code"]) &
                                (bid_df["phase"].isin(["Phase 1", "Phase 2"]))
                            ].copy()
                        if len(course_bid) > 0:
                            _render_price_bar_chart(course_bid, key=f"expl_{row['course_code']}_{row['quarter']}")
                        else:
                            st.caption("No bid history available.")
                    with radar_col:
                        faculty_name = row.get("instructor", row.get("faculty", ""))
                        if not _render_instructor_radar(eval_df, faculty_name, key=f"radar_{row['course_code']}_{row['quarter']}"):
                            st.caption("No evaluation data for this instructor.")

    else:
        # ---- Aggregated by course ----
        # Filter section_data first for quarter/day/time, then get unique dept_codes
        sd = section_data.copy()
        sd["_day"] = sd["schedule"].apply(_parse_day_from_schedule)
        sd["_time_slot"] = sd["schedule"].apply(_parse_time_slot)

        if selected_quarters:
            sd = sd[sd["quarter"].isin(selected_quarters)]
        if sel_days:
            sd = sd[sd["_day"].isin(sel_days)]
        if sel_times:
            sd = sd[sd["_time_slot"].isin(sel_times)]

        current_codes = sd["dept_code"].unique()
        df = features[features["dept_code"].isin(current_codes)].copy()

        if search:
            mask = (
                df["dept_code"].str.contains(search, case=False, na=False) |
                df["title"].astype(str).str.contains(search, case=False, na=False)
            )
            df = df[mask]

        df = _apply_requirement_filter(df, sel_foundation, sel_flmbe, sel_conc)

        sort_map = {
            "Course Code": ("dept_code", True),
            "Bid Price (High→Low)": ("p1_price_mean", False),
            "Bid Price (Low→High)": ("p1_price_mean", True),
            "Rating (High→Low)": ("eval_overall", False),
            "Workload (Low→High)": ("eval_hours", True),
        }
        sort_col, sort_asc = sort_map[sort_by]
        df = df.sort_values(sort_col, ascending=sort_asc, na_position="last")

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Courses Found", len(df))
        m2.metric("Avg Bid (P1)", format_price(df["p1_price_mean"].mean()))
        m3.metric("Avg Rating", f"{df['eval_overall'].mean():.2f}/5" if df["eval_overall"].notna().any() else "N/A")
        m4.metric("Avg Hours/Week", f"{df['eval_hours'].mean():.2f}" if df["eval_hours"].notna().any() else "N/A")

        # Course list — card UI
        st.markdown("---")
        if len(df) > 50:
            st.caption(f"Showing top 50 of {len(df)} courses. Refine filters to narrow results.")

        for _, row in df.head(50).iterrows():
            p1_price = row.get("p1_price_mean", 0)
            if pd.isna(p1_price):
                p1_price = 0
            level, css_class = get_bid_level(p1_price)
            rating = row.get("eval_overall", np.nan)
            hours = row.get("eval_hours", np.nan)

            instructors = section_data[section_data["dept_code"] == row["dept_code"]]["faculty"].unique()
            instr_str = ", ".join(instructors[:3])
            if len(instructors) > 3:
                instr_str += f" +{len(instructors)-3}"

            level_colors = {
                "Free": "#2ecc71", "Low": TEAL,
                "Moderate": "#f39c12", "High": "#e74c3c", "Very High": MAROON,
            }
            level_color = level_colors.get(level, DARK_GRAY)

            req_info = check_course_requirements(row["dept_code"])
            req_tags = _build_req_tags(req_info)

            with st.container(border=True):
                c_info, c_price, c_demand, c_rating, c_hours = st.columns([3.5, 0.8, 0.7, 0.5, 0.5])
                with c_info:
                    title_text = row["title"] if pd.notna(row["title"]) else ""
                    st.markdown(f'<div style="font-size:15px; font-weight:bold; color:{MAROON};">{row["dept_code"]} — {title_text}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:13px; color:{DARK_GRAY}; margin-top:-8px;">{instr_str}</div>', unsafe_allow_html=True)
                with c_price:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:18px; font-weight:bold; color:{MAROON};">{format_price(p1_price)}</div><div style="font-size:11px; color:{DARK_GRAY};">Bid P1</div></div>', unsafe_allow_html=True)
                with c_demand:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:14px; font-weight:bold; color:{level_color};">{level}</div><div style="font-size:11px; color:{DARK_GRAY};">Demand</div></div>', unsafe_allow_html=True)
                with c_rating:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:14px;">{f"{rating:.2f}" if pd.notna(rating) else "—"}</div><div style="font-size:11px; color:{DARK_GRAY};">Rating</div></div>', unsafe_allow_html=True)
                with c_hours:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:14px;">{f"{hours:.1f}" if pd.notna(hours) else "—"}</div><div style="font-size:11px; color:{DARK_GRAY};">Hrs/Wk</div></div>', unsafe_allow_html=True)
                c_tags, c_spacer = st.columns([5, 1])
                with c_tags:
                    st.markdown(f'<div style="margin-top:-8px; margin-bottom:8px;">{req_tags}</div>', unsafe_allow_html=True)
                with st.expander("📈 Price History & Instructor Rating", expanded=False):
                    chart_col, radar_col = st.columns(2)
                    with chart_col:
                        course_bid = bid_df[
                            (bid_df["dept_code"] == row["dept_code"]) &
                            (bid_df["phase"].isin(["Phase 1", "Phase 2"]))
                        ].copy()
                        if len(course_bid) > 0:
                            _render_price_bar_chart(course_bid, key=f"expl_agg_{row['dept_code']}")
                        else:
                            st.caption("No bid history available.")
                    with radar_col:
                        # Use first instructor for aggregated view
                        first_instr = instructors[0] if len(instructors) > 0 else ""
                        instr_key = first_instr.split(",")[0].strip() if "," in str(first_instr) else str(first_instr)
                        if not _render_instructor_radar(eval_df, first_instr, key=f"radar_agg_{row['dept_code']}"):
                            st.caption("No evaluation data available.")


def _render_instructor_radar(eval_df, instructor_name, key):
    """Render a radar chart for instructor evaluation scores."""
    # Match instructor by last name (faculty format: "Last, First")
    last_name = instructor_name.split(",")[0].strip() if "," in str(instructor_name) else str(instructor_name).strip()
    instr_data = eval_df[eval_df["instructor_last"].str.strip() == last_name]
    if instr_data.empty:
        return False

    # Recommend at top, then clockwise
    categories = ["Recommend", "Clarity", "Interesting", "Useful Tools", "How Much Got"]
    cols = ["recommend", "clarity", "interesting", "useful_tools", "how_much_got"]
    values = [instr_data[c].mean() for c in cols]
    if all(pd.isna(v) for v in values):
        return False
    values = [v if pd.notna(v) else 0 for v in values]

    # Course avg for comparison
    avg_values = [eval_df[c].mean() for c in cols]

    display_name = f"Prof. {last_name}"

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(128, 0, 0, 0.15)",
        line=dict(color=MAROON, width=2),
        name=display_name,
    ))
    fig.add_trace(go.Scatterpolar(
        r=avg_values + [avg_values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(126, 190, 197, 0.1)",
        line=dict(color=TEAL, width=1, dash="dot"),
        name="School Avg",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[1, 5], tickvals=[2, 3, 4, 5]),
            angularaxis=dict(rotation=90, direction="clockwise"),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        height=300,
        margin=dict(l=40, r=40, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
    return True


def _render_price_bar_chart(course_bid, key):
    """Render a Phase 1/2 bar chart from bid data."""
    course_bid["term"] = course_bid["quarter"] + " " + course_bid["year"].astype(str)
    course_bid = course_bid.sort_values("term_order")
    avg_by_term = course_bid.groupby(["term", "term_order", "phase"])["clearing_price"].mean().reset_index()
    avg_by_term = avg_by_term.sort_values("term_order")

    n_terms = avg_by_term["term"].nunique()

    fig = go.Figure()
    for phase, color in [("Phase 1", MAROON), ("Phase 2", TEAL)]:
        phase_data = avg_by_term[avg_by_term["phase"] == phase]
        fig.add_trace(go.Bar(
            x=phase_data["term"],
            y=phase_data["clearing_price"],
            name=phase,
            marker_color=color,
            text=phase_data["clearing_price"].apply(
                lambda v: f"{int(v):,}" if pd.notna(v) and v > 0 else "Free"
            ),
            textposition="outside",
            textfont_size=10,
            width=max(0.25, min(0.4, 0.4 * n_terms / 8)),
        ))

    max_price = avg_by_term["clearing_price"].max()
    y_max = max_price * 1.3 if pd.notna(max_price) and max_price > 0 else 100

    fig.update_layout(
        barmode="group",
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white",
        yaxis_title="Clearing Price (pts)",
        yaxis_range=[0, y_max],
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="right", x=1),
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ============================================================
# Page: Bid Predictor
# ============================================================
def page_bid_predictor(features, bid_df, eval_df, model_artifacts, section_data):
    st.header("Bid Price Predictor")

    if model_artifacts is None:
        st.error("Model not found. Please run bid_model_v2.py first.")
        return

    st.markdown("Select courses to see predicted bid prices and bidding strategy recommendations.")

    # Build options from section_data (current-year only, with instructor)
    # Merge same dept_code + faculty across quarters
    options_df = section_data.sort_values("quarter")
    options_df = options_df.groupby(["dept_code", "faculty"], as_index=False).agg(
        course_code=("course_code", "first"),
        title=("title", "first"),
        quarters=("quarter", lambda x: ", ".join(sorted(x.unique()))),
        ci_p1_mean=("ci_p1_mean", "first"),
        p1_price_mean=("p1_price_mean", "first"),
        p2_price_mean=("p2_price_mean", "first"),
        instr_rating=("instr_rating", "first"),
        eval_overall=("eval_overall", "first"),
        instr_hours=("instr_hours", "first"),
        eval_hours=("eval_hours", "first"),
    ).sort_values("dept_code")
    course_options = {
        f"{row['dept_code']} - {row['title']} - {row['faculty']} ({row['quarters']})": row
        for _, row in options_df.iterrows()
    }

    selected = st.multiselect(
        "Select sections to predict",
        options=list(course_options.keys()),
        max_selections=10,
        key="bid_sections",
    )

    # Methodology expander (always shown at top)
    with st.expander("About the Prediction Model"):
        st.markdown(f"""
**Model**: Two-stage Random Forest (scikit-learn)

**Stage 1 — Demand Classifier**: Predicts whether a course will have non-zero bidding demand.
A Random Forest classifier trained on historical bid data achieves ~96% accuracy.

**Stage 2 — Price Regressor**: For courses with expected demand, predicts the clearing price
using a Random Forest on log-transformed prices (right-skewed distribution).

**Key Features** (by importance):
1. **Phase 1 price** — strongest predictor for Phase 2
2. **Instructor historical max price** — captures instructor popularity
3. **Historical price mean / trend** — course-level demand patterns
4. **Fill ratio & capacity** — supply-demand dynamics
5. **Evaluation scores** — course quality signals
6. **Time/schedule features** — day of week, evening, quarter

**Training Data**: {len(bid_df):,} bid records across 11 quarters (Autumn 2023 – Spring 2026),
combined with {len(eval_df):,} course evaluation records.

**Why Random Forest?**
- Captures non-linear relationships (e.g., popular instructor + small class → price spike)
- Robust to outliers and right-skewed price distributions
- Provides uncertainty estimates via individual tree predictions
- Handles missing data and mixed feature types naturally

**Model Performance** (5-fold time-series cross-validation):
- Stage 1 Classifier Accuracy: ~96%
- Stage 2 Price Regressor MAE: ~650 pts (original scale)
- Stage 2 R² (train): ~0.95
        """)

    if not selected:
        # Default: top bid courses by instructor (current year only)
        st.markdown("### Most Expensive Sections — 2025-26")

        # Build instructor-level ranking from section_data
        # Aggregate across quarters: same dept_code + faculty = one entry
        top_sections = section_data.copy()
        top_sections["display_p1"] = top_sections["ci_p1_mean"].fillna(top_sections["p1_price_mean"])
        top_sections = top_sections.dropna(subset=["display_p1"])
        top_sections = top_sections[top_sections["display_p1"] > 0]
        # Group by dept_code + faculty to merge quarters
        top_sections = top_sections.sort_values("quarter")
        top_agg = top_sections.groupby(["dept_code", "faculty"], as_index=False).agg(
            title=("title", "first"),
            course_code=("course_code", "first"),
            display_p1=("display_p1", "first"),
            p2_price_mean=("p2_price_mean", "first"),
            instr_rating=("instr_rating", "first"),
            eval_overall=("eval_overall", "first"),
            instr_hours=("instr_hours", "first"),
            eval_hours=("eval_hours", "first"),
            quarters=("quarter", lambda x: ", ".join(sorted(x.unique()))),
        )
        top_agg = top_agg.nlargest(20, "display_p1")

        for idx, (_, row) in enumerate(top_agg.iterrows()):
            p1 = row["display_p1"]
            p2 = row.get("p2_price_mean", 0)
            if pd.isna(p2):
                p2 = 0
            rating = row.get("instr_rating", row.get("eval_overall", np.nan))
            hours = row.get("instr_hours", row.get("eval_hours", np.nan))
            level, css_class = get_bid_level(p1)

            level_colors = {
                "Free": "#2ecc71", "Low": TEAL,
                "Moderate": "#f39c12", "High": "#e74c3c", "Very High": MAROON,
            }
            level_color = level_colors.get(level, DARK_GRAY)

            with st.container(border=True):
                c_info, c_p1, c_p2, c_demand, c_rating = st.columns([3, 1, 1, 0.7, 0.7])
                with c_info:
                    title_text = row['title'] if pd.notna(row['title']) else ''
                    st.markdown(f'<div style="font-size:16px; font-weight:bold; color:{MAROON};">{row["dept_code"]} — {title_text}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:13px; color:{DARK_GRAY}; margin-top:-8px;">{row["faculty"]}  ·  {row["quarters"]}</div>', unsafe_allow_html=True)
                with c_p1:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:22px; font-weight:bold; color:{MAROON};">{format_price(p1)}</div><div style="font-size:11px; color:{DARK_GRAY};">Phase 1 Avg</div></div>', unsafe_allow_html=True)
                with c_p2:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:16px; color:{TEAL};">{format_price(p2)}</div><div style="font-size:11px; color:{DARK_GRAY};">Phase 2 Avg</div></div>', unsafe_allow_html=True)
                with c_demand:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:14px; font-weight:bold; color:{level_color};">{level}</div><div style="font-size:11px; color:{DARK_GRAY};">Demand</div></div>', unsafe_allow_html=True)
                with c_rating:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:14px;">{f"{rating:.2f}" if pd.notna(rating) else "—"}</div><div style="font-size:11px; color:{DARK_GRAY};">Rating</div></div>', unsafe_allow_html=True)

        return

    # Show predictions using the trained two-stage model
    st.markdown("---")

    total_budget = 0
    has_model = model_artifacts is not None and "clf_p1" in model_artifacts

    for course_label in selected:
        row_info = course_options[course_label]
        dept = row_info["dept_code"]
        feat_row = features[features["dept_code"] == dept]
        if len(feat_row) == 0:
            continue
        feat_row = feat_row.iloc[0]

        # Use instructor-specific price if available
        ci_price = row_info.get("ci_p1_mean", np.nan)
        p1_hist = ci_price if pd.notna(ci_price) else (feat_row.get("p1_price_mean", 0) or 0)
        p2_hist = feat_row.get("p2_price_mean", 0) or 0
        if pd.isna(p1_hist):
            p1_hist = 0
        if pd.isna(p2_hist):
            p2_hist = 0

        # Build feature dict for model prediction
        pred_result = None
        if has_model:
            # Map features DataFrame columns to model feature names
            _p1_mean = feat_row.get("p1_price_mean", 0) or 0
            _p1_max = feat_row.get("p1_price_max", 0) or 0
            _p1_std = feat_row.get("p1_price_std", 0) or 0
            _p1_count = feat_row.get("p1_count", 3) or 3
            _fill = feat_row.get("p1_fill_mean", 0.8) or 0.8
            _cap = feat_row.get("est_capacity", 50) or 50
            _ci_mean = row_info.get("ci_p1_mean", np.nan)
            _instr_avg = _ci_mean if pd.notna(_ci_mean) else _p1_mean
            _instr_max = row_info.get("ci_p1_max", _p1_max) if pd.notna(row_info.get("ci_p1_max", np.nan)) else _p1_max

            course_feats = {
                "hist_price_mean": _p1_mean,
                "hist_price_max": _p1_max,
                "hist_price_std": _p1_std,
                "hist_price_last": p1_hist,
                "hist_price_last2": _p1_mean,
                "hist_count": _p1_count,
                "price_trend": 0,
                "est_capacity": _cap,
                "fill_ratio": _fill,
                "is_evening": 0,
                "is_weekend": 0,
                "day_monday": 0, "day_tuesday": 0, "day_wednesday": 0,
                "day_thursday": 0, "day_friday": 0,
                "eval_clarity": feat_row.get("eval_clarity", np.nan),
                "eval_interesting": feat_row.get("eval_interesting", np.nan),
                "eval_useful": feat_row.get("eval_useful", np.nan),
                "eval_got_out": feat_row.get("eval_got_out", np.nan),
                "eval_recommend": feat_row.get("eval_recommend", np.nan),
                "eval_hours": feat_row.get("eval_hours", np.nan),
                "eval_count": feat_row.get("eval_count", 1) or 1,
                "instructor_avg_price": _instr_avg,
                "instructor_max_price": _instr_max,
                "instructor_course_count": _p1_count,
                "quarter_sin": 0, "quarter_cos": 1, "year_norm": 1.0,
                "dept_frequency": _p1_count,
                "p1_price": p1_hist,
            }
            try:
                pred_result = predict_two_stage(
                    model_artifacts["clf_p1"], model_artifacts["reg_p1"],
                    model_artifacts["features_p1"], course_feats
                )
            except Exception:
                pred_result = None

        # Use model prediction if available, else fall back to historical
        if pred_result:
            pred_price = pred_result["predicted_price_if_nonzero"]
            prob_nz = pred_result["prob_nonzero"]
            p25 = pred_result["price_p25"]
            p75 = pred_result["price_p75"]

            # Override: if historical data shows this course has had non-zero bids,
            # trust the history over the classifier's prob_nonzero
            hist_closed_rate = feat_row.get("p1_closed_rate", 0) or 0
            if p1_hist > 0 and (hist_closed_rate > 0.3 or prob_nz < 0.5):
                prob_nz = max(prob_nz, hist_closed_rate, 0.7)
                # Also floor the predicted price at historical mean
                pred_price = max(pred_price, p1_hist * 0.8)

            # Regenerate strategy with corrected values
            if prob_nz < 0.3:
                strategy = "FREE — Likely no bidding needed. Save your points."
                rec_bid = 0
            elif prob_nz < 0.6:
                strategy = f"LOW DEMAND — Consider minimal bid (~{int(pred_price * 0.3):,} pts)."
                rec_bid = int(pred_price * 0.3)
            elif pred_price < 2000:
                strategy = f"MODERATE — Bid around {int(pred_price * 1.1):,} pts to be safe."
                rec_bid = int(pred_price * 1.1)
            elif pred_price < 8000:
                strategy = f"HIGH DEMAND — Budget {int(pred_price * 1.2):,}+ pts. Popular course."
                rec_bid = int(pred_price * 1.2)
            else:
                strategy = f"VERY HIGH DEMAND — Premium course. Budget {int(pred_price * 1.3):,}+ pts or try alternate section/quarter."
                rec_bid = int(pred_price * 1.3)
        else:
            pred_price = p1_hist
            prob_nz = 1.0 if p1_hist > 0 else 0.0
            p25 = p1_hist * 0.7
            p75 = p1_hist * 1.3
            if p1_hist == 0:
                strategy = "No bid needed — course is typically free."
                rec_bid = 0
            elif p1_hist < 2000:
                strategy = f"Low demand. Bid ~{int(p1_hist * 1.2):,} pts to be safe."
                rec_bid = int(p1_hist * 1.2)
            else:
                strategy = f"Budget {int(p1_hist * 1.3):,}+ pts."
                rec_bid = int(p1_hist * 1.3)

        total_budget += rec_bid

        level, css_class = get_bid_level(pred_price)
        level_colors = {
            "Free": "#2ecc71", "Low": TEAL,
            "Moderate": "#f39c12", "High": "#e74c3c", "Very High": MAROON,
        }
        level_color = level_colors.get(level, DARK_GRAY)

        with st.container(border=True):
            c_info, c_pred, c_range, c_prob, c_demand = st.columns([3, 1, 1, 0.7, 0.7])
            with c_info:
                st.markdown(f'<div style="font-size:16px; font-weight:bold; color:{MAROON};">{row_info["dept_code"]} — {row_info["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:13px; color:{DARK_GRAY}; margin-top:-8px;">{row_info["faculty"]}  ·  {row_info["quarters"]}</div>', unsafe_allow_html=True)
            with c_pred:
                st.markdown(f'<div style="text-align:center;"><div style="font-size:22px; font-weight:bold; color:{MAROON};">{format_price(pred_price)}</div><div style="font-size:11px; color:{DARK_GRAY};">{"Predicted" if pred_result else "Hist. Avg"}</div></div>', unsafe_allow_html=True)
            with c_range:
                st.markdown(f'<div style="text-align:center;"><div style="font-size:13px; color:{DARK_GRAY};">{format_price(p25)} – {format_price(p75)}</div><div style="font-size:11px; color:{DARK_GRAY};">P25–P75</div></div>', unsafe_allow_html=True)
            with c_prob:
                prob_color = "#2ecc71" if prob_nz > 0.8 else "#f39c12" if prob_nz > 0.4 else TEAL
                st.markdown(f'<div style="text-align:center;"><div style="font-size:14px; font-weight:bold; color:{prob_color};">{prob_nz:.0%}</div><div style="font-size:11px; color:{DARK_GRAY};">P(Demand)</div></div>', unsafe_allow_html=True)
            with c_demand:
                st.markdown(f'<div style="text-align:center;"><div style="font-size:14px; font-weight:bold; color:{level_color};">{level}</div><div style="font-size:11px; color:{DARK_GRAY};">Demand</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:13px; color:{DARK_GRAY}; margin-top:-4px;">💡 <b>Strategy:</b> {strategy}</div>', unsafe_allow_html=True)

    st.markdown(f"### Total Recommended Budget: **{total_budget:,} pts**")

    st.info(
        "Full-Time MBA students receive 8,000 Bid Wealth points upon enrollment. "
        "After each quarter, points are replenished at 2,000 points per course taken."
    )

    # ============================================================
    # Bid Allocation Optimizer
    # ============================================================
    st.markdown("---")
    st.markdown("### Bid Allocation Optimizer")
    st.caption("Optimize how to distribute your bid points across selected courses to maximize enrollment probability.")

    budget = st.number_input(
        "Total Bid Budget (pts)",
        min_value=0, max_value=100000, value=8000, step=1000,
        help="Enter your available bid points for this phase.",
        key="bid_budget",
    )

    # Build course info for optimizer
    opt_courses = []
    for course_label in selected:
        row_info = course_options[course_label]
        dept = row_info["dept_code"]
        ci_price = row_info.get("ci_p1_mean", np.nan)
        feat_row = features[features["dept_code"] == dept]
        if len(feat_row) == 0:
            continue
        feat_row = feat_row.iloc[0]
        p1_mean = ci_price if pd.notna(ci_price) else (feat_row.get("p1_price_mean", 0) or 0)
        if pd.isna(p1_mean):
            p1_mean = 0

        # Get historical price distribution for this course
        hist_prices = bid_df[
            (bid_df["dept_code"] == dept) &
            (bid_df["phase"] == "Phase 1") &
            (bid_df["clearing_price"].notna()) &
            (bid_df["clearing_price"] >= 0)
        ]["clearing_price"].values

        p1_std = feat_row.get("p1_price_std", 0) or 0
        if pd.isna(p1_std):
            p1_std = max(p1_mean * 0.3, 1)

        # Smart upper bound: no need to bid more than historical max * 1.2
        if len(hist_prices) > 0 and max(hist_prices) > 0:
            bid_cap = int(max(hist_prices) * 1.2)
        elif p1_mean > 0:
            bid_cap = int(p1_mean * 2)
        else:
            bid_cap = 0  # Free course — no bid needed

        opt_courses.append({
            "label": f"{dept} — {row_info['title']}",
            "dept_code": dept,
            "faculty": row_info["faculty"],
            "p1_mean": p1_mean,
            "p1_std": max(p1_std, 1),
            "hist_prices": hist_prices,
            "bid_cap": bid_cap,
        })

    if len(opt_courses) >= 2 and budget > 0:
        from scipy.stats import norm as _norm

        def win_probability(bid, course):
            """Estimate probability of winning given a bid amount."""
            if course["p1_mean"] == 0:
                return 1.0
            hist = course["hist_prices"]
            if len(hist) >= 5:
                # Empirical CDF with smoothing
                return np.clip(np.mean(hist <= bid), 0.01, 0.99)
            else:
                return np.clip(
                    _norm.cdf(bid, loc=course["p1_mean"], scale=course["p1_std"]),
                    0.01, 0.99,
                )

        def neg_total_prob(alloc):
            """Negative sum of win probabilities (to minimize)."""
            return -sum(win_probability(alloc[i], c) for i, c in enumerate(opt_courses))

        n = len(opt_courses)
        # Per-course bounds: [0, min(bid_cap, budget)]
        bounds = [(0, min(c["bid_cap"], budget)) for c in opt_courses]
        # Use inequality: total <= budget (allows saving unneeded points)
        constraint = {"type": "ineq", "fun": lambda x: budget - np.sum(x)}

        # Smart initial guess: proportional to avg price
        total_mean = sum(c["p1_mean"] for c in opt_courses)
        if total_mean > 0:
            x0 = np.array([min(c["p1_mean"] / total_mean * budget, c["bid_cap"])
                           for c in opt_courses])
        else:
            x0 = np.zeros(n)

        result = minimize(neg_total_prob, x0, method="SLSQP",
                          bounds=bounds, constraints=constraint,
                          options={"maxiter": 500, "ftol": 1e-10})

        opt_alloc = result.x if result.success else x0

        # Also compute naive (equal split) allocation
        naive_allocs = np.array([min(int(budget / n), c["bid_cap"]) for c in opt_courses], dtype=float)
        opt_total_prob = -neg_total_prob(opt_alloc)
        naive_total_prob = sum(win_probability(naive_allocs[i], c) for i, c in enumerate(opt_courses))

        # Use whichever allocation is better
        if opt_total_prob >= naive_total_prob:
            alloc = opt_alloc
        else:
            alloc = naive_allocs

        # Round and build results
        results = []
        for i, course in enumerate(opt_courses):
            bid_amt = max(0, int(round(alloc[i])))
            prob = win_probability(bid_amt, course)
            naive_bid = int(naive_allocs[i])
            naive_prob = win_probability(naive_bid, course)
            results.append({
                "label": course["label"],
                "avg_price": course["p1_mean"],
                "bid_cap": course["bid_cap"],
                "optimal_bid": bid_amt,
                "win_prob": prob,
                "naive_bid": naive_bid,
                "naive_prob": naive_prob,
            })

        # Summary metrics
        total_alloc = sum(r["optimal_bid"] for r in results)
        avg_prob = np.mean([r["win_prob"] for r in results])
        naive_avg = np.mean([r["naive_prob"] for r in results])

        m1, m2, m3 = st.columns(3)
        delta_val = avg_prob - naive_avg
        delta_str = f"{delta_val:+.0%} vs equal split" if abs(delta_val) >= 0.005 else "same as equal split"
        m1.metric("Avg Win Probability", f"{avg_prob:.0%}", delta=delta_str)
        m2.metric("Budget Used", f"{total_alloc:,} / {budget:,} pts")
        saved = budget - total_alloc
        m3.metric("Points Saved", f"{saved:,} pts" if saved > 0 else "—")

        # Card UI for each course
        for r in results:
            prob_color = "#2ecc71" if r["win_prob"] >= 0.8 else "#f39c12" if r["win_prob"] >= 0.5 else "#e74c3c"
            diff = r["win_prob"] - r["naive_prob"]
            diff_str = f"{diff:+.0%}" if diff != 0 else "same"
            bar_pct = min(r["win_prob"] * 100, 100)

            with st.container(border=True):
                c_info, c_bid, c_prob = st.columns([3, 1, 1])
                with c_info:
                    st.markdown(f'<div style="font-size:14px; font-weight:bold; color:{MAROON};">{r["label"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:12px; color:{DARK_GRAY}; margin-top:-8px;">Hist. avg: {format_price(r["avg_price"])}  ·  Cap: {r["bid_cap"]:,} pts</div>', unsafe_allow_html=True)
                with c_bid:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:20px; font-weight:bold; color:{MAROON};">{r["optimal_bid"]:,}</div><div style="font-size:11px; color:{DARK_GRAY};">Optimal Bid</div></div>', unsafe_allow_html=True)
                with c_prob:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:20px; font-weight:bold; color:{prob_color};">{r["win_prob"]:.0%}</div><div style="font-size:11px; color:{DARK_GRAY};">Win Prob ({diff_str})</div></div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:{LIGHT_GRAY}; border-radius:4px; height:6px; margin-top:-4px;">'
                    f'<div style="background:{prob_color}; border-radius:4px; height:6px; width:{bar_pct:.0f}%;"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("How It Works"):
            st.markdown(
                "The optimizer uses **scipy SLSQP** (Sequential Least Squares Programming) "
                "to maximize the **total win probability** across all selected courses.\n\n"
                "- **Win probability** is estimated from the empirical distribution of historical clearing prices "
                "(or a normal approximation when data is sparse).\n"
                "- **Bid caps** prevent over-allocating to courses that don't need it "
                "(based on 120% of historical max clearing price).\n"
                "- **Unneeded budget** is saved rather than wasted on already-free courses."
            )

    elif len(opt_courses) < 2:
        st.caption("Select at least 2 courses above to use the optimizer.")


# ============================================================
# Page: Degree Tracker
# ============================================================
def page_degree_tracker(features, bid_df, eval_df, section_data=None):
    st.header("Degree Progress Tracker")

    st.markdown("Enter courses you've completed or plan to take to track your degree progress.")

    all_courses = sorted(features["dept_code"].unique())
    course_titles = dict(zip(features["dept_code"], features["title"]))

    taken_input = st.multiselect(
        "Courses completed / enrolled",
        options=[f"{c} - {course_titles.get(c, '')}" for c in all_courses],
        help="Select all courses you have taken or are currently enrolled in.",
        key="degree_taken",
    )

    taken_codes = [c.split(" - ")[0] for c in taken_input]

    course_titles_lookup = dict(zip(features["dept_code"], features["title"]))

    def code_with_title(code):
        title = course_titles_lookup.get(code, "")
        if title and pd.notna(title):
            return f"{code} ({title})"
        return code

    def _render_donut(value, total, label, color):
        """Render a donut chart."""
        pct = value / total if total > 0 else 0
        fig = go.Figure(go.Pie(
            values=[value, max(0, total - value)],
            hole=0.7,
            marker_colors=[color, LIGHT_GRAY],
            textinfo="none",
            hoverinfo="skip",
            sort=False,
        ))
        fig.add_annotation(
            text=f"<b>{value}/{total}</b>",
            x=0.5, y=0.5, font_size=20, showarrow=False,
            font_color=color,
        )
        fig.update_layout(
            height=180, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def _req_card(title, courses_str, done=None, color=MAROON):
        """Render a requirement area card."""
        if done is True:
            border_color = "#2ecc71"
            badge = "<span style='color:#2ecc71;font-weight:bold;'>Complete</span>"
        elif done is False:
            border_color = LIGHT_GRAY
            badge = f"<span style='color:{DARK_GRAY};'>Incomplete</span>"
        else:
            border_color = color
            badge = ""
        badge_html = f"<div>{badge}</div>" if badge else ""
        return (
            f'<div style="background:white; border:1px solid {LIGHT_GRAY}; border-left:4px solid {border_color}; border-radius:8px; padding:12px 16px; margin-bottom:6px;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<div style="font-size:14px; font-weight:bold; color:{MAROON};">{title}</div>'
            f'{badge_html}'
            f'</div>'
            f'<div style="font-size:12px; color:{DARK_GRAY}; margin-top:4px;">{courses_str}</div>'
            f'</div>'
        )

    if not taken_codes:
        st.info("Select courses above to see your degree progress.")

        st.markdown("### Degree Requirements Overview")
        st.caption("Total: 2,000 units (min 1,400 Booth courses) = Foundations 300 + FLMBE 700 + Electives 1,000")

        # Leadership (0 credit)
        st.markdown(f"#### Leadership — required (0 credit)")
        lead_codes = DEGREE_REQUIREMENTS["leadership"].get("courses", [])
        lead_str = ", ".join([code_with_title(c) for c in lead_codes])
        lead_str += f"<br><i style='color:{DARK_GRAY};'>Required but does not count toward the 2,000 unit total</i>"
        st.markdown(_req_card("LEAD", lead_str, color="#4A90D9"), unsafe_allow_html=True)

        # Foundations
        st.markdown(f"#### Foundations — 3 areas required (300 units)")
        for area_name, area in DEGREE_REQUIREMENTS["foundations"]["areas"].items():
            courses_str = ", ".join([code_with_title(c) for c in area["basic"]])
            if area["substitutes"]:
                subs = ", ".join([code_with_title(c) for c in area["substitutes"][:5]])
                courses_str += f"<br><i style='color:{DARK_GRAY};'>Substitutes: {subs}</i>"
            st.markdown(_req_card(area_name, courses_str, color=MAROON), unsafe_allow_html=True)

        # FLMBE
        st.markdown(f"#### FLMBE — 7 of 8 areas required (700 units)")
        for cat_name, category in DEGREE_REQUIREMENTS["flmbe"]["categories"].items():
            st.caption(cat_name)
            for area_name, area in category.items():
                courses_str = ", ".join([code_with_title(c) for c in area["basic"]])
                if area["substitutes"]:
                    subs = ", ".join([code_with_title(c) for c in area["substitutes"][:5]])
                    courses_str += f"<br><i style='color:{DARK_GRAY};'>Substitutes: {subs}</i>"
                st.markdown(_req_card(area_name, courses_str, color=TEAL), unsafe_allow_html=True)

        # Electives
        st.markdown(f"#### Electives — 1,000 units")
        st.markdown(_req_card("Elective Courses", "Choose any 1,000 units of electives to complete your degree.", color=DARK_GRAY), unsafe_allow_html=True)
        return

    progress = compute_degree_progress(taken_codes)

    # Leadership card (before donut charts)
    lead_done = progress["leadership"]
    lead_codes = DEGREE_REQUIREMENTS["leadership"].get("courses", [])
    lead_str = ", ".join([code_with_title(c) for c in lead_codes])
    lead_str += " — <i>0 credit (not counted in unit total)</i>"
    st.markdown("### Leadership")
    st.markdown(_req_card("LEAD", lead_str, done=lead_done, color="#4A90D9"), unsafe_allow_html=True)

    # Donut charts
    st.markdown("---")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.plotly_chart(
            _render_donut(progress["foundations_complete"], 3, "Foundations", MAROON),
            use_container_width=True, key="donut_found",
        )
        st.markdown(f"<div style='text-align:center;font-weight:bold;color:{MAROON};margin-top:-10px;'>Foundations</div>", unsafe_allow_html=True)
    with d2:
        st.plotly_chart(
            _render_donut(progress["flmbe_complete"], 7, "FLMBE", TEAL),
            use_container_width=True, key="donut_flmbe",
        )
        st.markdown(f"<div style='text-align:center;font-weight:bold;color:{TEAL};margin-top:-10px;'>FLMBE (7 of 8)</div>", unsafe_allow_html=True)
    with d3:
        units_done = progress["total_units"]
        st.plotly_chart(
            _render_donut(min(units_done, 2000), 2000, "Total Units", DARK_GRAY),
            use_container_width=True, key="donut_total",
        )
        st.markdown(f"<div style='text-align:center;font-weight:bold;color:{DARK_GRAY};margin-top:-10px;'>Total Units</div>", unsafe_allow_html=True)

    # Foundations cards
    st.markdown("### Foundations")
    for area, done in progress["foundations"].items():
        area_data = DEGREE_REQUIREMENTS["foundations"]["areas"][area]
        courses_str = ", ".join([code_with_title(c) for c in area_data["basic"]])
        st.markdown(_req_card(area, courses_str, done=done), unsafe_allow_html=True)

    # FLMBE cards
    st.markdown("### FLMBE Areas")
    for cat_name, category in DEGREE_REQUIREMENTS["flmbe"]["categories"].items():
        st.caption(cat_name)
        for area_name, area in category.items():
            done = progress["flmbe"].get(area_name, False)
            courses_str = ", ".join([code_with_title(c) for c in area["basic"]])
            st.markdown(_req_card(area_name, courses_str, done=done), unsafe_allow_html=True)

    # Concentration progress
    st.markdown("### Concentration Progress")
    conc_courses = get_all_concentration_courses()
    conc_progress = []
    for name, courses in conc_courses.items():
        matched = set(taken_codes) & courses
        units_req = CONCENTRATIONS[name]["units_required"]
        units_done = len(matched) * 100
        pct = min(units_done / units_req, 1.0) if units_req > 0 else 0
        conc_progress.append((name, units_done, units_req, pct, matched))

    conc_progress.sort(key=lambda x: x[3], reverse=True)

    for name, units_done, units_req, pct, matched in conc_progress:
        if pct > 0:
            matched_str = ", ".join([code_with_title(c) for c in sorted(matched)])
            border_color = "#2ecc71" if pct >= 1.0 else TEAL if pct >= 0.5 else LIGHT_GRAY
            badge = "Complete" if pct >= 1.0 else f"{units_done}/{units_req} units ({pct:.0%})"
            badge_color = "#2ecc71" if pct >= 1.0 else DARK_GRAY
            st.markdown(f"""
            <div style="background:white; border:1px solid {LIGHT_GRAY}; border-left:4px solid {border_color};
                        border-radius:8px; padding:12px 16px; margin-bottom:6px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:14px; font-weight:bold; color:{MAROON};">{name}</div>
                    <div style="font-size:13px; font-weight:bold; color:{badge_color};">{badge}</div>
                </div>
                <div style="background:{LIGHT_GRAY}; border-radius:4px; height:8px; margin-top:8px;">
                    <div style="background:{TEAL}; border-radius:4px; height:8px; width:{pct*100:.0f}%;"></div>
                </div>
                <div style="font-size:12px; color:{DARK_GRAY}; margin-top:6px;">{matched_str}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Smart Recommendations ----
    if section_data is not None and taken_codes:
        _render_smart_recommendations(taken_codes, progress, section_data, features, conc_progress, code_with_title)


def _render_smart_recommendations(taken_codes, progress, section_data, features, conc_progress, code_with_title):
    """Render Smart Recommendations based on unfulfilled requirements."""
    st.markdown("---")
    st.markdown(
        f'<div style="background:linear-gradient(135deg, {MAROON}11, {TEAL}18); border:1px solid {TEAL}55; '
        f'border-radius:10px; padding:16px 20px 10px 20px; margin-bottom:12px;">'
        f'<div style="font-size:18px; font-weight:700; color:{MAROON};">✨ Smart Recommendations</div>'
        f'<div style="font-size:13px; color:{DARK_GRAY}; margin-top:4px;">'
        f'Courses available in 2025–26 that efficiently fill your remaining requirements, '
        f'ranked by rating and bid affordability.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    taken_set = set(taken_codes)

    # Available current-year courses (not yet taken)
    avail = section_data[~section_data["dept_code"].isin(taken_set)].copy()
    if avail.empty:
        st.success("You've taken all available courses! No further recommendations.")
        return

    # Build a score for each available dept_code
    avail_codes = avail["dept_code"].unique()

    # --- Identify gaps ---
    # 1. Unfulfilled foundation areas
    gap_foundations = {area: data for area, data in DEGREE_REQUIREMENTS["foundations"]["areas"].items()
                       if not progress["foundations"].get(area, False)}
    foundation_gap_codes = set()
    for area_data in gap_foundations.values():
        foundation_gap_codes |= set(area_data["basic"]) | set(area_data["substitutes"])

    # 2. Unfulfilled FLMBE areas
    gap_flmbe = {}
    for cat_name, category in DEGREE_REQUIREMENTS["flmbe"]["categories"].items():
        for area_name, area_data in category.items():
            if not progress["flmbe"].get(area_name, False):
                gap_flmbe[area_name] = area_data
    flmbe_gap_codes = set()
    for area_data in gap_flmbe.values():
        flmbe_gap_codes |= set(area_data["basic"]) | set(area_data["substitutes"])

    # 3. Top 3 closest concentrations (by progress %)
    top_concs = [c for c in conc_progress if c[3] < 1.0 and c[3] > 0][:3]
    conc_gap_codes = {}
    conc_all = get_all_concentration_courses()
    for name, units_done, units_req, pct, matched in top_concs:
        remaining = conc_all.get(name, set()) - taken_set
        conc_gap_codes[name] = remaining

    # --- Score each course ---
    recommendations = []
    for code in avail_codes:
        if code in taken_set:
            continue
        feat_row = features[features["dept_code"] == code]
        if feat_row.empty:
            continue
        feat_row = feat_row.iloc[0]

        reasons = []
        priority = 0  # higher = more important

        # Check foundation gap
        if code in foundation_gap_codes:
            for area_name, area_data in gap_foundations.items():
                if code in set(area_data["basic"]) | set(area_data["substitutes"]):
                    reasons.append(f"Foundation: {area_name}")
                    priority += 30  # foundations are critical

        # Check FLMBE gap
        if code in flmbe_gap_codes:
            for area_name, area_data in gap_flmbe.items():
                if code in set(area_data["basic"]) | set(area_data["substitutes"]):
                    reasons.append(f"FLMBE: {area_name}")
                    priority += 20

        # Check concentration progress
        for conc_name, remaining in conc_gap_codes.items():
            if code in remaining:
                conc_pct = next((c[3] for c in top_concs if c[0] == conc_name), 0)
                reasons.append(f"{conc_name} ({conc_pct:.0%} done)")
                priority += 10 + int(conc_pct * 10)  # closer to completion = higher priority

        if not reasons:
            continue  # only recommend courses that fill a gap

        rating = feat_row.get("eval_overall", np.nan)
        p1_price = feat_row.get("p1_price_mean", np.nan)
        hours = feat_row.get("eval_hours", np.nan)
        title = feat_row.get("title", "")

        # Composite score: priority first, then rating, then low price
        rating_score = rating if pd.notna(rating) else 3.0
        price_penalty = min(p1_price / 10000, 1.0) if pd.notna(p1_price) and p1_price > 0 else 0.3
        composite = priority + rating_score * 2 - price_penalty * 3

        # Get sections available this year
        sections = avail[avail["dept_code"] == code]
        quarters = sorted(sections["quarter"].unique(), key=lambda q: {"Autumn 2025": 0, "Winter 2026": 1, "Spring 2026": 2, "Summer 2026": 3}.get(q, 99))
        instructors = sections["faculty"].unique().tolist()

        recommendations.append({
            "code": code,
            "title": title,
            "reasons": reasons,
            "priority": priority,
            "rating": rating,
            "p1_price": p1_price,
            "hours": hours,
            "composite": composite,
            "quarters": quarters,
            "instructors": instructors,
        })

    recommendations.sort(key=lambda x: x["composite"], reverse=True)

    if not recommendations:
        st.info("All your remaining requirements are fulfilled, or no matching courses are available this year.")
        return

    # Display top recommendations
    st.caption(f"Found {len(recommendations)} courses that fill your remaining gaps. Showing top {min(len(recommendations), 15)}.")

    for rec in recommendations[:15]:
        price = rec["p1_price"]
        price_val = price if pd.notna(price) else 0
        level, _ = get_bid_level(price_val)
        level_colors = {
            "Free": "#2ecc71", "Low": TEAL,
            "Moderate": "#f39c12", "High": "#e74c3c", "Very High": MAROON,
        }
        level_color = level_colors.get(level, DARK_GRAY)

        # Build reason tags
        reason_tags = ""
        for r in rec["reasons"]:
            if r.startswith("Foundation"):
                reason_tags += f'<span style="background:{MAROON};color:white;padding:2px 8px;border-radius:10px;font-size:11px;margin-right:4px;display:inline-block;margin-bottom:2px;">{r}</span>'
            elif r.startswith("FLMBE"):
                reason_tags += f'<span style="background:{TEAL};color:white;padding:2px 8px;border-radius:10px;font-size:11px;margin-right:4px;display:inline-block;margin-bottom:2px;">{r}</span>'
            else:
                reason_tags += f'<span style="background:#1a73e8;color:white;padding:2px 8px;border-radius:10px;font-size:11px;margin-right:4px;display:inline-block;margin-bottom:2px;">{r}</span>'

        quarters_str = ", ".join(rec["quarters"])
        instr_str = ", ".join(rec["instructors"][:3])
        if len(rec["instructors"]) > 3:
            instr_str += f" +{len(rec['instructors'])-3}"

        rating = rec["rating"]
        hours = rec["hours"]

        st.markdown(
            f'<div style="background:white; border:1px solid {LIGHT_GRAY}; border-left:4px solid {level_color}; border-radius:8px; padding:14px 18px; margin-bottom:6px;">'
            f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
            f'<div style="flex:3;">'
            f'<div style="font-size:15px; font-weight:bold; color:{MAROON};">{rec["code"]} — {rec["title"] if pd.notna(rec["title"]) else ""}</div>'
            f'<div style="font-size:13px; color:{DARK_GRAY}; margin-top:2px;">{instr_str}  ·  {quarters_str}</div>'
            f'<div style="margin-top:6px;">Fills: {reason_tags}</div>'
            f'</div>'
            f'<div style="flex:0.8; text-align:center;">'
            f'<div style="font-size:18px; font-weight:bold; color:{MAROON};">{format_price(price_val)}</div>'
            f'<div style="font-size:11px; color:{DARK_GRAY};">Bid P1</div>'
            f'</div>'
            f'<div style="flex:0.6; text-align:center;">'
            f'<div style="font-size:14px; font-weight:bold; color:{level_color};">{level}</div>'
            f'<div style="font-size:11px; color:{DARK_GRAY};">Demand</div>'
            f'</div>'
            f'<div style="flex:0.5; text-align:center;">'
            f'<div style="font-size:14px;">{f"{rating:.2f}" if pd.notna(rating) else "—"}</div>'
            f'<div style="font-size:11px; color:{DARK_GRAY};">Rating</div>'
            f'</div>'
            f'<div style="flex:0.5; text-align:center;">'
            f'<div style="font-size:14px;">{f"{hours:.1f}" if pd.notna(hours) else "—"}</div>'
            f'<div style="font-size:11px; color:{DARK_GRAY};">Hrs/Wk</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# Page: AI Course Planner
# ============================================================

# 6 quarters in a 2-year MBA
# Finance prerequisite chain
FINANCE_PREREQS = {
    "35200": ["35000"],  # Corp Finance requires Investments first
}
# All finance concentration courses that require both prereqs
FINANCE_CONC_CODES = set()
for _sub in CONCENTRATIONS.get("Finance", {}).get("structure", {}).values():
    FINANCE_CONC_CODES |= set(_sub.get("courses", []))
FINANCE_ADV_CODES = FINANCE_CONC_CODES - {"35000", "35200", "35001"}


def _build_course_plan(preferences, features, section_data):
    """
    Build a 2-year course plan based on user preferences.
    Two-phase approach:
      Phase 1 — Select the optimal set of ~20 courses globally (ensuring requirements).
      Phase 2 — Assign selected courses to quarters (respecting availability & prereqs).
    Returns a list of (quarter_label, [course_codes]) and set of all selected codes.
    """
    target_concs = list(preferences["concentrations"])
    must_take = preferences["must_take"]
    priority = preferences["priority"]  # 0=rating, 1=low workload
    balance = preferences.get("balance", 0.5)  # 0=concentration-focused, 1=balanced
    industry = preferences["industry"]

    # Auto-add Finance if Analytic Finance is selected (requirement per catalog)
    if "Analytic Finance" in target_concs and "Finance" not in target_concs:
        target_concs.append("Finance")

    # Build candidate pool: all unique dept_codes from section_data
    all_codes = set(section_data["dept_code"].unique())
    feat_lookup = {r["dept_code"]: r for _, r in features.iterrows()}

    # Quarter availability from section_data
    quarter_avail = {}  # dept_code -> set of season names
    for _, r in section_data.iterrows():
        code = r["dept_code"]
        season = r["quarter"].split()[0]
        quarter_avail.setdefault(code, set()).add(season)

    # Score each course
    def score_course(code):
        feat = feat_lookup.get(code)
        if feat is None:
            return 0
        rating = feat.get("eval_overall", 3.0)
        if pd.isna(rating):
            rating = 3.0
        hours = feat.get("eval_hours", 10.0)
        if pd.isna(hours):
            hours = 10.0
        rating_weight = 1.0 - priority
        hours_weight = priority
        return rating_weight * (rating / 5.0) + hours_weight * (1.0 - min(hours, 20) / 20.0)

    # --- Requirement helpers ---
    foundation_areas = DEGREE_REQUIREMENTS["foundations"]["areas"]
    flmbe_categories = DEGREE_REQUIREMENTS["flmbe"]["categories"]

    # Build FLMBE area -> course set mapping
    all_flmbe_area_names = []
    flmbe_area_courses = {}  # area_name -> set of valid course codes
    for cat in flmbe_categories.values():
        for area_name, area in cat.items():
            all_flmbe_area_names.append(area_name)
            flmbe_area_courses[area_name] = set(area["basic"]) | set(area["substitutes"])

    def get_flmbe_areas_for(code):
        return {a for a, codes in flmbe_area_courses.items() if code in codes}

    def get_foundation_areas_for(code):
        return {a for a, area in foundation_areas.items()
                if code in area["basic"] or code in area["substitutes"]}

    conc_all = get_all_concentration_courses()

    # Multi-value score: bonus weights shift with balance slider
    # balance=1 (balanced) → FLMBE bonus high, conc bonus low
    # balance=0 (focused)  → FLMBE bonus low,  conc bonus high
    flmbe_bonus = 0.05 + 0.20 * balance    # 0.05–0.25
    conc_bonus = 0.25 - 0.15 * balance      # 0.10–0.25

    def multi_score(code):
        v = score_course(code)
        if get_flmbe_areas_for(code):
            v += flmbe_bonus
        for cn in target_concs:
            if code in conc_all.get(cn, set()):
                v += conc_bonus
        return v

    # Available seasons for scheduling
    all_seasons = {"Autumn", "Winter", "Spring"}

    def is_schedulable(code):
        """Check if a course can be placed in at least one season."""
        return bool(quarter_avail.get(code, set()) & all_seasons)

    # ================================================================
    # PHASE 1: Select the set of ~20 courses to take
    # ================================================================
    selected = set()
    pool = {c for c in all_codes if is_schedulable(c)}

    # 1a: Must-take courses
    for code in must_take:
        if code in pool:
            selected.add(code)
            pool.discard(code)

    # 1b: Foundation courses (3 areas — pick best available for each)
    for area_name, area in foundation_areas.items():
        area_codes = set(area["basic"]) | set(area["substitutes"])
        if selected & area_codes:
            continue  # already covered
        candidates = area_codes & pool
        if candidates:
            best = max(candidates, key=multi_score)
            selected.add(best)
            pool.discard(best)

    # 1c: Finance prereqs — ensure 35000 and 35200 are selected if any
    # finance-related concentration is targeted
    finance_concs = {"Finance", "Analytic Finance"}
    if finance_concs & set(target_concs):
        if "35000" not in selected and "35000" in pool:
            selected.add("35000")
            pool.discard("35000")
        if "35200" not in selected and "35200" in pool:
            selected.add("35200")
            pool.discard("35200")

    # 1d: FLMBE courses
    # balance >= 0.7 → aim for 8/8 (all areas)
    # balance <  0.7 → aim for 7/8 (minimum graduation requirement)
    flmbe_target = 8 if balance >= 0.7 else 7

    def covered_flmbe_areas():
        areas = set()
        for c in selected:
            areas |= get_flmbe_areas_for(c)
        return areas

    while len(covered_flmbe_areas()) < flmbe_target:
        current_covered = covered_flmbe_areas()
        uncovered = set(all_flmbe_area_names) - current_covered
        if len(uncovered) <= (8 - flmbe_target):
            break  # reached target (can skip remaining)

        # Find best course that covers an uncovered area
        best_code = None
        best_val = -1
        for code in pool:
            new_areas = get_flmbe_areas_for(code) & uncovered
            if not new_areas:
                continue
            v = score_course(code) + len(new_areas) * flmbe_bonus
            for cn in target_concs:
                if code in conc_all.get(cn, set()):
                    v += conc_bonus
            if v > best_val:
                best_val = v
                best_code = code
        if best_code is None:
            break
        selected.add(best_code)
        pool.discard(best_code)

    # 1e: Concentration courses — fill each target concentration
    for c_name in target_concs:
        c_course_set = conc_all.get(c_name, set())
        c_req_units = CONCENTRATIONS[c_name]["units_required"]
        matched = len(selected & c_course_set) * 100
        needed = c_req_units - matched

        candidates = sorted(c_course_set & pool, key=lambda c: -score_course(c))
        for code in candidates:
            if needed <= 0:
                break
            # Check finance prereqs
            if code in FINANCE_ADV_CODES and ("35000" not in selected or "35200" not in selected):
                continue
            selected.add(code)
            pool.discard(code)
            needed -= 100

    # 1f: Fill to 20 with best remaining courses
    # When balanced: diversify across FLMBE categories and avoid area repetition
    # When focused: pure score-based (naturally favors concentration courses)
    def fill_score(code):
        v = score_course(code)
        if balance >= 0.5:
            # Bonus for covering FLMBE areas with fewer courses already selected
            areas = get_flmbe_areas_for(code)
            for a in areas:
                count_in_area = sum(1 for c in selected if a in get_flmbe_areas_for(c))
                if count_in_area == 0:
                    v += 0.3 * balance  # strong bonus for new area
                elif count_in_area == 1:
                    v += 0.1 * balance  # small bonus for depth
            # Bonus for covering different FLMBE categories
            covered_cats = set()
            for c in selected:
                for cat_name, cat in flmbe_categories.items():
                    for area_name in cat:
                        if code in flmbe_area_courses.get(area_name, set()):
                            if cat_name not in covered_cats:
                                v += 0.05 * balance
                            covered_cats.add(cat_name)
        else:
            # Focused: bonus for concentration courses
            for cn in target_concs:
                if code in conc_all.get(cn, set()):
                    v += conc_bonus
        return v

    while len(selected) < 20:
        remaining = sorted(pool, key=lambda c: -fill_score(c))
        if not remaining:
            break
        best = remaining[0]
        if best in FINANCE_ADV_CODES and ("35000" not in selected or "35200" not in selected):
            pool.discard(best)
            continue
        selected.add(best)
        pool.discard(best)

    # 1g: If concentrations still not met, allow up to 22
    for c_name in target_concs:
        if len(selected) >= 22:
            break
        c_course_set = conc_all.get(c_name, set())
        c_req_units = CONCENTRATIONS[c_name]["units_required"]
        matched = len(selected & c_course_set) * 100
        needed = c_req_units - matched
        if needed <= 0:
            continue
        candidates = sorted(c_course_set & pool, key=lambda c: -score_course(c))
        for code in candidates:
            if needed <= 0 or len(selected) >= 22:
                break
            selected.add(code)
            pool.discard(code)
            needed -= 100

    total_courses = len(selected)

    # ================================================================
    # PHASE 2: Assign selected courses to quarters
    # ================================================================
    # Distribute across 6 quarters (Y1 Autumn = 3, rest evenly)
    rest = total_courses - 3
    base = rest // 5
    extras = rest % 5
    per_q = [base + (1 if i < extras else 0) for i in range(5)]
    quarter_caps = [3] + per_q

    planner_quarters = [
        ("Year 1 — Autumn", "Autumn", 1, quarter_caps[0]),
        ("Year 1 — Winter", "Winter", 1, quarter_caps[1]),
        ("Year 1 — Spring", "Spring", 1, quarter_caps[2]),
        ("Year 2 — Autumn", "Autumn", 2, quarter_caps[3]),
        ("Year 2 — Winter", "Winter", 2, quarter_caps[4]),
        ("Year 2 — Spring", "Spring", 2, quarter_caps[5]),
    ]

    # Assignment priority: foundations first, then prereqs, FLMBE, then rest
    def assign_priority(code):
        if get_foundation_areas_for(code):
            return 0
        if code == "35000":
            return 1
        if code == "35200":
            return 2
        if get_flmbe_areas_for(code):
            return 3
        if code in FINANCE_ADV_CODES:
            return 5
        return 4

    ordered = sorted(selected, key=lambda c: (assign_priority(c), -score_course(c)))

    plan = []
    assigned = set()

    for q_label, season, year, max_courses in planner_quarters:
        quarter_courses = []
        to_assign = [c for c in ordered if c not in assigned]

        for code in to_assign:
            if len(quarter_courses) >= max_courses:
                break
            # Check season availability (Year 2 assumes same seasons as Year 1)
            if season not in quarter_avail.get(code, set()):
                continue
            # Finance prereq: 35200 needs 35000 in a prior quarter
            prior = assigned - set(quarter_courses)
            if code == "35200" and "35000" not in prior:
                continue
            if code in FINANCE_ADV_CODES and ("35000" not in prior or "35200" not in prior):
                continue
            quarter_courses.append(code)
            assigned.add(code)

        plan.append((q_label, quarter_courses))

    # Place any unassigned courses — try quarters with room first
    unassigned = [c for c in ordered if c not in assigned]
    for code in unassigned:
        placed = False
        # Pass 1: respect cap, ignore season (Year 2 may differ)
        for i, (q_label, season, year, max_courses) in enumerate(planner_quarters):
            if len(plan[i][1]) >= max_courses:
                continue
            prior = set()
            for j in range(i):
                prior |= set(plan[j][1])
            if code == "35200" and "35000" not in prior:
                continue
            if code in FINANCE_ADV_CODES and ("35000" not in prior or "35200" not in prior):
                continue
            plan[i][1].append(code)
            assigned.add(code)
            placed = True
            break
        # Pass 2: force-place into the quarter with fewest courses (exceed cap if needed)
        if not placed:
            best_i = min(range(len(plan)), key=lambda i: len(plan[i][1]))
            plan[best_i][1].append(code)
            assigned.add(code)

    return plan, assigned


def _generate_plan_explanation(plan, preferences, features):
    """Use Claude Haiku to generate a natural language explanation of the plan."""
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "Set ANTHROPIC_API_KEY in .streamlit/secrets.toml to enable AI-generated plan explanations."
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception:
        return "Could not connect to AI service. Plan was generated using algorithmic optimization."

    feat_lookup = {r["dept_code"]: r for _, r in features.iterrows()}

    plan_text = ""
    for q_label, courses in plan:
        course_strs = []
        for c in courses:
            feat = feat_lookup.get(c)
            if feat is not None:
                title = feat.get("title", "") if pd.notna(feat.get("title", "")) else ""
                rating = feat.get("eval_overall", np.nan)
            else:
                title = ""
                rating = np.nan
            r_str = f" (rated {rating:.1f}/5)" if pd.notna(rating) else ""
            course_strs.append(f"{c} {title}{r_str}")
        plan_text += f"\n{q_label}: " + ", ".join(course_strs)

    conc_str = ", ".join(preferences["concentrations"]) if preferences["concentrations"] else "None specified"
    priority_str = "high ratings" if preferences["priority"] < 0.3 else "low workload" if preferences["priority"] > 0.7 else "balanced"

    prompt = f"""A Chicago Booth MBA student wants a 2-year course plan. Their preferences:
- Target industry: {preferences['industry']}
- Target concentrations: {conc_str}
- Priority: {priority_str}
- Must-take courses: {', '.join(preferences['must_take']) if preferences['must_take'] else 'None'}

Here is the generated plan:{plan_text}

Write a brief (3-4 sentences) explanation of why this plan works well for the student. Mention how it sequences prerequisites, balances workload, and progresses toward concentrations. Be specific and concise."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"(Could not generate explanation: {e})"


def page_course_planner(features, bid_df, eval_df, section_data):
    st.header("AI Course Planner")

    st.markdown(
        f'<div style="background:linear-gradient(135deg, {MAROON}11, {TEAL}18); border:1px solid {TEAL}55; '
        f'border-radius:10px; padding:16px 20px 10px 20px; margin-bottom:16px;">'
        f'<div style="font-size:14px; font-weight:700; color:{MAROON}; margin-bottom:6px;">🗓️ Plan Your 2-Year MBA Curriculum</div>'
        f'<div style="font-size:12px; color:{DARK_GRAY}; margin-bottom:8px;">'
        f'Enter your preferences below, and AI will generate a quarter-by-quarter course lineup '
        f'that fulfills degree requirements, progresses toward your target concentrations, and '
        f'respects prerequisite sequencing.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- Input Form ---
    col_left, col_right = st.columns(2)

    with col_left:
        industry = st.selectbox(
            "Target Industry",
            ["Finance", "Consulting", "Technology", "Entrepreneurship",
             "Marketing", "Healthcare", "Real Estate", "Private Equity / VC",
             "General Management", "Other"],
            key="planner_industry",
        )

        target_concs = st.multiselect(
            "Target Concentrations",
            sorted(CONCENTRATIONS.keys()),
            help="Select 1-3 concentrations to focus on.",
            max_selections=3,
            key="planner_concentrations",
        )

        all_courses = sorted(features["dept_code"].unique())
        course_titles = dict(zip(features["dept_code"], features["title"]))
        must_take_input = st.multiselect(
            "Must-take courses",
            options=[f"{c} - {course_titles.get(c, '')}" for c in all_courses],
            help="Courses you definitely want in your plan.",
            key="planner_must_take",
        )
        must_take_codes = set(c.split(" - ")[0] for c in must_take_input)

    with col_right:
        priority = st.slider(
            "Priority",
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            format="%.1f",
            help="0.0 = Prioritize high ratings · 1.0 = Prioritize low workload",
            key="planner_priority",
        )
        st.caption("← High Ratings · Low Workload →")

        balance = st.slider(
            "Course Balance",
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            format="%.1f",
            help="0.0 = Concentration-focused (bold picks) · 1.0 = Balanced (diversified across areas, all FLMBE)",
            key="planner_balance",
        )
        st.caption("← Concentration Focus · Balanced →")

    # Generate button
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        generate = st.button("🚀 Generate Plan", use_container_width=True, type="primary")

    if generate:
        if not target_concs:
            st.warning("Please select at least one target concentration.")
            return

        preferences = {
            "industry": industry,
            "concentrations": target_concs,
            "priority": priority,
            "balance": balance,
            "must_take": must_take_codes,
        }

        with st.spinner("Building your personalized course plan..."):
            plan, taken = _build_course_plan(preferences, features, section_data)
            explanation = _generate_plan_explanation(plan, preferences, features)

        # Store in session state
        st.session_state["planner_plan"] = plan
        st.session_state["planner_prefs"] = preferences
        st.session_state["planner_taken"] = taken
        st.session_state["planner_explanation"] = explanation

    # Display plan from session state
    plan = st.session_state.get("planner_plan")
    preferences = st.session_state.get("planner_prefs")
    taken = st.session_state.get("planner_taken")

    if plan is None:
        st.info("Configure your preferences above and click **Generate Plan** to create your 2-year curriculum.")
        return

    st.markdown("---")

    feat_lookup = {r["dept_code"]: r for _, r in features.iterrows()}

    # Summary metrics
    total_courses = sum(len(courses) for _, courses in plan)
    total_units = total_courses * 100

    # Check requirement fulfillment
    progress = compute_degree_progress(list(taken))
    conc_all = get_all_concentration_courses()
    conc_progress_list = []
    for c_name in preferences["concentrations"]:
        c_courses = conc_all.get(c_name, set())
        matched = taken & c_courses
        req = CONCENTRATIONS[c_name]["units_required"]
        conc_progress_list.append((c_name, len(matched) * 100, req))

    # Estimate total bid budget for the plan
    total_bid_est = 0
    for _, courses in plan:
        for code in courses:
            feat = feat_lookup.get(code)
            if feat is not None:
                p1 = feat.get("p1_price_mean", 0)
                total_bid_est += int(p1) if pd.notna(p1) else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Courses", total_courses)
    m2.metric("Total Units", f"{total_units:,}")
    m3.metric("Foundations", f"{progress['foundations_complete']}/3")
    m4.metric("FLMBE", f"{progress['flmbe_complete']}/7")
    m5.metric("Est. Total Bid", f"{total_bid_est:,} pts")

    # Concentration progress bars
    for c_name, c_done, c_req in conc_progress_list:
        pct = min(c_done / c_req, 1.0) if c_req > 0 else 0
        status = "Complete" if pct >= 1.0 else f"{c_done}/{c_req} units"
        st.markdown(
            f'<div style="display:flex; align-items:center; margin-bottom:6px;">'
            f'<div style="width:180px; font-size:13px; font-weight:bold; color:{MAROON};">{c_name}</div>'
            f'<div style="flex:1; background:{LIGHT_GRAY}; border-radius:4px; height:8px; margin:0 12px;">'
            f'<div style="background:{TEAL}; border-radius:4px; height:8px; width:{pct*100:.0f}%;"></div></div>'
            f'<div style="font-size:12px; color:{DARK_GRAY}; width:100px;">{status}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Quarter-by-quarter plan
    for q_label, courses in plan:
        # Compute quarter bid subtotal
        q_bid = 0
        for code in courses:
            feat = feat_lookup.get(code)
            if feat is not None:
                p1 = feat.get("p1_price_mean", 0)
                q_bid += int(p1) if pd.notna(p1) else 0
        bid_label = f"  ·  Est. Bid: {q_bid:,} pts" if q_bid > 0 else ""
        st.markdown(f"### {q_label}{bid_label}")
        if not courses:
            st.caption("No courses scheduled.")
            continue

        for code in courses:
            feat = feat_lookup.get(code)
            title = ""
            rating = np.nan
            hours = np.nan
            p1_price = 0
            if feat is not None:
                title = feat.get("title", "") if pd.notna(feat.get("title", "")) else ""
                rating = feat.get("eval_overall", np.nan)
                hours = feat.get("eval_hours", np.nan)
                p1 = feat.get("p1_price_mean", 0)
                p1_price = p1 if pd.notna(p1) else 0

            level, _ = get_bid_level(p1_price)
            level_colors = {
                "Free": "#2ecc71", "Low": TEAL,
                "Moderate": "#f39c12", "High": "#e74c3c", "Very High": MAROON,
            }
            level_color = level_colors.get(level, DARK_GRAY)

            req_info = check_course_requirements(code)
            req_tags = _build_req_tags(req_info)

            with st.container(border=True):
                c_info, c_price, c_demand, c_rating, c_hours = st.columns([3.5, 0.8, 0.7, 0.5, 0.5])
                with c_info:
                    st.markdown(f'<div style="font-size:15px; font-weight:bold; color:{MAROON};">{code} — {title}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="margin-top:-8px; margin-bottom:8px;">{req_tags}</div>', unsafe_allow_html=True)
                with c_price:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:18px; font-weight:bold; color:{MAROON};">{format_price(p1_price)}</div><div style="font-size:11px; color:{DARK_GRAY};">Bid P1</div></div>', unsafe_allow_html=True)
                with c_demand:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:14px; font-weight:bold; color:{level_color};">{level}</div><div style="font-size:11px; color:{DARK_GRAY};">Demand</div></div>', unsafe_allow_html=True)
                with c_rating:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:14px;">{f"{rating:.2f}" if pd.notna(rating) else "—"}</div><div style="font-size:11px; color:{DARK_GRAY};">Rating</div></div>', unsafe_allow_html=True)
                with c_hours:
                    st.markdown(f'<div style="text-align:center;"><div style="font-size:14px;">{f"{hours:.1f}" if pd.notna(hours) else "—"}</div><div style="font-size:11px; color:{DARK_GRAY};">Hrs/Wk</div></div>', unsafe_allow_html=True)
            st.markdown('<div style="margin-bottom:4px;"></div>', unsafe_allow_html=True)

    # AI explanation
    st.markdown("---")
    st.markdown("### AI Analysis")
    explanation = st.session_state.get("planner_explanation", "")
    if explanation:
        st.markdown(
            f'<div style="background:linear-gradient(135deg, {MAROON}08, {TEAL}10); '
            f'border:1px solid {TEAL}55; border-radius:10px; padding:16px 20px;">'
            f'<div style="font-size:13px; color:{BODY_TEXT}; line-height:1.6;">{explanation}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "Note: This plan is based on 2025-26 course offerings and assumes similar availability in Year 2. "
        "Actual course availability, scheduling conflicts, and additional prerequisites may require adjustments."
    )


if __name__ == "__main__":
    main()
