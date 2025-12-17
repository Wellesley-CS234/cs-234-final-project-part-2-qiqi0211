# Author: QiQi Li
# Description: CS 234 Final Project Streamlit
# Import all libraries
import streamlit as st
import pandas as pd
import duckdb
from scipy.signal import find_peaks
import plotly.graph_objects as go
import altair as alt
import plotly.express as px
import requests
import os
import time

st.set_page_config(layout="wide", page_title="Wikipedia Trends During the Olympics🏅")

DUCKDB_URL = "https://cs.wellesley.edu/~eni/duckdb/final.duckdb"
LOCAL_PATH = "final.duckdb"


@st.cache_resource
def ensure_database():
    if os.path.exists(LOCAL_PATH):
        return True

    with st.spinner("Downloading database..."):
        for attempt in range(1, 4):
            try:
                r = requests.get(DUCKDB_URL, stream=True, timeout=20)
                r.raise_for_status()
                with open(LOCAL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                # Validate DuckDB
                conn = duckdb.connect(LOCAL_PATH, read_only=True)
                conn.execute("SELECT 1").fetchall()
                conn.close()
                return True
            except Exception as e:
                if attempt < 3:
                    time.sleep(5)
                else:
                    st.error(f"Failed to download or validate DuckDB: {e}")
                    return False
    return False

@st.cache_data
def load_all_data():
    if not os.path.exists(LOCAL_PATH):
        return pd.DataFrame()
    conn = duckdb.connect(LOCAL_PATH, read_only=True)
    df = conn.execute("SELECT * FROM wiki_pageviews").df()
    conn.close()
    df["article"] = df["article"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

if ensure_database():
    df_all = load_all_data()
    if df_all.empty:
        st.warning("Database is empty or could not be read.")
    else:
        st.success("Database loaded!")
        st.dataframe(df_all.head())
else:
    st.error("Cannot proceed without a valid database.")


# Create tabs to organize all the content
intro, data_summary, features, classification, hypothesis, visuals, summary = st.tabs(["1. Introduction",
        "2. Data Summary",
        "3. New Features",
        "4. Text Classification",
        "5. Hypothesis Testing",
        "6. Interactive Visualization",
        "7. Summary"
        ])

# --- Introduction ---
with intro:

    # -- Title ---
    st.title("From Pre-Game to Post-Game: Wikipedia Trends During the Olympics🏅")

    # --- Introduction ---
    st.header("Introduction")
    st.write(""" During Summer 2024, The 2024 Summer Olympics (July 26 – August 11, 2024) earned a lot of public attention. Athletes from across the globe are competing in a wide range of sports, and the public is eager to follow their performances and achievements. 
            To understand which topics, athletes, and events generated the most interest, we can look at the number of Wikipedia pageviews for related articles. Wikipedia pageview data provide a good proxy for public attention. 
            By examining trends **a month before, during, and a month after the Olympics**, we can see how public interest shifts over time. 
            This analysis allows us to investigate which athletes and topics dominated public attention, and to compare the popularity of sports-related articles versus non-sports topics. 
            Understanding these trends gives us insight into how global events like the Olympics capture and sustain public interest.""")

    st.write("Therefore, the three research questions I explored are:")
    st.write("1. How do popular article topics differ before, during, and after the 2024 Summer Olympics?")
    st.write("2. How does public attention differ between sports-related and non-sports articles before, during, and after the Olympics?")
    st.write("3. How does public attention vary across languages before, during, and after the 2024 Olympics?")

    st.write("My initial thoughts are: \n" \
    "1. Some popular article topics might stay even after olympics since people are still interested in the olympics.\n"
    "2. After the olympics, there are more public attention on sports-related articles.\n"
    "3. After the Olympics, there might be an increase in French in terms of public attention.")

with data_summary:
    st.header("Data Summary")
    st.write("""
            I collected my data through connecting to the DuckDB database to the Wellesley Server. My data is from the 2024 Wikipedia DPDP data for all countries.
             Then, I queried all the articles within the range of one month before, during, and one month after the olympics. Data is from "2024-06-26" to "2024-09-11".
             Next, I handled all the unicode characters in the article names.
             I dropped all rows where their QID is none. I also use the language_mapping.json file we used earlier in the semester to perfrom the language mapping for the data file.
             Then, I groupby all the QID and get the totalpageviews for all QIDS.
             I sorted all the QIDs and got the highest 10000 for each period.""")
    st.write("In this case, I took out all the QIDs for each period for wikidata processing later and saved all dataframes into csv files.")

    st.write("One note is that I dropped all the articles that didn't have QIDs. In this case, these articles are ignored in my analysis.")

    st.write("Then, I combined all the dataframes into a csv called d_all.csv and use all QID.txt files and the given getwikidata.py to get wikidata attributes and labels using their QIDs.")

    st.write("I used all these files for all three of my research questions. For the sports one, I performed text classification and combined it with d_all.csv and got final.csv to answer my research question.")
    
    st.subheader("Descriptive Statistics")

    total_unique_articles = df_all["qid"].nunique()

    total_unique_languages = df_all["language_full"].nunique()

    total_pageviews_before = df_all[df_all["period"] == "Before"]["pageviews"].sum()

    total_pageviews_during = df_all[df_all["period"] == "During"]["pageviews"].sum()

    total_pageviews_after = df_all[df_all["period"] == "After"]["pageviews"].sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Unique Articles",
            value=f"{total_unique_articles:,}" 
        )
        st.caption(f"Count from the column: **{'qid'}**")

    with col2:
        st.metric(
            label="Total Unique Languages",
            value=f"{total_unique_languages:,}"
        )
        st.caption(f"Count from the column: **{"language_full"}**")

    with col3:
        st.metric(
            label="Total Pageviews A Month Before Olympics",
            value=f"{total_pageviews_before:,}"
        )
        st.caption(f"Count from the column: **{"pageviews"}**")
    
    with col4:
        st.metric(
            label="Total Pageviews During Olympics",
            value=f"{total_pageviews_during:,}"
        )
        st.caption(f"Count from the column: **{"pageviews"}**")
        st.markdown("---")

    with col5:
        st.metric(
            label="Total Pageviews A Month After Olympics",
            value=f"{total_pageviews_after:,}"
        )
        st.caption(f"Count from the column: **{"pageviews"}**")
    
    
    st.info(
        f"""
        **Summary of Counts:**
        * Unique Articles: **{total_unique_articles:,}**
        * Unique Languages: **{total_unique_languages:,}**
        * Total Pageviews Before Olympics: **{total_pageviews_before:,}**
        * Total Pageviews During Olympics: **{total_pageviews_during:,}**
        * Total Pageviews After Olympics: **{total_pageviews_after:,}**

        """
    )

with features:
    st.header("New Features")
    st.write("""
    ### 1️⃣ Sports Classification Feature
    **Purpose:** Identify whether each Wikipedia article is related to sports or not.

    - **Method:**
    - Get the sport attribute as my truth value for each article and save it to a separate CSV (`sport.csv`) containing `QID` and `truth`.
    - Merged `truth` into the main dataset using `QID`.
    - `truth` labels: NS = Non-Sports, S = Sports

    ### 2️⃣ Daily Aggregation Feature (for Visualization)
    **Purpose:** Aggregate pageviews per day for Sports vs Non-Sports comparison.

    - **Method:** Group by `date` and `truth` to calculate total pageviews.""")
    st.write("""These features allow us to explore differences in attention between Sports and Non-Sports articles over time.""")

with classification:
    st.header("Article Classification")
    st.markdown("""**Method:** Articles were labeled using zero-shot text classification based on their titles, descriptions, and their occupations.  
    This allows categorization without a labeled training dataset, but may introduce misclassification for ambiguous topics.""")
    st.write("Note: S: Sports and NS: Non-Sports")

    st.subheader("Sample of Classified Articles")
    df_unique = df_all.sort_values("pageviews", ascending=False).drop_duplicates(subset="qid")
    df_top50 = df_unique[["article", "truth", "pageviews"]].head(50)
    st.dataframe(df_top50, use_container_width=True)

    st.subheader("""Classification Evaluation""")
    st.write("""Confusion Matrix (True Labels: ['S', 'NS']):\n
    [[ 3783   319]\n
    [  519 11767]]""")

    st.write("""
    Accuracy: 0.9489\n
    Recall (for 'S'): 0.9222\n
    Precision (for 'S'): 0.8794\n
    F1-Score (for 'S'): 0.9003""")

    st.subheader("Distribution of Classification Labels")

    df_period_count = df_all.groupby(["period", "truth"], as_index=False)["article"].count()
    df_period_count.rename(columns={"article": "count"}, inplace=True)

# --- Compute percentage within each period ---
    df_period_count["percent"] = df_period_count.groupby("period")["count"].transform(lambda x: x / x.sum() * 100)

# --- Stacked bar chart with percentage tooltip ---
    chart = alt.Chart(df_period_count).mark_bar().encode(
    x=alt.X("period:N", title="Period"),
    y=alt.Y("count:Q", title="Percentage of Articles", stack="normalize"),
    color=alt.Color("truth:N", title="Article Type"),
    tooltip=[
        alt.Tooltip("period:N", title="Period"),
        alt.Tooltip("truth:N", title="Article Type"),
        alt.Tooltip("count:Q", title="Number of Articles"),
        alt.Tooltip("percent:Q", title="Percentage", format=".1f")  # Show percentage with 1 decimal
    ]).properties(
    height=400)

    st.altair_chart(chart, use_container_width=True)
    st.markdown("""These classification labels are used to analyze differences in attention patterns between sports and non-sports content before, during, and after the Olympic Games.""")

with hypothesis:
    st.header("Hypothesis Testing")
    st.write("Null Hypothesis: The proportion of total Wikipedia pageviews attributed to sports-related articles is the same during the Olympic period and the post-Olympic period.")
    st.write("Alternative Hypothesis: The proportion of total Wikipedia pageviews attributed to sports-related articles is higher during the Olympic period than the post-Olympic period.")

    # Define Olympic period
    O_START = pd.to_datetime("2024-07-26").to_pydatetime()
    O_END = pd.to_datetime("2024-08-11").to_pydatetime()

    # Slider date range
    min_date = pd.to_datetime("2024-06-26").to_pydatetime()
    max_date = pd.to_datetime("2024-09-11").to_pydatetime()


    h_start, h_end = st.slider(
        "Select Date Range for Hypothesis Analysis",
        min_value=min_date,
        max_value=max_date,
        value=(O_START, O_END),
        format="YYYY-MM-DD",
        key="hypothesis_slider"
    )

    # Filter and aggregate
    df_hyp = df_all[(df_all["date"] >= pd.Timestamp(h_start)) & (df_all["date"] <= pd.Timestamp(h_end))]
    df_daily = df_hyp.groupby(["date", "truth"], as_index=False)["pageviews"].sum()

    # Pivot for share calculation
    df_share = df_daily.pivot(index="date", columns="truth", values="pageviews").fillna(0)
    df_share["sports_share"] = df_share.get("S", 0) / (df_share.get("S", 0) + df_share.get("NS", 0))
    df_share = df_share.reset_index()

    # Plot
    fig = px.line(
        df_share,
        x="date",
        y="sports_share",
        title="Share of Sports-Related Wikipedia Pageviews Over Time"
    )
    fig.add_vrect(
        x0=O_START,
        x1=O_END,
        fillcolor="orange",
        opacity=0.25,
        layer="below",
        line_width=0,
        annotation_text="Olympics",
        annotation_position="top left"
    )
    fig.update_layout(yaxis_title="Sports Share of Total Pageviews", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    # Descriptive stats
    def period_stats(df):
        sports_views = df[df["truth"] == "S"]["pageviews"].sum()
        total_views = df["pageviews"].sum()
        return sports_views, total_views, sports_views / total_views

    selected = period_stats(df_hyp)
    stats_df = pd.DataFrame({
        "Period": ["Selected Range"],
        "Sports Pageviews": [selected[0]],
        "Total Pageviews": [selected[1]],
        "Sports Share": [f"{selected[2]*100:.2f}%"]
    })

    st.subheader("Descriptive Statistics")
    st.write("data of selected period")
    st.dataframe(stats_df, use_container_width=True)

    st.write("Data of Olympics periods")
    # Aggregate total pageviews per period
    period_df = (
    df_all.groupby("period", as_index=False)
    .agg(
        Sports_Pageviews=("pageviews", lambda x: x[df_all.loc[x.index, 'truth'] == "S"].sum()),
        Total_Pageviews=("pageviews", "sum")
    ))

    # Calculate sports share
    period_df["Sports_Share"] = (period_df["Sports_Pageviews"] / period_df["Total_Pageviews"] * 100).round(2).astype(str) + "%"
    st.dataframe(period_df,use_container_width=True)

    st.write("Note: Sport share is claculated by number of pageviews of sports articles over the total number of pageviews of sports articles in the selected period.")
    
    st.subheader("Hypothesis Evaluation")
    st.write("""
    **Evaluation:**  
    The share of sports-related Wikipedia pageviews increases noticeably during the Olympic period 
    and declines afterward. Descriptive statistics show a higher proportion of sports-related 
    attention during the Olympics compared to the post-Olympic period. Based on these visual and 
    descriptive results, the null hypothesis is rejected. The observed pattern is consistent 
    with the alternative hypothesis.
    """)


with visuals:
    st.header("Visualizations")   
    st.subheader("Use the interactive feature to see how the popular article topics differ before, during, and after the 2024 Summer Olympics.")
# --- Aggregate total pageviews per day ---
    df_time = df_all.groupby("date", as_index=False)["pageviews"].sum().sort_values("date")


# Example: start and end dates from your dataset
    min_date = df_all['date'].min().to_pydatetime()  # convert pd.Timestamp -> datetime
    max_date = df_all['date'].max().to_pydatetime()

# Use in slider
    start_date, end_date = st.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
    )

# Filter based on selected range
    df_time_range = df_time[(df_time['date'] >= start_date) & (df_time['date'] <= end_date)]
    df_all_range = df_all[(df_all['date'] >= start_date) & (df_all['date'] <= end_date)]

# --- Detect peaks ---
    y = df_time_range["pageviews"].values
    peaks, _ = find_peaks(y, prominence=0.1 * y.max(), distance=5)
    peak_dates = df_time_range.iloc[peaks]["date"]

# --- Build hover text for peaks ---
    peak_hover = []
    for d in peak_dates:
        top_articles = (
        df_all_range[df_all_range["date"] == d]
        .groupby(["article", "truth"], as_index=False)["pageviews"]
        .sum()
        .sort_values("pageviews", ascending=False)
        .head(5))

        total_views = df_time_range.loc[df_time_range['date'] == d, 'pageviews'].values[0]
        text = f"<b>{d.strftime('%Y-%m-%d')}</b><br>Total views: {total_views:,}<br>"
        for _, r in top_articles.iterrows():
            text += f"{r['article']} ({r['truth']}): {r['pageviews']:,}<br>"
        peak_hover.append(text)

# --- Plotly Figure ---
    fig = go.Figure()

# Line chart
    fig.add_trace(go.Scatter(
        x=df_time_range["date"],
        y=df_time_range["pageviews"],
        mode="lines",
        name="Total Pageviews"
    ))

# Peaks
    fig.add_trace(go.Scatter(
        x=peak_dates,
        y=df_time_range.iloc[peaks]["pageviews"],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="Detected Peaks",
        hovertext=peak_hover,
        hoverinfo="text"
    ))

    # Layout
    fig.update_layout(
        title="Wikipedia Pageviews with Detected Peaks",
        xaxis_title="Date",
        yaxis_title="Pageviews",
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Conditional Peak Report ---
    if 'show_report' not in st.session_state:
        st.session_state['show_report'] = False

    def toggle_report():
        st.session_state['show_report'] = not st.session_state['show_report']

    label = "Hide Peak Report" if st.session_state['show_report'] else "Generate Peak Report"
    st.button(label, on_click=toggle_report)

    if st.session_state['show_report']:
        st.subheader("🚨 Peak Contribution Report (Top 5 Articles per Peak)")
        report_rows = []
        for d in peak_dates:
            top_articles = (
                df_all_range[df_all_range["date"] == d]
                .groupby(["article","truth"], as_index=False)["pageviews"]
                .sum()
                .sort_values("pageviews", ascending=False)
                .head(5)
            )
            for _, r in top_articles.iterrows():
                report_rows.append({
                "Peak Date": d.date(),
                "Article": r['article'],
                "Pageviews": r['pageviews'],
                "Truth": r['truth'], 
            })
        peak_report_df = pd.DataFrame(report_rows)
        st.dataframe(peak_report_df, use_container_width=True)

    st.subheader("Explore Sports vs Non-Sports Wikipedia Pageviews Before, During, and After the 2024 Summer Olympics.")
    # --- Aggregate total pageviews per day for S and NS ---
    df_time = df_all.groupby(["date", "truth"], as_index=False)["pageviews"].sum()
    df_time_pivot = df_time.pivot(index="date", columns="truth", values="pageviews").fillna(0).reset_index()

    # --- Slider for selecting date range ---
    min_date = df_all['date'].min().to_pydatetime()
    max_date = df_all['date'].max().to_pydatetime()

    start_date, end_date = st.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD",
    key="pageview_date_range_slider"  # <-- unique key
    )

# --- Filter data based on selected range ---
    df_time_range = df_time_pivot[(df_time_pivot['date'] >= pd.Timestamp(start_date)) &
                              (df_time_pivot['date'] <= pd.Timestamp(end_date))]

# --- Plotly Figure ---
    fig = go.Figure()

# Line chart for Sports
    if "S" in df_time_range.columns:
        fig.add_trace(go.Scatter(
        x=df_time_range["date"],
        y=df_time_range["S"],
        mode="lines",
        name="Sports Pageviews",
        line=dict(color="blue")
    ))

# Line chart for Non-Sports
    if "NS" in df_time_range.columns:
        fig.add_trace(go.Scatter(
        x=df_time_range["date"],
        y=df_time_range["NS"],
        mode="lines",
        name="Non-Sports Pageviews",
        line=dict(color="green")
    ))

# Layout
    fig.update_layout(
    title="Wikipedia Pageviews: Sports vs Non-Sports",
    xaxis_title="Date",
    yaxis_title="Pageviews",
    hovermode="x unified"  # shows both lines in same hover
)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pageviews by Language Over Time")
    st.subheader("Interactive view of how popular article topics differ across languages.")

# --- Aggregate total pageviews per day by language ---
    df_lang = df_all.groupby(["date", "language_full"], as_index=False)["pageviews"].sum()

# --- Option to exclude English ---
    exclude_english = st.checkbox("Exclude English", value=False)
    if exclude_english:
        df_lang = df_lang[df_lang["language_full"] != "English"]

# --- Select top N languages ---
    top_languages = (
    df_lang.groupby("language_full")["pageviews"].sum()
    .sort_values(ascending=False)
    .head(5)
    .index
)
    df_lang = df_lang[df_lang["language_full"].isin(top_languages)]

# --- Pivot for plotting ---
    df_lang_pivot = df_lang.pivot(index="date", columns="language_full", values="pageviews").fillna(0).reset_index()

# --- Date range slider ---
    min_date = df_all['date'].min().to_pydatetime()
    max_date = df_all['date'].max().to_pydatetime()

    start_date, end_date = st.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD",
    key="language_line_slider"
)

    df_lang_range = df_lang_pivot[(df_lang_pivot['date'] >= pd.Timestamp(start_date)) &
                              (df_lang_pivot['date'] <= pd.Timestamp(end_date))]

# --- Plot multi-line chart ---
    fig = go.Figure()
    for lang in df_lang_range.columns[1:]:  # skip 'date'
        fig.add_trace(go.Scatter(
        x=df_lang_range["date"],
        y=df_lang_range[lang],
        mode="lines+markers",
        name=lang
    ))

    fig.update_layout(
    title="Wikipedia Pageviews by Language Over Time",
    xaxis_title="Date",
    yaxis_title="Pageviews",
    hovermode="x unified"
)

    st.plotly_chart(fig, use_container_width=True)


with summary:
    st.header("Summar and Ethical Considerations")
    st.subheader("Summary")
    st.write("My takeaways are:\n")
    st.write("1. Before the Olympics: Attention is spread across diverse topics, with non-sports articles dominating. English articles lead in pageviews, though other major languages maintain steady engagement. Once I exclude English articles, Japanese is consistent higher than any other languages.\n")
    st.write("2. During the Olympics: Sports-related articles surge, almost surpassing non-sports articles. Pageviews increase across multiple languages, with spikes strongest in Japanese and French.\n")
    st.write("3. After the Olympics: Sports attention declines, and non-sports articles regain prominence. Pageviews stabilize across languages, with Japanese articles remaining higher than all others.\n")
    
    st.write("Overall, I am pretty confident that the results are reliable.")
    st.write("I used zero-shot text classification. I included label, description, and occupation to classify. The accuracy is 94%.")

    st.subheader("Limitations")
    st.write(
    "1. The data ignores articles without QIDs, even if their pageviews are high, so some of the articles are not accurate.\n"
    "2. I used zero-shot text classification, so it may have lower accuracy, especially for ambiguous or multi-topic articles.\n"
    "3. Some spikes in pageviews may be caused by unrelated events (news, viral trends) coinciding with the Olympics, not the Games themselves.\n"
    "4. The method assumes uniform interest across countries, ignoring regional differences in Olympic coverage."
)

    st.subheader("Ethical Considerations")
    st.write("""While the dataset comes from open-source Wikipedia pageviews, there are still some ethical considerations:

    1. Data Representation Bias
   - Wikipedia content is not evenly distributed across topics, languages, or regions.
   - Certain sports, countries, or athletes might be overrepresented due to higher online activity in some regions.
   - Less popular topics may appear underrepresented, which can bias the results toward highly-viewed articles.

    2. Language Bias
   - Pageviews may favor articles in English or other widely used languages.
   - Non-English speakers’ interests might be underrepresented, leading to skewed conclusions about global attention.

    3. Algorithmic Bias in Classification
   - My sports vs. non-sports classification may mislabel some articles.
   - Misclassifications can affect downstream analyses like visualizations and trend comparisons.""")

    st.write("My results show a bias toward English-language articles and sports that are more popular globally, such as track and swimming, while niche sports have far fewer pageviews.")
