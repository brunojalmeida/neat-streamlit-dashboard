import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Container height constants
CHART_HEIGHT = 480
DATAFRAME_HEIGHT = 380


@st.cache_data
def generate_daily_metrics(days: int = 90) -> pd.DataFrame:
    """Generate mock daily metrics with distinct patterns for each metric."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days), end=datetime.now(), freq="D"
    )

    data = []
    for i, date in enumerate(dates):
        progress = i / len(dates)  # 0 to 1 over the period
        weekend_factor = 0.8 if date.weekday() >= 5 else 1.0

        # Different patterns for each metric

        # 1. Daily active users: Steady growth with weekly pattern
        users_trend = 1 + progress * 0.4  # 40% growth
        users_seasonal = 1 + 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
        users_noise = np.random.normal(1, 0.08)
        users = int(10000 * users_trend * users_seasonal * weekend_factor * users_noise)

        # 2. Revenue: More volatile, some periods of decline
        revenue_cycles = np.sin(2 * np.pi * i / 45) * 0.2  # ~6 week cycles
        revenue_trend = 1 + progress * 0.2 - progress**2 * 0.1  # Growth then plateau
        revenue_volatility = np.random.normal(1, 0.15)  # More volatile
        revenue = (
            50000
            * revenue_trend
            * (1 + revenue_cycles)
            * weekend_factor
            * revenue_volatility
        )

        # 3. Conversions: Declining trend (common in maturing products)
        conv_decline = 1 - progress * 0.3  # 30% decline over period
        conv_noise = np.random.normal(1, 0.12)
        conversions = int(300 * conv_decline * weekend_factor * conv_noise)

        # 4. Conversion rate: Calculated but will show declining pattern
        conversion_rate = (conversions / users) * 100 if users > 0 else 0

        # 5. AOV: Stable with slight upward trend and less volatility
        aov_trend = 1 + progress * 0.15  # 15% growth
        aov_stability = np.random.normal(1, 0.05)  # Low volatility
        avg_order_value = (
            (revenue / conversions * aov_trend * aov_stability)
            if conversions > 0
            else 0
        )

        data.append(
            {
                "date": date,
                "daily_active_users": users,
                "revenue": revenue,
                "conversions": conversions,
                "conversion_rate": conversion_rate,
                "avg_order_value": avg_order_value,
            }
        )

    return pd.DataFrame(data)


@st.cache_data
def generate_channel_data(days: int = 90) -> pd.DataFrame:
    """Generate mock daily signups data for 10 different acquisition channels."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days), end=datetime.now(), freq="D"
    )

    # Define 10 different acquisition channels with realistic names
    channels = [
        "Google Ads",
        "Facebook Ads",
        "Organic Search",
        "Direct Traffic",
        "Email Marketing",
        "LinkedIn Ads",
        "Referrals",
        "YouTube Ads",
        "Twitter/X Ads",
        "Product Hunt",
    ]

    data = []
    for i, date in enumerate(dates):
        progress = i / len(dates)  # 0 to 1 over the period
        weekend_factor = 0.7 if date.weekday() >= 5 else 1.0

        for j, channel in enumerate(channels):
            # Each channel has different base performance and patterns
            channel_configs = {
                "Google Ads": {
                    "base": 150,
                    "growth": 0.3,
                    "volatility": 0.15,
                    "seasonal": 7,
                },
                "Facebook Ads": {
                    "base": 120,
                    "growth": 0.2,
                    "volatility": 0.20,
                    "seasonal": 14,
                },
                "Organic Search": {
                    "base": 200,
                    "growth": 0.5,
                    "volatility": 0.10,
                    "seasonal": 30,
                },
                "Direct Traffic": {
                    "base": 80,
                    "growth": 0.1,
                    "volatility": 0.08,
                    "seasonal": 365,
                },
                "Email Marketing": {
                    "base": 60,
                    "growth": 0.4,
                    "volatility": 0.25,
                    "seasonal": 7,
                },
                "LinkedIn Ads": {
                    "base": 40,
                    "growth": 0.6,
                    "volatility": 0.30,
                    "seasonal": 5,
                },
                "Referrals": {
                    "base": 35,
                    "growth": 0.8,
                    "volatility": 0.35,
                    "seasonal": 90,
                },
                "YouTube Ads": {
                    "base": 25,
                    "growth": 1.2,
                    "volatility": 0.40,
                    "seasonal": 14,
                },
                "Twitter/X Ads": {
                    "base": 20,
                    "growth": -0.2,
                    "volatility": 0.45,
                    "seasonal": 3,
                },
                "Product Hunt": {
                    "base": 15,
                    "growth": 0.9,
                    "volatility": 0.60,
                    "seasonal": 30,
                },
            }

            config = channel_configs[channel]

            # Apply different patterns for each channel
            growth_factor = 1 + progress * config["growth"]
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * i / config["seasonal"])
            noise_factor = np.random.normal(1, config["volatility"])

            # Some channels perform better on weekends, others worse
            if channel in ["Direct Traffic", "Organic Search"]:
                weekend_factor = 1.2 if date.weekday() >= 5 else 1.0

            signups = max(
                0,
                int(
                    config["base"]
                    * growth_factor
                    * seasonal_factor
                    * weekend_factor
                    * noise_factor
                ),
            )

            data.append({"date": date, "channel": channel, "signups": signups})

    return pd.DataFrame(data)


def create_simple_chart(
    df, x, y, chart_type="line", annotations=None, time_buttons=False, labels=None
):
    """Create simple, clean charts with good labels and tooltips."""
    chart_labels = labels or {}

    if chart_type == "line":
        fig = px.line(df, x=x, y=y, labels=chart_labels)
    elif chart_type == "bar":
        fig = px.bar(df, x=x, y=y, labels=chart_labels)
    elif chart_type == "area":
        fig = px.area(df, x=x, y=y, labels=chart_labels)
    else:
        fig = px.scatter(df, x=x, y=y, labels=chart_labels)

    # Clean layout - no title, no modebar
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)

    # Add time range buttons if requested
    if time_buttons:
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(
                                count=30, label="30D", step="day", stepmode="backward"
                            ),
                            dict(
                                count=90, label="90D", step="day", stepmode="backward"
                            ),
                            dict(
                                count=180, label="6M", step="day", stepmode="backward"
                            ),
                            dict(
                                count=365, label="1Y", step="day", stepmode="backward"
                            ),
                            dict(step="all", label="ALL"),
                        ]
                    ),
                    x=0.01,  # Position from left edge
                    y=1.02,  # Position above chart
                    xanchor="left",
                    yanchor="bottom",
                ),
                rangeslider=dict(visible=False),
                type="date",
            )
        )

    # Add annotations if provided
    if annotations:
        for annotation in annotations:
            fig.add_annotation(**annotation)

    return fig


def create_sparkline_chart(df, col, periods=30, rolling_window=7):
    """Create minimal sparkline chart with no axes or margins."""
    # Apply rolling average for smoother lines
    raw_data = df[col].tail(periods + rolling_window)
    smoothed_data = (
        raw_data.rolling(window=rolling_window, center=True).mean().tail(periods)
    )
    dates = df["date"].tail(periods)

    fig = px.line(x=dates, y=smoothed_data)

    # Minimal styling - no axes, no margins, no background
    fig.update_layout(
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
            visible=False,
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False,
            visible=False,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )

    # Format hover template based on column type
    if "revenue" in col or "value" in col:
        hover_format = "<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra></extra>"
    elif "rate" in col:
        hover_format = "<b>%{x|%b %d}</b><br>%{y:.1f}%<extra></extra>"
    else:
        hover_format = "<b>%{x|%b %d}</b><br>%{y:,.0f}<extra></extra>"

    fig.update_traces(hovertemplate=hover_format, line=dict(width=2))

    return fig


@st.fragment
def create_chart_segmented(chart_fig, df, sql_query, chart_id="chart"):
    """Create segmented control for chart, data, and SQL views using Material icons."""

    # Add CSS for better alignment
    st.html(f"""
    <style>
    .st-key-view-mode-{chart_id}, .st-key-date-filter-{chart_id} {{
        transform: scale(0.85);
        transform-origin: left center;
    }}
    
    .st-key-date-filter-{chart_id} {{
        transform: scale(0.85);
        transform-origin: right center;
        text-align: right;
        display: flex;
        justify-content: flex-end;
    }}
    
    .st-key-view-mode-{chart_id} > div, .st-key-date-filter-{chart_id} > div {{
        margin-bottom: 0 !important;
    }}
    </style>
    """)

    # Layout with right-aligned date filter
    col1, col2 = st.columns([1, 1])

    with col1:
        view_mode = st.segmented_control(
            "View mode",
            options=[
                ":material/show_chart:",
                ":material/data_table:",
                ":material/code:",
            ],
            default=":material/show_chart:",
            label_visibility="collapsed",
            key=f"view-mode-{chart_id}",
        )

    with col2:
        # Only show date filter when chart view is selected
        if view_mode == ":material/show_chart:":
            date_filter = st.segmented_control(
                "Time range",
                options=["30D", "90D", "6M", "1Y", "ALL"],
                default="ALL",
                label_visibility="collapsed",
                key=f"date-filter-{chart_id}",
            )
        else:
            date_filter = "ALL"

    if view_mode == ":material/show_chart:":
        # Apply date filtering to the chart
        filtered_fig = apply_date_filter(chart_fig, date_filter)

        st.plotly_chart(
            filtered_fig, use_container_width=True, config={"displayModeBar": False}
        )
    elif view_mode == ":material/data_table:":
        st.dataframe(df, use_container_width=True, height=DATAFRAME_HEIGHT)
    elif view_mode == ":material/code:":
        st.code(sql_query, language="sql")


def apply_date_filter(fig, date_filter):
    """Apply date filtering to plotly figure."""
    if date_filter == "ALL":
        return fig

    # Simple approach: calculate dates from current date
    # Use current date as end point
    end_date = datetime.now()

    # Calculate start date based on filter
    if date_filter == "30D":
        start_date = end_date - timedelta(days=30)
    elif date_filter == "90D":
        start_date = end_date - timedelta(days=90)
    elif date_filter == "6M":
        start_date = end_date - timedelta(days=180)
    elif date_filter == "1Y":
        start_date = end_date - timedelta(days=365)
    else:
        return fig

    # Create a copy and set the range using string format to avoid array issues
    fig_copy = copy.deepcopy(fig)
    fig_copy.update_layout(
        xaxis=dict(
            range=[start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
        )
    )
    return fig_copy
