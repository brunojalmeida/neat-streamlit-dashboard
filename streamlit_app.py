import numpy as np
import plotly.express as px
import streamlit as st

from utils import (
    CHART_HEIGHT,
    DATAFRAME_HEIGHT,
    create_chart_segmented,
    create_simple_chart,
    create_sparkline_chart,
    generate_channel_data,
    generate_daily_metrics,
)

st.set_page_config(
    page_title="Product analytics dashboard",
    page_icon=":material/dashboard:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.logo("https://1000logos.net/wp-content/uploads/2016/10/ACME-Logo-1984.png")


# Import page functions
def highlights_page():
    """Main highlights page with key metrics and charts."""
    st.title("Highlights")

    # Load full year of data - filtering will be done in charts
    df = generate_daily_metrics(365)

    # Key metrics in containers with borders
    st.subheader("Key performance indicators", divider="gray")

    # Current vs previous period calculations
    current_period = df.tail(30)
    previous_period = (
        df.tail(60).head(30) if len(df) > 30 else df.head(max(1, len(df) // 2))
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True, height=225):
            current_users = current_period["daily_active_users"].mean()
            previous_users = previous_period["daily_active_users"].mean()
            user_change = (
                ((current_users - previous_users) / previous_users) * 100
                if previous_users > 0
                else 0
            )
            st.metric(
                label="Average daily active users",
                value=f"{current_users:,.0f}",
                delta=f"{user_change:+.1f}%",
            )
            # Add sparkline
            sparkline_fig = create_sparkline_chart(
                df, "daily_active_users", rolling_window=5
            )
            st.plotly_chart(
                sparkline_fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with col2:
        with st.container(border=True, height=225):
            current_revenue = current_period["revenue"].sum()
            previous_revenue = previous_period["revenue"].sum()
            revenue_change = (
                ((current_revenue - previous_revenue) / previous_revenue) * 100
                if previous_revenue > 0
                else 0
            )
            st.metric(
                label="Monthly revenue",
                value=f"${current_revenue:,.0f}",
                delta=f"{revenue_change:+.1f}%",
            )
            # Add sparkline
            sparkline_fig = create_sparkline_chart(df, "revenue", rolling_window=10)
            st.plotly_chart(
                sparkline_fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with col3:
        with st.container(border=True, height=225):
            current_conv_rate = current_period["conversion_rate"].mean()
            previous_conv_rate = previous_period["conversion_rate"].mean()
            conv_change = current_conv_rate - previous_conv_rate
            st.metric(
                label="Conversion rate",
                value=f"{current_conv_rate:.2f}%",
                delta=f"{conv_change:+.2f}%",
            )
            # Add sparkline
            sparkline_fig = create_sparkline_chart(
                df, "conversion_rate", rolling_window=7
            )
            st.plotly_chart(
                sparkline_fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with col4:
        with st.container(border=True, height=225):
            # Revenue goal
            revenue_goal = 120000
            current_revenue = df["revenue"].iloc[-1]
            revenue_progress = (current_revenue / revenue_goal) * 100

            st.metric(label="Revenue goal progress", value=f"{revenue_progress:.1f}%")
            st.caption(f"{current_revenue:,.0f} / {revenue_goal:,.0f}")

            # Progress bar
            st.progress(min(revenue_progress / 100, 1.0))
            st.badge("On track to meet goal", icon=":material/check:")

        # Time series charts
    st.subheader("Time series analytics", divider="gray")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True, height=CHART_HEIGHT):
            st.markdown("**Daily active users over time**")

            # Apply rolling average for smoother time series
            df_smooth = df.copy()
            df_smooth["daily_active_users"] = (
                df["daily_active_users"].rolling(window=7, center=True).mean()
            )

            # Add meaningful business event annotations
            # Pick some strategic dates for annotations
            product_hunt_date = df_smooth["date"].iloc[len(df_smooth) // 3]
            product_hunt_users = df_smooth.loc[
                df_smooth["date"] == product_hunt_date, "daily_active_users"
            ].iloc[0]

            feature_launch_date = df_smooth["date"].iloc[2 * len(df_smooth) // 3]
            feature_launch_users = df_smooth.loc[
                df_smooth["date"] == feature_launch_date, "daily_active_users"
            ].iloc[0]

            annotations = [
                {
                    "x": product_hunt_date,
                    "y": product_hunt_users,
                    "text": "🚀 Launched on Product Hunt",
                    "showarrow": True,
                    "arrowhead": 2,
                    "arrowsize": 1,
                    "arrowwidth": 2,
                    "arrowcolor": "#333",
                    "bgcolor": "#ffebeb",  # Light pink, full opacity
                    "font": {"color": "#333", "size": 12},
                },
                {
                    "x": feature_launch_date,
                    "y": feature_launch_users,
                    "text": "✨ New AI features released",
                    "showarrow": True,
                    "arrowhead": 2,
                    "arrowsize": 1,
                    "arrowwidth": 2,
                    "arrowcolor": "#333",
                    "bgcolor": "#e8f5e8",  # Light green, full opacity
                    "font": {"color": "#333", "size": 12},
                },
            ]

            fig_users = create_simple_chart(
                df_smooth,
                "date",
                "daily_active_users",
                time_buttons=False,  # Handled by segmented control now
                annotations=annotations,
                labels={"date": "Date", "daily_active_users": "Daily active users"},
            )

            # Mock SQL query
            users_sql = """
SELECT 
    date,
    COUNT(DISTINCT user_id) as daily_active_users
FROM user_activity 
WHERE date >= CURRENT_DATE - INTERVAL '1 YEAR'
GROUP BY date
ORDER BY date
            """

            create_chart_segmented(
                fig_users, df[["date", "daily_active_users"]], users_sql, "users"
            )

    with col2:
        with st.container(border=True, height=CHART_HEIGHT):
            st.markdown("**Daily revenue over time**")

            # Apply rolling average for smoother time series
            df_smooth_revenue = df.copy()
            df_smooth_revenue["revenue"] = (
                df["revenue"].rolling(window=7, center=True).mean()
            )

            # Add meaningful business event annotations
            # Black Friday campaign
            black_friday_date = df_smooth_revenue["date"].iloc[
                len(df_smooth_revenue) // 4
            ]
            black_friday_revenue = df_smooth_revenue.loc[
                df_smooth_revenue["date"] == black_friday_date, "revenue"
            ].iloc[0]

            # Enterprise client signed
            enterprise_date = df_smooth_revenue["date"].iloc[
                3 * len(df_smooth_revenue) // 4
            ]
            enterprise_revenue = df_smooth_revenue.loc[
                df_smooth_revenue["date"] == enterprise_date, "revenue"
            ].iloc[0]

            annotations = [
                {
                    "x": black_friday_date,
                    "y": black_friday_revenue,
                    "text": "🛍️ Black Friday campaign",
                    "showarrow": True,
                    "arrowhead": 2,
                    "arrowsize": 1,
                    "arrowwidth": 2,
                    "arrowcolor": "#333",
                    "bgcolor": "#fffaeb",  # Light yellow, full opacity
                    "font": {"color": "#333", "size": 12},
                },
                {
                    "x": enterprise_date,
                    "y": enterprise_revenue,
                    "text": "🏢 Major enterprise client signed",
                    "showarrow": True,
                    "arrowhead": 2,
                    "arrowsize": 1,
                    "arrowwidth": 2,
                    "arrowcolor": "#333",
                    "bgcolor": "#e6f7ff",  # Light blue, full opacity
                    "font": {"color": "#333", "size": 12},
                },
            ]

            fig_revenue = create_simple_chart(
                df_smooth_revenue,
                "date",
                "revenue",
                time_buttons=False,  # Handled by segmented control now
                annotations=annotations,
                labels={"date": "Date", "revenue": "Revenue ($)"},
            )

            # Mock SQL query
            revenue_sql = """
SELECT 
    date,
    SUM(order_amount) as revenue
FROM orders 
WHERE date >= CURRENT_DATE - INTERVAL '1 YEAR'
GROUP BY date
ORDER BY date
            """

            create_chart_segmented(
                fig_revenue, df[["date", "revenue"]], revenue_sql, "revenue"
            )

    # Multi-line interactive chart
    st.subheader("User acquisition channels", divider="gray")

    with st.container(border=True, height=CHART_HEIGHT):
        st.markdown("**Daily signups by acquisition channel**")

        # Generate multi-channel data
        channels_df = generate_channel_data(365)

        # Create multi-line chart
        fig_channels = px.line(
            channels_df,
            x="date",
            y="signups",
            color="channel",
            labels={"date": "Date", "signups": "Daily signups", "channel": "Channel"},
        )

        # Style for better legend interaction
        fig_channels.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.8)",
            ),
            hovermode="x unified",
        )

        # Mock SQL query for channels
        channels_sql = """
SELECT 
    date,
    acquisition_channel as channel,
    COUNT(DISTINCT user_id) as signups
FROM user_signups 
WHERE date >= CURRENT_DATE - INTERVAL '1 YEAR'
GROUP BY date, acquisition_channel
ORDER BY date, channel
        """

        create_chart_segmented(fig_channels, channels_df, channels_sql, "channels")

    # Growth summary table
    st.subheader("Growth summary", divider="gray")

    with st.container(border=True, height=DATAFRAME_HEIGHT):
        # Create weekly summary
        df["week"] = df["date"].dt.to_period("W")
        weekly_summary = (
            df.groupby("week")
            .agg(
                {
                    "daily_active_users": "mean",
                    "revenue": "sum",
                }
            )
            .round(2)
        )

        # Calculate week-over-week growth
        weekly_summary["revenue_growth"] = weekly_summary["revenue"].pct_change() * 100

        # Add additional columns for enhanced display
        weekly_summary["health_score"] = np.clip(
            weekly_summary["revenue_growth"].fillna(0) + 50,
            0,
            100,
        )

        # Add performance status
        weekly_summary["status"] = weekly_summary["health_score"].apply(
            lambda x: "🟢 Excellent"
            if x >= 75
            else "🟡 Good"
            if x >= 50
            else "🔴 Needs attention"
        )

        # Create daily revenue arrays for AreaChartColumn with enhanced variance
        def create_daily_revenue_pattern(week_period):
            week_data = df[df["week"] == week_period]["revenue"].values
            if len(week_data) == 0:
                return []

            # Create a more interesting daily pattern within the week
            base_revenue = week_data.mean() / 7  # Average daily for the week

            # Much more dramatic daily patterns for better visual impact
            daily_multipliers = [0.3, 0.7, 1.8, 2.1, 1.5, 0.2, 0.1]  # Mon-Sun pattern

            # Add significant randomness for dramatic visual effect
            np.random.seed(
                hash(str(week_period)) % 1000
            )  # Consistent but varied per week
            noise = np.random.normal(1, 0.6, 7)  # 60% variance for dramatic effect

            daily_revenues = [
                int(max(base_revenue * 0.05, base_revenue * mult * noise_val))
                for mult, noise_val in zip(daily_multipliers, noise)
            ]

            return daily_revenues[: len(week_data)]  # Match actual days in week

        weekly_summary["daily_revenue_chart"] = weekly_summary.index.map(
            create_daily_revenue_pattern
        )

        # Add report links (mock)
        weekly_summary["report_link"] = weekly_summary.index.map(
            lambda x: f"https://dashboard.example.com/reports/{x}"
        )

        # Format the data
        weekly_summary = weekly_summary.tail(8)  # Last 8 weeks
        weekly_summary.index = weekly_summary.index.astype(str)
        weekly_summary.index.name = "Week"

        # Display with enhanced column configuration
        st.dataframe(
            weekly_summary,
            column_config={
                "daily_active_users": st.column_config.NumberColumn(
                    "Average daily users",
                    format="%.0f",
                    help="Average daily active users for the week",
                ),
                "revenue": st.column_config.NumberColumn(
                    "Weekly revenue",
                    format="$%.0f",
                    help="Total revenue for the week",
                ),
                "daily_revenue_chart": st.column_config.AreaChartColumn(
                    "Daily revenue trend",
                    help="Daily revenue pattern within the week",
                ),
                "revenue_growth": st.column_config.NumberColumn(
                    "Revenue growth",
                    format="%.1f%%",
                    help="Week-over-week growth in revenue",
                ),
                "health_score": st.column_config.ProgressColumn(
                    "Health score",
                    help="Overall performance score (0-100)",
                    min_value=0,
                    max_value=100,
                    format="%.0f",
                ),
                "status": st.column_config.TextColumn(
                    "Status", help="Performance status based on health score"
                ),
                "report_link": st.column_config.LinkColumn(
                    "Detailed report",
                    help="Link to detailed weekly report",
                    display_text="View report",
                ),
            },
            use_container_width=True,
            hide_index=False,
        )


def customers_page():
    """Customers page."""
    st.title(":material/people: Customers")
    st.info("Customer analytics and insights coming soon...")


def retention_page():
    """Retention page."""
    st.title(":material/repeat: Retention")
    st.info("Retention analysis and cohort insights coming soon...")


def forecasts_page():
    """Forecasts page."""
    st.title(":material/trending_up: Forecasts")
    st.info("Predictive analytics and forecasting models coming soon...")


def settings_page():
    """Settings page."""
    st.title(":material/settings: Settings")
    st.info("Application settings and configuration coming soon...")


# Navigation
highlights = st.Page(
    highlights_page, title="Highlights", icon=":material/star:", default=True
)
customers = st.Page(customers_page, title="Customers", icon=":material/people:")
retention = st.Page(retention_page, title="Retention", icon=":material/repeat:")
forecasts = st.Page(forecasts_page, title="Forecasts", icon=":material/trending_up:")
settings = st.Page(settings_page, title="Settings", icon=":material/settings:")

pg = st.navigation(
    [highlights, customers, retention, forecasts, settings], position="top"
)


# Run the page
pg.run()
