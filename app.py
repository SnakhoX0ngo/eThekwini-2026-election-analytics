# ============================================================
# eThekwini 2026 Local Government Election Analytics
# Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="eThekwini 2026 Election Analytics",
    page_icon="🗳️",
    layout="wide"
)

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "eThekwini 2026 Election Analytics"
MUNICIPALITY = "eThekwini"
ELECTION_DATE = "04 November 2026"

HISTORICAL_YEARS = [2011, 2016, 2021]

DATA_DIR = Path("Datasets")

FILE_2011 = DATA_DIR / "2011_ETH.csv"
FILE_2016 = DATA_DIR / "2016_ETH.csv"
FILE_2021 = DATA_DIR / "2021_ETH.csv"

# ============================================================
# PAGE TITLE
# ============================================================

st.title("🗳️ eThekwini 2026 Election Analytics")

st.markdown(
    """
    ### South African Local Government Election Analytics

    This dashboard analyses historical eThekwini Metropolitan Municipality
    election results from **2011, 2016 and 2021** and presents model-based
    projections for the **04 November 2026 Local Government Election**.

    > ⚠️ **Important:** 2026 figures shown in this dashboard are statistical
    > model estimates, not official election results.
    """
)

st.divider()


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    df_2011 = pd.read_csv(FILE_2011, encoding="utf-16")
    df_2016 = pd.read_csv(FILE_2016, encoding="utf-16")
    df_2021 = pd.read_csv(FILE_2021, encoding="utf-16")

    df_2011["ElectionYear"] = 2011
    df_2016["ElectionYear"] = 2016
    df_2021["ElectionYear"] = 2021

    df = pd.concat(
        [df_2011, df_2016, df_2021],
        ignore_index=True
    )

    # Standardise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^A-Za-z0-9_]", "", regex=True)
    )

    # Clean text columns
    text_columns = df.select_dtypes(include=["object"]).columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    # Convert numeric columns
    numeric_columns = [
        "RegisteredVoters",
        "SpoiltVotes",
        "TotalValidVotes"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


try:
    historical_df = load_data()

except Exception as e:

    st.error("Could not load the election datasets.")

    st.code(
        """
Make sure your GitHub project has this structure:

eThekwini-Election-Analytics/
│
├── app.py
│
├── Datasets/
│   ├── 2011_ETH.csv
│   ├── 2016_ETH.csv
│   └── 2021_ETH.csv
        """
    )

    st.exception(e)
    st.stop()


# ============================================================
# CREATE PARTY-LEVEL DATA
# ============================================================

@st.cache_data
def prepare_party_data(df):

    party_votes = (
        df.groupby(
            [
                "ElectionYear",
                "Ward",
                "VotingDistrict",
                "BallotType",
                "PartyName"
            ],
            as_index=False
        )["TotalValidVotes"]
        .sum()
    )

    return party_votes


party_votes = prepare_party_data(historical_df)


# ============================================================
# METRO PR RESULTS
# ============================================================

@st.cache_data
def calculate_metro_results(party_votes):

    pr_votes = party_votes[
        party_votes["BallotType"] == "PR"
    ].copy()

    municipal_party_votes = (
        pr_votes
        .groupby(
            ["ElectionYear", "PartyName"],
            as_index=False
        )["TotalValidVotes"]
        .sum()
    )

    municipal_totals = (
        municipal_party_votes
        .groupby("ElectionYear")["TotalValidVotes"]
        .sum()
        .reset_index(
            name="MunicipalPRValidVotes"
        )
    )

    municipal_party_votes = municipal_party_votes.merge(
        municipal_totals,
        on="ElectionYear",
        how="left"
    )

    municipal_party_votes["VoteShare"] = (
        municipal_party_votes["TotalValidVotes"]
        /
        municipal_party_votes["MunicipalPRValidVotes"]
        * 100
    )

    return municipal_party_votes


municipal_pr_party_votes = calculate_metro_results(
    party_votes
)


# ============================================================
# TOP PARTIES
# ============================================================

@st.cache_data
def create_party_dataset(municipal_pr_party_votes):

    top_2021_parties = (
        municipal_pr_party_votes[
            municipal_pr_party_votes["ElectionYear"] == 2021
        ]
        .sort_values(
            "TotalValidVotes",
            ascending=False
        )
        .head(10)["PartyName"]
        .tolist()
    )

    selected_data = (
        municipal_pr_party_votes[
            municipal_pr_party_votes["PartyName"]
            .isin(top_2021_parties)
        ]
        .pivot_table(
            index="ElectionYear",
            columns="PartyName",
            values="VoteShare",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    selected_data["OTHER PARTIES"] = (
        100
        -
        selected_data[top_2021_parties].sum(axis=1)
    )

    return selected_data, top_2021_parties


selected_party_data, top_2021_parties = create_party_dataset(
    municipal_pr_party_votes
)


# ============================================================
# METRO LINEAR REGRESSION
# ============================================================

@st.cache_data
def metro_forecast(selected_party_data):

    model_columns = [
        column
        for column in selected_party_data.columns
        if column != "ElectionYear"
    ]

    X = selected_party_data[
        ["ElectionYear"]
    ]

    # --------------------------------------------------------
    # Temporal validation
    # Train: 2011 + 2016
    # Test: 2021
    # --------------------------------------------------------

    train = selected_party_data[
        selected_party_data["ElectionYear"] < 2021
    ]

    test = selected_party_data[
        selected_party_data["ElectionYear"] == 2021
    ]

    validation_predictions = []
    validation_actuals = []

    for party in model_columns:

        model = LinearRegression()

        model.fit(
            train[["ElectionYear"]],
            train[[party]]
        )

        prediction = model.predict(
            test[["ElectionYear"]]
        )[0][0]

        actual = test[party].iloc[0]

        validation_predictions.append(
            prediction
        )

        validation_actuals.append(
            actual
        )

    validation_mae = mean_absolute_error(
        validation_actuals,
        validation_predictions
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            validation_actuals,
            validation_predictions
        )
    )

    # --------------------------------------------------------
    # Final models
    # Train using all available historical elections
    # --------------------------------------------------------

    forecast = {}

    for party in model_columns:

        model = LinearRegression()

        model.fit(
            X,
            selected_party_data[[party]]
        )

        predicted_share = model.predict(
            np.array([[2026]])
        )[0][0]

        # Vote shares cannot be negative
        predicted_share = max(
            0,
            predicted_share
        )

        forecast[party] = predicted_share

    # Normalise to 100%
    total_forecast = sum(
        forecast.values()
    )

    for party in forecast:

        forecast[party] = (
            forecast[party]
            /
            total_forecast
            * 100
        )

    forecast_df = pd.DataFrame(
        {
            "PartyName": list(forecast.keys()),
            "ProjectedVoteShare": list(
                forecast.values()
            )
        }
    ).sort_values(
        "ProjectedVoteShare",
        ascending=False
    )

    return (
        forecast_df,
        validation_mae,
        validation_rmse
    )


(
    metro_forecast_df,
    metro_mae,
    metro_rmse
) = metro_forecast(
    selected_party_data
)


# ============================================================
# WARD-LEVEL DATA
# ============================================================

@st.cache_data
def calculate_ward_results(party_votes):

    ward_votes = party_votes[
        party_votes["BallotType"] == "Ward"
    ].copy()

    ward_party_votes = (
        ward_votes
        .groupby(
            [
                "ElectionYear",
                "Ward",
                "PartyName"
            ],
            as_index=False
        )["TotalValidVotes"]
        .sum()
    )

    ward_totals = (
        ward_party_votes
        .groupby(
            [
                "ElectionYear",
                "Ward"
            ]
        )["TotalValidVotes"]
        .sum()
        .reset_index(
            name="WardValidVotes"
        )
    )

    ward_party_votes = ward_party_votes.merge(
        ward_totals,
        on=[
            "ElectionYear",
            "Ward"
        ],
        how="left"
    )

    ward_party_votes["VoteShare"] = (
        ward_party_votes["TotalValidVotes"]
        /
        ward_party_votes["WardValidVotes"]
        * 100
    )

    ward_winners = (
        ward_party_votes
        .sort_values(
            [
                "ElectionYear",
                "Ward",
                "TotalValidVotes"
            ],
            ascending=[
                True,
                True,
                False
            ]
        )
        .groupby(
            [
                "ElectionYear",
                "Ward"
            ]
        )
        .first()
        .reset_index()
    )

    return (
        ward_party_votes,
        ward_winners
    )


(
    ward_party_votes,
    ward_winners
) = calculate_ward_results(
    party_votes
)


# ============================================================
# GEOGRAPHIC CONTINUITY
# ============================================================

@st.cache_data
def calculate_ward_continuity(df):

    ward_vd_sets = {}

    for year in HISTORICAL_YEARS:

        year_df = df[
            (df["ElectionYear"] == year)
            &
            (df["BallotType"] == "Ward")
        ]

        ward_vd_sets[year] = (
            year_df
            .groupby("Ward")["VotingDistrict"]
            .apply(set)
            .to_dict()
        )

    continuity_records = []

    common_wards = (
        set(ward_vd_sets[2011].keys())
        &
        set(ward_vd_sets[2016].keys())
        &
        set(ward_vd_sets[2021].keys())
    )

    for ward in common_wards:

        set_2011 = ward_vd_sets[2011][ward]
        set_2016 = ward_vd_sets[2016][ward]
        set_2021 = ward_vd_sets[2021][ward]

        intersection_1 = len(
            set_2011 & set_2016
        )

        union_1 = len(
            set_2011 | set_2016
        )

        intersection_2 = len(
            set_2016 & set_2021
        )

        union_2 = len(
            set_2016 | set_2021
        )

        jaccard_2011_2016 = (
            intersection_1 / union_1
            if union_1 > 0 else 0
        )

        jaccard_2016_2021 = (
            intersection_2 / union_2
            if union_2 > 0 else 0
        )

        average_continuity = (
            jaccard_2011_2016
            +
            jaccard_2016_2021
        ) / 2

        continuity_records.append(
            {
                "Ward": ward,
                "Jaccard_2011_2016":
                    jaccard_2011_2016,
                "Jaccard_2016_2021":
                    jaccard_2016_2021,
                "AverageContinuity":
                    average_continuity
            }
        )

    continuity_df = pd.DataFrame(
        continuity_records
    )

    return continuity_df.sort_values(
        "AverageContinuity",
        ascending=False
    )


continuity_df = calculate_ward_continuity(
    historical_df
)


# ============================================================
# SELECT THREE GEOGRAPHICALLY STABLE WARDS
# ============================================================

selected_wards = (
    continuity_df
    .head(3)["Ward"]
    .tolist()
)


# ============================================================
# RANDOM FOREST WARD CLASSIFICATION
# ============================================================

@st.cache_data
def build_ward_classifier(
    ward_winners,
    selected_wards
):

    # --------------------------------------------------------
    # Create transition dataset
    # --------------------------------------------------------

    transitions = []

    for previous_year, next_year in [
        (2011, 2016),
        (2016, 2021)
    ]:

        previous = ward_winners[
            ward_winners["ElectionYear"]
            == previous_year
        ][
            [
                "Ward",
                "PartyName",
                "VoteShare"
            ]
        ].copy()

        previous = previous.rename(
            columns={
                "PartyName":
                    "PreviousWinner",
                "VoteShare":
                    "PreviousWinnerShare"
            }
        )

        next_results = ward_winners[
            ward_winners["ElectionYear"]
            == next_year
        ][
            [
                "Ward",
                "PartyName"
            ]
        ].copy()

        next_results = next_results.rename(
            columns={
                "PartyName":
                    "NextWinner"
            }
        )

        transition = previous.merge(
            next_results,
            on="Ward",
            how="inner"
        )

        transition["PreviousYear"] = (
            previous_year
        )

        transition["NextYear"] = (
            next_year
        )

        transitions.append(
            transition
        )

    transition_df = pd.concat(
        transitions,
        ignore_index=True
    )

    # Only selected wards for validation/model output
    transition_df = transition_df[
        transition_df["Ward"].isin(
            selected_wards
        )
    ].copy()

    # --------------------------------------------------------
    # One-hot encode previous winner
    # --------------------------------------------------------

    features = pd.get_dummies(
        transition_df[
            [
                "PreviousWinner",
                "PreviousWinnerShare"
            ]
        ],
        columns=["PreviousWinner"]
    )

    target = transition_df[
        "NextWinner"
    ]

    # --------------------------------------------------------
    # Temporal validation
    # Train 2011 -> 2016
    # Test 2016 -> 2021
    # --------------------------------------------------------

    train_mask = (
        transition_df["PreviousYear"] == 2011
    )

    test_mask = (
        transition_df["PreviousYear"] == 2016
    )

    X_train = features[
        train_mask
    ]

    y_train = target[
        train_mask
    ]

    X_test = features[
        test_mask
    ]

    y_test = target[
        test_mask
    ]

    # Ensure same columns
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    validation_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    validation_model.fit(
        X_train,
        y_train
    )

    y_pred = validation_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=sorted(
            target.unique()
        )
    )

    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    final_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    final_model.fit(
        features,
        target
    )

    # --------------------------------------------------------
    # 2026 input
    # --------------------------------------------------------

    latest = ward_winners[
        (
            ward_winners["ElectionYear"]
            == 2021
        )
        &
        (
            ward_winners["Ward"]
            .isin(selected_wards)
        )
    ][
        [
            "Ward",
            "PartyName",
            "VoteShare"
        ]
    ].copy()

    latest = latest.rename(
        columns={
            "PartyName":
                "PreviousWinner",
            "VoteShare":
                "PreviousWinnerShare"
        }
    )

    future_features = pd.get_dummies(
        latest[
            [
                "PreviousWinner",
                "PreviousWinnerShare"
            ]
        ],
        columns=["PreviousWinner"]
    )

    future_features = future_features.reindex(
        columns=features.columns,
        fill_value=0
    )

    predictions = final_model.predict(
        future_features
    )

    latest["ProjectedWinner"] = predictions

    return (
        latest,
        accuracy,
        precision,
        recall,
        f1,
        cm,
        sorted(target.unique())
    )


(
    ward_predictions,
    ward_accuracy,
    ward_precision,
    ward_recall,
    ward_f1,
    ward_cm,
    ward_classes
) = build_ward_classifier(
    ward_winners,
    selected_wards
)


# ============================================================
# TURNOUT / PARTICIPATION PROXY
# ============================================================

@st.cache_data
def calculate_turnout_proxy(df):

    turnout_records = []

    for year in HISTORICAL_YEARS:

        year_df = df[
            (df["ElectionYear"] == year)
            &
            (df["BallotType"] == "PR")
        ].copy()

        # Registered voters and spoilt votes repeat
        # across party rows, therefore deduplicate
        # at voting-district level.
        vd_info = (
            year_df[
                [
                    "VotingDistrict",
                    "RegisteredVoters",
                    "SpoiltVotes"
                ]
            ]
            .drop_duplicates(
                subset=["VotingDistrict"]
            )
        )

        registered = vd_info[
            "RegisteredVoters"
        ].sum()

        spoilt = vd_info[
            "SpoiltVotes"
        ].sum()

        valid_votes = year_df[
            "TotalValidVotes"
        ].sum()

        participation = (
            (valid_votes + spoilt)
            /
            registered
            * 100
            if registered > 0
            else np.nan
        )

        turnout_records.append(
            {
                "ElectionYear": year,
                "RegisteredVoters": registered,
                "ValidVotes": valid_votes,
                "SpoiltVotes": spoilt,
                "ParticipationProxy":
                    participation
            }
        )

    return pd.DataFrame(
        turnout_records
    )


turnout_df = calculate_turnout_proxy(
    historical_df
)


# ============================================================
# TURNOUT FORECAST
# ============================================================

turnout_model = LinearRegression()

turnout_model.fit(
    turnout_df[["ElectionYear"]],
    turnout_df["ParticipationProxy"]
)

projected_turnout = turnout_model.predict(
    np.array([[2026]])
)[0]

projected_turnout = np.clip(
    projected_turnout,
    0,
    100
)

# Forecast registered voters
registered_model = LinearRegression()

registered_model.fit(
    turnout_df[["ElectionYear"]],
    turnout_df["RegisteredVoters"]
)

projected_registered_voters = registered_model.predict(
    np.array([[2026]])
)[0]

projected_registered_voters = max(
    0,
    projected_registered_voters
)

estimated_2026_voters = (
    projected_registered_voters
    *
    projected_turnout
    /
    100
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Controls")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Historical Results",
        "2026 Metro Projection",
        "Ward Projections",
        "Model Performance",
        "Turnout",
        "Sources & Limitations"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    projected_leader = (
        metro_forecast_df.iloc[0]
    )

    col1.metric(
        "Projected 2026 Leader",
        projected_leader["PartyName"]
    )

    col2.metric(
        "Projected Vote Share",
        f"{projected_leader['ProjectedVoteShare']:.2f}%"
    )

    col3.metric(
        "Metro Validation MAE",
        f"{metro_mae:.2f} pp"
    )

    col4.metric(
        "Metro Validation RMSE",
        f"{metro_rmse:.2f} pp"
    )

    st.divider()

    st.subheader("2026 Metro Projection")

    st.dataframe(
        metro_forecast_df.style.format(
            {
                "ProjectedVoteShare":
                    "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        The projected leading party has the largest estimated PR vote share.
        A vote share below 50% may suggest that coalition formation could be
        relevant, but council control cannot be determined from vote share
        alone.
        """
    )

    st.subheader("Selected Ward Projections")

    display_wards = ward_predictions.copy()

    display_wards["PreviousWinnerShare"] = (
        display_wards["PreviousWinnerShare"]
        .round(2)
    )

    st.dataframe(
        display_wards[
            [
                "Ward",
                "PreviousWinner",
                "PreviousWinnerShare",
                "ProjectedWinner"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# HISTORICAL RESULTS
# ============================================================

elif page == "Historical Results":

    st.header("Historical Election Results")

    selected_year = st.selectbox(
        "Select Election Year",
        HISTORICAL_YEARS
    )

    historical_year = (
        municipal_pr_party_votes[
            municipal_pr_party_votes["ElectionYear"]
            == selected_year
        ]
        .sort_values(
            "VoteShare",
            ascending=False
        )
    )

    st.subheader(
        f"{selected_year} PR Vote Share"
    )

    chart_data = historical_year[
        [
            "PartyName",
            "VoteShare"
        ]
    ].set_index("PartyName")

    st.bar_chart(
        chart_data
    )

    st.subheader(
        f"{selected_year} Party Results"
    )

    st.dataframe(
        historical_year[
            [
                "PartyName",
                "TotalValidVotes",
                "VoteShare"
            ]
        ].style.format(
            {
                "TotalValidVotes": "{:,.0f}",
                "VoteShare": "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Historical Vote-Share Trend"
    )

    party_options = [
        "All Major Parties"
    ] + top_2021_parties

    selected_party = st.selectbox(
        "Select Party",
        party_options
    )

    if selected_party == "All Major Parties":

        trend = selected_party_data.set_index(
            "ElectionYear"
        )

        st.line_chart(
            trend
        )

    else:

        trend = selected_party_data[
            [
                "ElectionYear",
                selected_party
            ]
        ].set_index(
            "ElectionYear"
        )

        st.line_chart(
            trend
        )


# ============================================================
# 2026 METRO PROJECTION
# ============================================================

elif page == "2026 Metro Projection":

    st.header(
        "Projected 2026 eThekwini Metro Results"
    )

    st.warning(
        "These are model estimates, not official 2026 election results."
    )

    leader = metro_forecast_df.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Projected Leader",
        leader["PartyName"]
    )

    col2.metric(
        "Projected Share",
        f"{leader['ProjectedVoteShare']:.2f}%"
    )

    col3.metric(
        "Number of Parties",
        len(metro_forecast_df)
    )

    st.divider()

    st.subheader(
        "Projected Party Vote Shares"
    )

    chart_data = (
        metro_forecast_df
        .set_index("PartyName")
    )

    st.bar_chart(
        chart_data[
            ["ProjectedVoteShare"]
        ]
    )

    st.subheader(
        "Projected 2026 Party Results"
    )

    st.dataframe(
        metro_forecast_df.style.format(
            {
                "ProjectedVoteShare":
                    "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Historical vs Projected Results"
    )

    comparison = selected_party_data.copy()

    projection_row = {
        "ElectionYear": 2026
    }

    for party in comparison.columns:

        if party == "ElectionYear":
            continue

        value = metro_forecast_df.loc[
            metro_forecast_df["PartyName"]
            == party,
            "ProjectedVoteShare"
        ]

        projection_row[party] = (
            value.iloc[0]
            if not value.empty
            else 0
        )

    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame([projection_row])
        ],
        ignore_index=True
    )

    comparison = comparison.set_index(
        "ElectionYear"
    )

    st.line_chart(
        comparison
    )

    st.subheader(
        "Coalition Interpretation"
    )

    if leader["ProjectedVoteShare"] < 50:

        st.info(
            f"""
            The model projects {leader['PartyName']} as the largest party
            with approximately {leader['ProjectedVoteShare']:.2f}% of the
            PR vote.

            Because this is below 50%, the result may indicate that coalition
            arrangements could be relevant. However, this does **not** mean
            that the model predicts who will govern eThekwini. Council
            composition depends on the electoral system, seat allocation and
            post-election political agreements.
            """
        )

    else:

        st.info(
            f"""
            The model projects {leader['PartyName']} above 50% of the
            estimated PR vote share. This is still a model estimate and
            should not be interpreted as an official election result or
            guaranteed governing outcome.
            """
        )


# ============================================================
# WARD PROJECTIONS
# ============================================================

elif page == "Ward Projections":

    st.header(
        "Selected Ward Projections"
    )

    st.markdown(
        """
        Three wards were selected using historical voting-district
        continuity across the 2011, 2016 and 2021 elections.

        Higher Jaccard similarity indicates greater geographic continuity
        between election periods.
        """
    )

    st.subheader(
        "Ward Geographic Continuity"
    )

    st.dataframe(
        continuity_df.head(10).style.format(
            {
                "Jaccard_2011_2016":
                    "{:.3f}",
                "Jaccard_2016_2021":
                    "{:.3f}",
                "AverageContinuity":
                    "{:.3f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "2026 Ward Predictions"
    )

    ward_display = ward_predictions.copy()

    ward_display["PreviousWinnerShare"] = (
        ward_display["PreviousWinnerShare"]
        .round(2)
    )

    st.dataframe(
        ward_display[
            [
                "Ward",
                "PreviousWinner",
                "PreviousWinnerShare",
                "ProjectedWinner"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    selected_ward = st.selectbox(
        "Select a Ward",
        selected_wards
    )

    ward_info = ward_predictions[
        ward_predictions["Ward"]
        == selected_ward
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Ward",
        selected_ward
    )

    col2.metric(
        "2021 Winner",
        ward_info["PreviousWinner"]
    )

    col3.metric(
        "Projected 2026 Winner",
        ward_info["ProjectedWinner"]
    )

    st.metric(
        "2021 Winner Vote Share",
        f"{ward_info['PreviousWinnerShare']:.2f}%"
    )

    st.warning(
        """
        Ward projections are conditional on the historical ward identifiers
        remaining sufficiently comparable. eThekwini's ward structure has
        changed over time, so these results should not be interpreted as
        proof that historical and 2026 ward boundaries are identical.
        """
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header(
        "Model Performance & Diagnostics"
    )

    # --------------------------------------------------------
    # Metro model
    # --------------------------------------------------------

    st.subheader(
        "Metro Vote-Share Model"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Validation MAE",
        f"{metro_mae:.2f} percentage points"
    )

    col2.metric(
        "Validation RMSE",
        f"{metro_rmse:.2f} percentage points"
    )

    st.markdown(
        """
        **Validation approach:** the model was trained on the 2011 and
        2016 elections and evaluated on the 2021 election.

        The final forecasting model was then trained using all three
        historical elections before producing the 2026 estimate.
        """
    )

    # --------------------------------------------------------
    # Ward model
    # --------------------------------------------------------

    st.subheader(
        "Ward Winner Classification Model"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{ward_accuracy:.3f}"
    )

    col2.metric(
        "Precision",
        f"{ward_precision:.3f}"
    )

    col3.metric(
        "Recall",
        f"{ward_recall:.3f}"
    )

    col4.metric(
        "F1 Score",
        f"{ward_f1:.3f}"
    )

    st.subheader(
        "Confusion Matrix"
    )

    confusion_df = pd.DataFrame(
        ward_cm,
        index=[
            f"Actual: {x}"
            for x in ward_classes
        ],
        columns=[
            f"Predicted: {x}"
            for x in ward_classes
        ]
    )

    st.dataframe(
        confusion_df,
        use_container_width=True
    )

    st.markdown(
        """
        ### Model limitations

        - Only three historical election cycles are available.
        - Political behaviour can change substantially between elections.
        - The ward classifier uses a small historical sample.
        - Historical ward boundaries are not guaranteed to be identical to
          the 2026 boundaries.
        - A high classification score would not guarantee correct 2026
          predictions.
        """
    )

    st.subheader(
        "Algorithm Complexity"
    )

    st.markdown(
        """
        **Linear Regression**

        With a small number of observations and features, the computational
        cost is very low. Training is approximately linear in the number of
        observations for this application.

        **Random Forest**

        Training complexity is approximately related to:

        `O(T × n × log(n) × p)`

        where:

        - `T` = number of trees
        - `n` = number of training observations
        - `p` = number of features

        Because the dataset is relatively small, computational complexity
        is not a practical limitation for this project.
        """
    )


# ============================================================
# TURNOUT
# ============================================================

elif page == "Turnout":

    st.header(
        "Turnout & Voter Participation"
    )

    st.warning(
        """
        The turnout calculation shown here is a **PR-ballot participation
        proxy** derived from the available IEC election records. It should
        not be presented as an official IEC turnout figure unless separately
        verified against the official IEC turnout publication.
        """
    )

    st.subheader(
        "Historical Participation"
    )

    turnout_display = turnout_df.copy()

    turnout_display[
        "ParticipationProxy"
    ] = turnout_display[
        "ParticipationProxy"
    ].round(2)

    st.dataframe(
        turnout_display.style.format(
            {
                "RegisteredVoters":
                    "{:,.0f}",
                "ValidVotes":
                    "{:,.0f}",
                "SpoiltVotes":
                    "{:,.0f}",
                "ParticipationProxy":
                    "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    chart = turnout_df[
        [
            "ElectionYear",
            "ParticipationProxy"
        ]
    ].set_index(
        "ElectionYear"
    )

    st.line_chart(
        chart
    )

    st.divider()

    st.subheader(
        "2026 Model Estimate"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Projected Participation",
        f"{projected_turnout:.2f}%"
    )

    col2.metric(
        "Estimated Registered Voters",
        f"{projected_registered_voters:,.0f}"
    )

    col3.metric(
        "Estimated Participating Voters",
        f"{estimated_2026_voters:,.0f}"
    )

    st.info(
        """
        The 2026 voter estimate is an extrapolation from historical
        registration and participation patterns. It is therefore subject
        to substantial uncertainty, particularly because the official
        2026 voter registration total and final election turnout are not
        available before Election Day.
        """
    )


# ============================================================
# SOURCES & LIMITATIONS
# ============================================================

elif page == "Sources & Limitations":

    st.header(
        "Sources, Methodology & Limitations"
    )

    st.subheader(
        "Official Data Sources"
    )

    st.markdown(
        """
        **Electoral Commission of South Africa (IEC)**

        https://results.elections.org.za/

        **IEC Election Reports**

        https://www.elections.org.za/content/pages/reports/npe/selection.aspx

        **Municipal Demarcation Board (MDB)**

        https://www.demarcation.org.za/

        **Statistics South Africa**

        https://www.statssa.gov.za/

        **eThekwini Municipality**

        https://www.durban.gov.za/
        """
    )

    st.subheader(
        "Historical Datasets"
    )

    st.markdown(
        """
        The dashboard uses the following historical eThekwini datasets:

        - 2011 election results
        - 2016 election results
        - 2021 election results

        The files contain ward, voting district, ballot type, party and
        valid-vote information.
        """
    )

    st.subheader(
        "Methodology"
    )

    st.markdown(
        """
        ### Metro projection

        PR-ballot results are aggregated at municipality level and converted
        into party vote shares.

        Linear regression is used to extrapolate historical vote-share
        trends toward 2026.

        ### Ward projection

        Ward-ballot results are used to identify historical ward winners.

        Three wards with the strongest historical voting-district continuity
        were selected.

        A Random Forest classifier is then used to estimate the likely
        leading party in the selected wards.

        ### Validation

        Temporal validation is used instead of random train/test splitting.
        This prevents information from later elections being used to predict
        earlier elections.
        """
    )

    st.subheader(
        "Important Limitations"
    )

    st.markdown(
        """
        1. Only 2011, 2016 and 2021 historical elections are available.
        2. Three elections provide a very small sample for forecasting.
        3. Political alliances, campaigns and voter behaviour may change.
        4. Historical ward boundaries may not perfectly match 2026 wards.
        5. Model estimates contain uncertainty and should not be treated as
           official election results.
        6. Vote share does not directly determine which political party will
           govern the municipality.
        7. Turnout estimates are based on historical patterns and available
           data and should not be interpreted as official IEC forecasts.
        """
    )

    st.subheader(
        "Responsible Use"
    )

    st.markdown(
        """
        This dashboard is intended for academic and analytical purposes.
        The projections should not be interpreted as guarantees of electoral
        outcomes.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "eThekwini 2026 Election Analytics | Academic Project | "
    "2026 South African Local Government Elections"
)
