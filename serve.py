from scrape import scrape, get_live_matches
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts
import joblib
import numpy as np
import math, os
import datetime, time
import matplotlib.pyplot as plt

import pathlib

for folder in ["data", "model", "history", "result", "temp"]:
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

# ## Test on realdata

# In[16]:

from model import features
import streamlit_analytics

# features = [
#     "batting_team",
#     "balls",
#     "runs",
#     "wickets",
#     "wkt_last_5_overs",
#     "runrate_last_5_overs",
#     "current_RR",
#     "average",
#     "balls_left",
#     "wkts_left",
#     "required_RR",
#     "projected_score_more",
#     "min_score_more",
#     "max_score_more",
#     "projected_avg_score_more",
# ]

all_teams_enc = list(range(len(np.load("model/team.npy", allow_pickle=True))))


def overtoball(over):
    over = str(over)
    full = int(over.split(".")[0]) * 6
    part = min(int(over.split(".")[-1]), 6)
    print(f"{over=}", "balls=", full + part)
    return full + part


def save_history(fname, row, total_balls):
    row.to_csv(
        os.path.join("history", fname),
        mode="a" if os.path.isfile(os.path.join("history", fname)) else "w+",
        header=not os.path.isfile(os.path.join("history", fname)),
    )
    fig, ax = plt.subplots()
    historydf = pd.read_csv(os.path.join("history", fname))
    balls = (total_balls - historydf["balls_left"]).to_list()
    runs = historydf["runs"].astype(int).to_list()
    ax.plot(balls, runs, label="So Far")
    balls.append(total_balls)
    pred_runs = runs + [historydf["predicted"].astype(int).iloc[-1]]
    ax.plot(balls[-2:], pred_runs[-2:], label="Predicted")
    proj_runs = runs + [historydf["projected"].astype(int).iloc[-1]]
    ax.plot(balls[-2:], proj_runs[-2:], label="Projected")
    ax.annotate(str(runs[-1]), xy=(balls[-2], runs[-1]))
    ax.annotate(str(pred_runs[-1]), xy=(balls[-1], pred_runs[-1]))
    ax.annotate(str(proj_runs[-1]), xy=(balls[-1], proj_runs[-1]))
    plt.xlim([0, total_balls])
    plt.ylim([0, max(pred_runs[-1], proj_runs[-1]) + 100])
    ax.set_xlabel("Balls")
    ax.set_ylabel("Runs")
    ax.legend()
    return fig


def load_model(format):
    return joblib.load(
        "model/"
        + (
            "t20features.feather.joblib"
            if format == "T20"
            else "odifeatures.feather.joblib"
            if format == "ODI"
            else None
        )
    )


def simulator(args, format):
    inputdf = pd.DataFrame([args.values()], columns=args.keys())
    model = load_model(format)
    h = model.predict(inputdf)
    return h


def predict(url):
    fname = "".join(list(filter(str.isalnum, url))) + ".csv"
    ret = scrape(url)
    print(ret)
    if len(ret) == 1:
        err = ret[0]
        return [err]
    else:
        (
            matchState,
            score,
            run_last_5_overs,
            wkt_last_5_overs,
            runs,
            wkts,
            overs,
            req_rr,
            req,
            current_rr,
            format,
            title,
            status,
            batting_team,
            bowling_team,
            batting_team_enc,
            bowling_team_enc,
            inning,
        ) = ret
    if matchState != "inprogress":
        return matchState, None, score, format, title, status, None, None, None, None

    total_balls = 120 if format == "T20" else 300 if format == "ODI" else None
    balls = overtoball(overs)
    rr_last_5_overs = (int(run_last_5_overs) * 6) / min(30, balls)
    # current_rr = (runs * 6) / balls
    avg = runs / (wkts + 1)
    req_rr = req_rr
    wkts_left = 10 - wkts
    balls_left = (total_balls - balls) if inning == 1 else math.ceil(req * 6 / req_rr)
    min_score_avg, max_score_avg = (
        math.ceil(balls_left * 0.5),
        math.ceil(balls_left * 3),
    )
    rr_diff = rr_last_5_overs - current_rr
    inputs = {
        "batting_team": batting_team_enc,
        "balls": balls,
        "runs": runs,
        "wickets": wkts,
        "wkt_last_5_overs": wkt_last_5_overs,
        "runrate_last_5_overs": rr_last_5_overs,
        "current_RR": current_rr,
        "runrate_last_5_overs-current_RR": rr_diff,
        "average": avg,
        "balls_left": int(balls_left),
        "wkts_left": int(wkts_left),
        "required_RR": -9999,
        "projected_score_more": math.ceil(balls_left * ((runs) / (balls))),
        "min_score_more": math.ceil(balls_left * 0.5),
        "max_score_more": math.ceil(balls_left * 3),
        "projected_avg_score_more": math.ceil((10 - wkts) * runs / (1 + wkts)),
    }
    inputdf = pd.DataFrame(inputs, index=[0])
    if batting_team_enc is None:
        inputdf = inputdf.drop(columns=["batting_team"])
        inputdf = pd.concat([inputdf] * len(all_teams_enc))
        inputdf["batting_team"] = all_teams_enc
    inputdf = inputdf[features]
    model = load_model(format)
    h_rate = model.predict(inputdf)
    print(f"{h_rate=}")
    h = h_rate * balls_left / 6
    projected_score_more = balls_left * current_rr / 6
    projected = math.ceil(projected_score_more + runs)
    predicted_score_more = math.ceil(h.mean() + projected_score_more)
    # predicted_score_more = min(max(min_score_avg, predicted_score_more), max_score_avg)
    predicted = runs + predicted_score_more

    print(f"{runs=}, {projected=}, {predicted=}")
    inputdf["timestamp"] = datetime.datetime.now()
    inputdf["runs"] = runs
    if inning == 2:
        target = req + runs
        print(f"{target=}")
        inputdf["target"] = target
        batting_team_win = int(predicted - target)
    else:
        batting_team_win = None
        inputdf["target"] = -9999
    inputdf["predicted"] = int(predicted)
    inputdf["projected"] = int(projected)
    print(inputdf.to_string())
    fig = save_history(fname, inputdf, total_balls)

    return (
        matchState,
        predicted,
        score,
        format,
        title,
        status,
        inning,
        batting_team,
        batting_team_win,
        fig,
    )


def getoption(predicted, maxscore):
    return {
        "series": [
            {
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "min": 0,
                "max": maxscore,
                "center": ["50%", "50%"],
                "splitNumber": 4,
                "axisLine": {
                    "lineStyle": {
                        "width": 6,
                        "color": [
                            [0.25, "#FF403F"],
                            [0.5, "#FDDD60"],
                            [0.75, "#00FF00"],
                            [1, "#0000FF"],
                        ],
                    }
                },
                "pointer": {
                    "icon": "path://M12.8,0.7l12,40.1H0.7L12.8,0.7z",
                    "length": "12%",
                    "width": 30,
                    "offsetCenter": [0, "-60%"],
                    "itemStyle": {"color": "auto"},
                },
                "axisTick": {
                    "length": 10,
                    "lineStyle": {"color": "auto", "width": 2},
                },
                "splitLine": {
                    "length": 15,
                    "lineStyle": {"color": "auto", "width": 5},
                },
                "axisLabel": {
                    "fontSize": 12,
                    "distance": -60,
                },
                "title": {
                    "offsetCenter": [0, "-20%"],
                    "fontSize": 20,
                    "color": "#0000FF"
                    if predicted > maxscore * 0.75
                    else "#00FF00"
                    if predicted > maxscore * 0.5
                    else "#FDDD60"
                    if predicted > maxscore * 0.25
                    else "#FF403F",
                },
                "detail": {
                    "fontSize": 15,
                    "offsetCenter": [0, "0%"],
                    "valueAnimation": True,
                    "color": "auto",
                    "formatter": "Predicted Score: {value}",
                },
                "data": [
                    {
                        "value": round(predicted),
                    }
                    # {
                    #     "value": round(predicted),
                    #     "name": "Great"
                    #     if predicted > maxscore * 0.75
                    #     else "Decent"
                    #     if predicted > maxscore * 0.5
                    #     else "Average"
                    #     if predicted > maxscore * 0.25
                    #     else "Bad",
                    # }
                ],
            }
        ]
    }


def timestamp(func):
    def caller(*args):
        print(
            "\n---->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Initiated:  ",
            datetime.datetime.now(),
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<----",
        )
        ret = func(*args)
        print(
            "\n---->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Completed:  ",
            datetime.datetime.now(),
            "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<----",
        )
        return ret

    return caller


@timestamp
def render(url):
    markdown = []
    option = None
    print("fetching from", url)
    ret = predict(url.strip())
    if len(ret) == 1:
        err = ret[0]
        markdown.append("Error fetching url...")
        return markdown, None, None
    (
        matchState,
        predicted,
        score,
        format,
        title,
        status,
        inning,
        batting_team,
        batting_team_win,
        fig,
    ) = ret

    if matchState:
        markdown.append("Live score credits: cricbuzz.com")
        if title:
            if "|" in title:
                l1 = (
                    title.split("|")[1]
                    .replace("Cricbuzz.com", "")
                    .replace("Cricbuzz", "")
                )
                if l1.strip():
                    markdown.append(l1.strip())
                l2 = (
                    title.split("|")[0]
                    .replace("Cricbuzz.com", "")
                    .replace("Cricbuzz", "")
                )
                if l2.strip():
                    markdown.append(l2.strip())
            else:
                markdown.append(
                    title.replace("Cricbuzz.com", "").replace("Cricbuzz", "")
                )
        nutshell = ""
        if status:
            nutshell += status + "; "
        if score:
            nutshell += score + "; "
        if matchState:
            nutshell += matchState + "; "
        if nutshell:
            markdown.append(nutshell)
    # if matchState and matchState != "inprogress":
    #     markdown.append(matchState)
    if predicted:
        if inning == 2:
            if batting_team_win >= 0:
                markdown.append(f"{batting_team} may win")
            else:
                markdown.append(
                    f"{batting_team} may lose by {-int(batting_team_win)} runs"
                )
        maxscore = 300 if format == "T20" else 500 if format == "ODI" else None
        option = getoption(predicted, maxscore)
    if matchState is None:
        markdown.append("Error fetching url...")

    return "\n".join(markdown), option, fig


if __name__ == "__main__":
    with streamlit_analytics.track(unsafe_password="credict123"):
        st.set_page_config(page_title="Cricket Prophet")
        st.title("Cricket Prophet")
        st.write("**An ML-driven Cricket Score Predictor**")

        live_matches = get_live_matches("https://cricbuzz.com")
        if live_matches:
            option = st.sidebar.selectbox(
                "Choose a live match here",
                list(live_matches.keys()) + ["Custom URL", "Simulator"],
            )
            if option == "Simulator":
                format = st.selectbox("Format", ["T20", "ODI"])
                args = {}
                args["batting_team"] = 1
                args["wkt_last_5_overs"] = st.number_input(
                    "wkt_last_5_overs", value=0.0, step=0.01, format="%f"
                )
                args["current_RR"] = st.number_input(
                    "current_RR", value=0.0, step=0.01, format="%f"
                )
                args["balls_left"] = st.number_input(
                    "balls_left", value=0.0, step=0.01, format="%f"
                )
                args["wkts_left"] = st.number_input(
                    "wkts_left", value=0.0, step=0.01, format="%f"
                )
                args["runrate_last_5_overs-current_RR"] = (
                    st.number_input(
                        "runrate_last_5_overs", value=0.0, step=0.01, format="%f"
                    )
                    - args["current_RR"]
                )
                balls = 300 if format == "ODI" else 120
                if st.button("Predict"):
                    st.text(
                        str(
                            int(
                                (balls * args["current_RR"] / 6)
                                + simulator(args, format) * args["balls_left"] / 6
                            )
                        )
                    )
            else:
                if option == "Custom URL":
                    url = st.text_input("Enter cricbuzz match link")
                else:
                    url = live_matches.get(option)

                col1, col2 = st.columns([3.5, 0.6])

                with col1:
                    live = st.button("Live", help="Livestream")
                with col2:
                    fetch = st.button("Fetch", help="Refresh")

                col3, _ = st.columns([1, 4])
                with col3:
                    interval = st.number_input(
                        label="Sync Interval (Seconds)", step=1, min_value=1, value=100
                    )

                placeholder = st.empty()

                if fetch:
                    if url:
                        markdown, option, fig = render(url)
                        placeholder.empty()
                        with placeholder.container():
                            st.text(markdown)
                            st.text(f"Last updated at {time.strftime('%H:%M %p')}")
                            if option:
                                st_echarts(
                                    option,
                                    width="450px",
                                    height="350px",
                                    key="gauge" + str(datetime.datetime.now()),
                                )
                                if fig:
                                    st.pyplot(fig)

                if live:
                    if url:
                        while True:
                            markdown, option, fig = render(url)
                            placeholder.empty()
                            with placeholder.container():
                                st.text(markdown)
                                st.text(f"Last updated at {time.strftime('%H:%M %p')}")
                                if option:
                                    st_echarts(
                                        option,
                                        width="450px",
                                        height="350px",
                                        key="gauge" + str(datetime.datetime.now()),
                                    )
                                    if fig:
                                        st.pyplot(fig)
                                else:
                                    break
                            time.sleep(interval)
        else:
            st.text("Error fetching matches")
