import subprocess, sys
from multiprocessing import Pool
import pandas as pd, json, os, math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from cricksheet import get_all_matches

import ydata_profiling


## Reading IPL dataset
total_wickets = 10
n_pools = 100


## Feature selection/creation and ngram creation

features_for_profiling = features = [
    # "batting_team",
    # "bowling_team",
    # "balls",
    # "runs",
    # "wickets",
    "wkt_last_5_overs",
    # "runrate_last_5_overs",
    "current_RR",
    # "average",
    "balls_left",
    "wkts_left",
    # "required_RR",
    # "projected_score_more",
    # "min_score_more",
    # "max_score_more",
    # "projected_avg_score_more",
    "runrate_last_5_overs-current_RR",
    "deviation_from_projected_rate",
    "deviation_from_projected",
]

features = [
    "matchid",
    "format",
    "inning",
    "batting_team",
    "bowling_team",
    "balls",
    "runs",
    "wickets",
    "wkt_last_5_overs",
    "runrate_last_5_overs",
    "runrate_last_5_overs-current_RR",
    "current_RR",
    # "average",
    "balls_left",
    "wkts_left",
    # "required_RR",
    # "projected_score_more",
    # "min_score_more",
    # "max_score_more",
    # "projected_avg_score_more",
    "final_score",
    "final_score_more",
    "deviation_from_projected",
    "deviation_from_projected_rate",
]

getformat = {"ODI": 1, "T20": 2}


def extract_features(inning):
    data = []
    # total_balls = (
    #     120 if inning.format == "T20" else 300 if inning.format == "ODI" else None
    # )
    total_balls = len(inning.df)
    df = inning.df
    # matchid = inning.matchid
    # batting_team = inning.battingteam

    for i in range(1, len(df)):
        min_RR = 0.5
        max_RR = 2.5
        runs = df.iloc[:i]["run"].sum()
        run_last_5_overs = df["run"].iloc[-30:].sum()
        runrate_last_5_overs = run_last_5_overs / 6

        wickets = df.iloc[:i]["wicket"].sum()
        wkt_last_5_overs = df.iloc[:i]["wicket"].iloc[-30:].sum()

        balls = len(df.iloc[:i])

        current_RR = (runs * 6) / balls
        rr_diff = runrate_last_5_overs - current_RR
        average = runs / (wickets + 1)

        balls_left = total_balls - balls
        wk_left = total_wickets - wickets

        required_RR = (
            ((inning.target - runs) * 6) / balls if inning.inning == 2 else -9999
        )

        projected_score_more = current_RR * balls_left / 6
        min_score_more = min_RR * balls_left / 6
        max_score_more = max_RR * balls_left / 6
        projected_avg_score_more = average * wk_left / 6

        final_score_more = inning.final_score - runs
        format = getformat[inning.format]

        deviation_from_projected = final_score_more - projected_score_more
        data.append(
            (
                inning.matchid,
                format,
                inning.inning,
                inning.battingteam,
                inning.bowlingteam,
                balls,
                runs,
                wickets,
                wkt_last_5_overs,
                round(runrate_last_5_overs, 2),
                round(rr_diff, 2),
                round(current_RR, 2),
                # average,
                balls_left,
                wk_left,
                # required_RR,
                # projected_score_more,
                # min_score_more,
                # max_score_more,
                # projected_avg_score_more,
                inning.final_score,
                final_score_more,
                round(deviation_from_projected),
                (deviation_from_projected * 6) / balls_left,
            )
        )
    return data


def save_features(innings, fname):
    print("Feature enggineering and ngram creation...")

    n_innings = len(innings)
    print(f"{n_innings=}")
    pool = Pool(processes=n_pools)
    Xy = pool.map(extract_features, innings)

    Xy = [xi for Xi in Xy for xi in Xi]
    print(f"{len(Xy)=}")
    featuresdf = pd.DataFrame(Xy, columns=features)
    ydata_profiling.ProfileReport(
        featuresdf[features_for_profiling], title=fname
    ).to_file(fname + ".html")
    featuresdf.to_feather(fname)
    featuresdf.to_csv(fname + ".csv")


if __name__ == "__main__":
    print("Loading t20 data...")
    innings = get_all_matches(format="T20", since=2021)
    print("Saving t20 data")
    save_features(innings, "data/t20features.feather")

    print("Loading odi data...")
    innings = get_all_matches(format="ODI", since=2021)
    print("Saving odi data")
    save_features(innings, "data/odifeatures.feather")
