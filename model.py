import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import math
import matplotlib.pyplot as plt, joblib


# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# from sklearn.tree import DecisionTreeRegressor

# from catboost import CatBoostRegressor
import warnings, random
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import statistics

# from sklearn import tree
# from sklearn.svm import SVR
# from sklearn.ensemble import VotingRegressor
import os

warnings.filterwarnings("ignore")
features = [
    "batting_team",
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
]
target = "deviation_from_projected_rate"


# evaluate
def evaluate(model, featuresdf, x_test, fname):
    predictdf = featuresdf.loc[x_test.index].copy()
    # print(predictdf.columns)
    predictdf["h_deviation_from_projected_rate"] = model.predict(
        featuresdf.loc[x_test.index][features]
    )
    predictdf["h_deviation_from_projected"] = (
        predictdf["h_deviation_from_projected_rate"] * predictdf["balls_left"] / 6
    )
    predictdf["error"] = (
        predictdf["h_deviation_from_projected"] - predictdf["deviation_from_projected"]
    )
    predictdf["abs_error"] = predictdf["error"].abs()
    ax = plt.gca()
    plt.plot(predictdf.groupby("balls").aggregate({"abs_error": "mean"}))
    plt.legend("Abs deviation")

    # ax.set_ylim([-50, 50])
    plt.title(type(model).__name__)
    plt.xlabel("Balls on which prediction was made")
    plt.ylabel("Mean Abs Prediction error")
    plt.savefig("result/" + fname + ".png")
    plt.clf()
    predictdf.sample(frac=0.0001).to_csv("result/" + fname + "_sample.csv")
    # fig = plt.figure(figsize=(25, 20))
    # tree.plot_tree(model)
    # fig.savefig(fname + ".png")
    # plt.clf()batting_teamsort_values("overs", ascending=False).to_string(index=False))


def train_test_split_matchid(df, matchids, split=0.2):
    unique_match_ids = set(matchids)
    print(f"{len(unique_match_ids)=}")
    testids = random.sample(unique_match_ids, int(len(unique_match_ids) * split))
    trainids = list(unique_match_ids.difference(testids))
    return (
        df[features][df.matchid.isin(trainids)],
        df[features][df.matchid.isin(testids)],
        df[target][df.matchid.isin(trainids)],
        df[target][df.matchid.isin(testids)],
    )


def encode_teams(series):
    encoder = LabelEncoder()
    encoder.fit(series)
    np.save("model/team.npy", encoder.classes_)


def transform_teams(series):
    encoder = LabelEncoder()
    encoder.classes_ = np.load("model/team.npy", allow_pickle=True)
    return encoder.transform(np.array(series).reshape(-1, 1)).reshape(-1)


def plot_feature_importance(f, imp, fname):
    importance = (
        pd.DataFrame(
            zip(*[f, imp]),
            columns=["feature", "importance"],
        )
        .sort_values("importance", ascending=False)
        .set_index("feature")
    )
    importance["importance"] = importance["importance"] / importance["importance"].sum()
    fig, ax = plt.subplots()
    importance.plot.bar(ax=ax)
    ax.bar_label(ax.containers[0], labels=f, rotation=90, label_type="center")
    ax.set_xticks([])
    ax.set_title(
        "Feature importances for 'Predicted deviation of final score from projected score' "
        + fname.split(".")[0]
    )
    ax.set_ylabel("Feature Importance")
    ax.set_xlabel("Features")
    plt.savefig("result/" + fname + "featureimp.png")
    plt.clf()


def compare_util(model, fname, x_train, x_test, y_train, y_test, balls_left):
    print(f'After {(300 if "odi" in fname.lower() else 120) - balls_left} balls')

    train_index = x_train["balls_left"] <= balls_left
    test_index = x_test["balls_left"] <= balls_left
    y_train_data = y_train[train_index] * x_train["balls_left"][train_index] / 6
    y_test_data = y_test[test_index] * x_test["balls_left"][test_index] / 6
    h_train_data = (
        model.predict(x_train[train_index]) * x_train["balls_left"][train_index] / 6
    )
    h_test_data = (
        model.predict(x_test[test_index]) * x_test["balls_left"][test_index] / 6
    )

    print(
        f"Data: Train Variance: {statistics.variance(y_train_data)}, Test Variance: {statistics.variance(y_test_data)}"
    )
    print(
        f"Model: Train MSE: {mse(h_train_data, y_train_data, squared=False)}, Test MSE: {mse(h_test_data, y_test_data, squared=False)}"
    )
    print(
        f"Model: Train R2: {r2_score(y_train_data, h_train_data)}, Test R2: {r2_score(y_test_data, h_test_data)}"
    )


def compare(model, fname, x_train, x_test, y_train, y_test):
    print("Let's see variation of of deviation_from_projected")
    if "odi" in fname.lower():
        compare_util(
            model, fname, x_train, x_test, y_train, y_test, balls_left=300 - 120
        )
        compare_util(
            model, fname, x_train, x_test, y_train, y_test, balls_left=300 - 180
        )
        compare_util(
            model, fname, x_train, x_test, y_train, y_test, balls_left=300 - 240
        )

    if "t20" in fname.lower():
        compare_util(
            model, fname, x_train, x_test, y_train, y_test, balls_left=120 - 30
        )
        compare_util(
            model, fname, x_train, x_test, y_train, y_test, balls_left=120 - 60
        )
        compare_util(
            model, fname, x_train, x_test, y_train, y_test, balls_left=120 - 90
        )


def train(fname, max_depth=-1):
    print("training on", fname, "...")
    featuresdf = pd.read_feather(fname)
    featuresdf = featuresdf[featuresdf["inning"] == 2]
    encode_teams(
        featuresdf["batting_team"].to_list() + featuresdf["bowling_team"].to_list()
    )
    featuresdf["batting_team"] = transform_teams(featuresdf["batting_team"])
    featuresdf["bowling_team"] = transform_teams(featuresdf["bowling_team"])
    x_train, x_test, y_train, y_test = train_test_split_matchid(
        featuresdf, featuresdf["matchid"], 0.2
    )
    print(f"{len(x_train)=} {len(x_test)=}")

    model = RandomForestRegressor(max_depth=8)
    model.fit(x_train, y_train)

    # for xgb
    # plot_feature_importance(
    #     model.get_booster().get_score(importance_type="gain").keys(),
    #     model.get_booster().get_score(importance_type="gain").values(),
    #     fname,
    # )

    # for rf
    plot_feature_importance(
        features,
        np.std([tree.feature_importances_ for tree in model.estimators_], axis=0),
        os.path.basename(fname),
    )
    print("Depth:", [e.tree_.max_depth for e in model.estimators_])

    # for dt
    # plot_feature_importance(
    #     features,
    #     model.feature_importances_,
    #     fname,
    # )
    # print(model.tree_.max_depth)

    # print(f"{model.score(x_train, y_train)=}, {model.score(x_test, y_test)=}")
    print(
        f"Train MSE: {mse(model.predict(x_train)*x_train['balls_left']/6, y_train*x_train['balls_left']/6, squared=False)}, Test MSE: {mse(model.predict(x_test)*x_test['balls_left']/6, y_test*x_test['balls_left']/6, squared=False)}"
    )

    print(
        f"Train R2: {r2_score(y_train*x_train['balls_left']/6, model.predict(x_train)*x_train['balls_left']/6)}, Test R2: {r2_score(y_test*x_test['balls_left']/6, model.predict(x_test)*x_test['balls_left']/6)}"
    )

    compare(model, fname, x_train, x_test, y_train, y_test)

    evaluate(model, featuresdf, x_test, os.path.basename(fname))
    model.fit(featuresdf[features], featuresdf[target])

    joblib.dump(model, f"model/{os.path.basename(fname)}.joblib")

    return model


if __name__ == "__main__":
    train("data/t20features.feather")
    train("data/odifeatures.feather")
