import json, os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

root = "cricsheet/all_json"

# print([json.load(open(os.path.join(root, f)))['meta']['data_version'] for f in os.listdir(root) if json.load(open(os.path.join(root, f)))['meta']['data_version']=='1.1.0'])
# print(set([json.load(open(os.path.join(root, f)))['info']['match_type'] for f in os.listdir(root) if f.endswith('.json') and json.load(open(os.path.join(root, f)))['meta']['data_version']=='1.1.0']))

# formats: 'ODI', 'MDM', 'IT20', 'ODM', 'Test', 'T20'


class Inning(object):
    def __init__(self, df, inning, format):
        self.df = df
        self.inning = inning
        self.final_score = df["run"].sum()
        self.format = format

    def settarget(self, target):
        if self.inning == 1:
            print("first innning: don't set target")
        self.target = target


def process_inning(ballbyball):
    score = []
    for over in ballbyball["overs"]:
        overall = []
        for ballcount, dlv in enumerate(over["deliveries"]):
            run = dlv["runs"]["total"]
            wicket = len(dlv.get("wickets", []))
            if ballcount < 6:
                overall.append((run, wicket))
            else:
                lastrun, lastwkt = overall.pop()
                overall.append((run + lastrun, wicket + lastwkt))
        score.extend(overall)
    df = pd.DataFrame(score, columns=["run", "wicket"], index=range(1, len(score) + 1))
    df.index.name = "balls"
    return df


def process_matches(matches, format):
    print("processing jsons...")
    ID = 0
    for match in tqdm(matches):
        if len(match) == 2:
            inning1, inning2 = [
                Inning(process_inning(inning), i + 1, format)
                for i, inning in enumerate(match)
            ]
            inning2.settarget(inning1.final_score)
            inning1.battingteam, inning2.battingteam = (
                match[0]["team"],
                match[1]["team"],
            )
            inning1.bowlingteam, inning2.bowlingteam = (
                match[1]["team"],
                match[0]["team"],
            )
            ID += 1
            inning1.matchid = inning2.matchid = ID
            yield inning1
            yield inning2


def get_all_matches(
    format,
    since=1990,
):
    matches = []
    print("Loading jsons...")
    for f in tqdm(os.listdir(root)[:]):
        if f.endswith(".json"):
            obj = json.load(open(os.path.join(root, f)))
            if (
                format in obj["info"]["match_type"]
                and int(datetime.strptime(obj["info"]["dates"][0], "%Y-%m-%d").year)
                >= since
            ):
                matches.append(obj["innings"])
    return list(process_matches(matches, format))


# get_all_T20s()
