import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import numpy as np
from sklearn.preprocessing import LabelEncoder
import traceback
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

import chromedriver_autoinstaller
from selenium.common import exceptions


chromedriver_autoinstaller.install()


options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")


def selnium(url):
    try:
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        with open("temp/temp.html", "w+") as f:
            f.write(driver.page_source)
        driver.quit()
        return True
    except exceptions.InvalidSessionIdException as e:
        print(traceback.format_exc())
        print(e.message)
        return False
    except BaseException as e:
        print(traceback.format_exc())
        print(e.message)
        return False


def get_batting_team(soup, status, inning, teams_this_match):
    # teams_this_match = sorted(
    #     np.load("team.npy", allow_pickle=True),
    #     key=lambda x: soup.text.lower().count(x.lower()),
    # )[-2:]
    # print(f"{teams_this_match=}")
    batting_team = ""
    if inning == 2:
        batting_team = status.split("need")[0].strip()
        for idx, team in enumerate(teams_this_match):
            if team.lower() in batting_team.lower():
                batting_team = team
    else:
        for idx, team in enumerate(teams_this_match):
            if team.lower() in status.lower():
                if "opt to bowl" in status.lower():
                    batting_team = teams_this_match[int(~idx)]
                elif "opt to bat" in status.lower():
                    batting_team = team
                else:
                    print("Could not get batting team)")
    bowling_team = list(set(teams_this_match).difference([batting_team]))[0]
    print(f"{batting_team=}, {bowling_team=}")
    batting_team_enc, bowling_team_enc = None, None
    le = LabelEncoder()
    le.classes_ = np.load("model/team.npy", allow_pickle=True)
    if batting_team in le.classes_:
        batting_team_enc = le.transform([batting_team])[0]
    if bowling_team in le.classes_:
        bowling_team_enc = le.transform([bowling_team])[0]
    return batting_team, bowling_team, batting_team_enc, bowling_team_enc


def scrape(url):
    try:
        if selnium(url) is False:
            return ("Selenium scrape error",)
        soup = BeautifulSoup(open("temp/temp.html", "r").read(), "html.parser")
        # print("Debug>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.", soup.text)
        matchState = re.findall(
            'var matchState ="([\da-zA-Z]*)"',
            "\n".join(map(lambda x: x.text, soup.find_all("script"))),
        )[0].lower()
        print(f"{matchState=}")
        title = soup.find_all("title")[0].text
        format = re.findall(
            'var matchFormat = "([\da-zA-Z]*)"',
            "\n".join(map(lambda x: x.text, soup.find_all("script"))),
        )[0]
        print(f"{format=}")
        if format not in {"ODI", "T20"}:
            raise BaseException("Not ODI or T20")
        status = (
            soup.find_all("div", {"class": "cb-text-inprogress"})[0].text
            if matchState == "inprogress"
            else soup.find_all("div", {"class": "cb-text-complete"})[0].text
            if matchState == "complete"
            else soup.find_all("div", {"class": "cb-text-inningsbreak"})[0].text
            if matchState == "inningsbreak"
            else ""
        )
        score = (
            soup.find_all("div", {"class": "cb-min-bat-rw"})[0].text
            if matchState in ["complete", "inprogress", "inningbreak"]
            else ""
        )
        if matchState != "inprogress":
            return (
                matchState,
                score,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                format,
                title,
                status,
                None,
                None,
                None,
                None,
                None,
            )
        teams_this_match = re.match(
            r"(.*) vs (.*)",
            soup.find_all("a", {"class": "cb-nav-tab"})[0]["title"].split(",")[0],
        ).groups()
        print(f"{teams_this_match=}")

        data = re.findall("(\d+)/(\d+) \(([\.\d]+)\)", soup.text)
        runs, wkts, overs = map(float, data[-1])
        print(f"{runs=}, {wkts=}, {overs=}")

        if overs >= 5:
            last_5_ovs = (
                soup.find_all("span", string="Last 5 overs")[0].findNext("span").text
            )
            run_last_5_overs, wkt_last_5_overs = map(
                float, re.match("(\d+) runs, (\d+) wkts", last_5_ovs).groups()
            )
        else:
            run_last_5_overs, wkt_last_5_overs = runs, wkts
        print(f"{run_last_5_overs=}, {wkt_last_5_overs=}")

        req_rr = -9999
        if soup.find_all("span", string="\xa0\xa0REQ:\xa0"):
            reqdata = (
                soup.find_all("span", string="\xa0\xa0REQ:\xa0")[0]
                .findNext("span")
                .text
            )
            if reqdata.strip() != "":
                req_rr = list(map(float, re.match("([\d\.]+)", reqdata).groups()))[0]
        else:
            print("REQ_RR not parsed")

        crr = -9999
        if soup.find_all("span", string="\xa0\xa0CRR:\xa0"):
            crrdata = (
                soup.find_all("span", string="\xa0\xa0CRR:\xa0")[0]
                .findNext("span")
                .text
            )
            if crrdata.strip() != "":
                crr = list(map(float, re.match("([\d\.]+)", crrdata).groups()))[0]
        else:
            print("CRR not parsed")

        print(f"{crr=}, {req_rr=}")

        inning = 2 if req_rr > 0 else 1
        (
            batting_team,
            bowling_team,
            batting_team_enc,
            bowling_team_enc,
        ) = get_batting_team(soup, status, inning, teams_this_match)

        req = -9999
        if inning == 2:
            req = int(re.match(r".*need (\d+) runs", status).groups()[0])
            print(f"{req=}")
        else:
            print("Not chasing so target not set")

        return (
            matchState,
            score,
            run_last_5_overs,
            wkt_last_5_overs,
            runs,
            wkts,
            overs,
            req_rr,
            req,
            crr,
            format,
            title,
            status,
            batting_team,
            bowling_team,
            batting_team_enc,
            bowling_team_enc,
            inning,
        )
    except BaseException as e:
        print(traceback.format_exc())
        return (str(e),)


def get_live_matches(url):
    if selnium(url) is False:
        return None
    soup = BeautifulSoup(open("temp/temp.html", "r").read(), "html.parser")
    matches = soup.find_all("a", {"class": "cb-mat-mnu-itm cb-ovr-flo"})
    return {
        m.text: urljoin(url, m.get("href"))
        for m in matches
        if m not in soup.find_all("a", {"id": "live-scores-link"})
    }


if __name__ == "__main__":
    url = "https://cricbuzz.com/live-cricket-scores/79055/wa-vs-saus-3rd-match-australia-domestic-one-day-cup-2023-24"
    print(scrape(url))
    # print(get_live_matches("https://cricbuzz.com"))
