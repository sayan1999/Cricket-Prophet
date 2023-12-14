---
title: Cricket-Prophet
emoji: 💻
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: 1.29.0
app_file: serve.py
pinned: false
---

# Cricket-Prophet is an AI-Ml based cricket score prediction app.

The app is online at https://cricket-prophet.streamlit.app/

It takes account of batting team, current run, fall of wkts and gives a realistic prediction of the final score using a #randomforest. Scores are fetched from #cricbuzz site in realtime. It is a better prediction than the projected score as it doesn't only rely on current run rate, but also balls left, wkts left and batting team.

## ![Cricket-Prophet](static/UI.jpg)

# ODI

![correlation](static/ODI_correlation.png)
![feature importance](static/odifeatures.featherfeatureimp.png)
![evaluation](static/odifeatures.feather.png)

# T20

![correlation](static/T20_correlation.png)
![feature importance](static/t20features.featherfeatureimp.png)
![evaluation](static/t20features.feather.png)
