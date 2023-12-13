#!/bin/zsh
mkdir -p data history model result temp
source env/bin/activate && pip install -r requirements.txt &&  python features.py && python model.py &&  streamlit run serve.py
