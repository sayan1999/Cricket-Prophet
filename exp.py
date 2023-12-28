import os, joblib
import pysplitter as pysp


def load_model(format):
    format = format.lower()
    print(os.listdir("model"))
    # print(os.listdir("model/splits"))
    pysp.unsplit(
        f"model/{format}*.split",
        f"model/{format}.feather.joblib",
    )
    return joblib.load(f"model/{format}.feather.joblib")


load_model("T20")
