'''
Possibly the quickest and dirtiest sentiment analysis of all time

To be run on data as of 19 JUN 2022
'''

import json
import os
import pandas as pd

'''
Ingest JSON formatted facebook message files into dataframe
'''

DIR = "messages/inbox/"
TARGET = "Joseph Jeong"

messages = []

folders = os.listdir(DIR)
# folders = os.listdir(DIR)
for folder in folders:
    files = os.listdir(DIR + folder)
    chat_files = list(filter(lambda s: ".json" in s, files))
    for chat_file in chat_files:
        with open(DIR + f"{folder}/{chat_file}", "r") as f:
            chat_dict = json.load(f)

            participants= chat_dict["participants"]
            participants = list(map(lambda o: o["name"], participants))
            try: participants.remove(TARGET)
            except: participants = []
            participants = str(participants)

            msg_objs = chat_dict["messages"]
            msg_objs = sorted(msg_objs, key=lambda o: o["timestamp_ms"])

            first_msg_time = 0
            combined_msg = ""

            for msg_obj in msg_objs:
                # sometimes messages are different formats (e.g. sticker)
                try: msg = msg_obj["content"]
                except: continue

                sender = msg_obj["sender_name"]
                unix = msg_obj["timestamp_ms"]

                if not first_msg_time:first_msg_time = unix

                # combines multiple messages into longer message
                if sender == TARGET:
                    combined_msg += " " + msg
                elif combined_msg:
                    messages.append({
                        "participants": participants,
                        "unix": first_msg_time,
                        "message": combined_msg
                    })
                    combined_msg = ""
                    first_msg_time = 0

df = pd.DataFrame(messages)
# remove any insubstantial messages
df = df[df["message"].str.len() > 10]

'''
Moving onto Sentiment Analysis
VADER is used 
Why? Apparently its trained on social media, also its readily available.
'''
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download([
    "vader_lexicon"
])

SID = SentimentIntensityAnalyzer()

def classify(row) -> float:
    message = row["message"]
    ss = SID.polarity_scores(message)
    score = ss["compound"]
    neg = 1 if score < 0 else 0
    pos = 1 if score > 0 else 0
    return neg, pos

df[["neg", "pos"]] = df.apply(classify, axis=1, result_type="expand")
df_messages = df.copy()
print(df.shape[0])
print(df.columns)

'''
Start Aggregating by month
'''
df["date"] = pd.to_datetime(df["unix"], unit="ms")
df.drop(columns=["unix", "message", "participants"], inplace=True)
print(df.columns)

df = df.groupby(pd.Grouper(freq="Y", key="date")).agg(
    neg_count=("neg", "count"), 
    neg_sum=("neg", "sum"),
    pos_count=("pos", "count"), 
    pos_sum=("pos", "sum"),
)
df["neg"] = df["neg_sum"] / df["neg_count"]
df["pos"] = df["pos_sum"] / df["pos_count"]
df.drop(columns=["neg_sum", "pos_sum", "neg_count", "pos_count"], inplace=True)
df["date"] = df.index
df["date"] = df["date"].dt.to_period('Y')
# df.set_axis(["date", "percentage"], axis=1, inplace=True)

'''
plot it
'''
import matplotlib.pyplot as plt

print(df.columns)
df["neu"] = 1 - df["pos"] - df["neg"]
colour_dict = {
    "pos": "g",
    "neu": "gray",
    "neg": "r"
}
df.plot(kind="bar", color=colour_dict, stacked=True, x="date", y=["pos", "neu", "neg"]).get_figure().savefig('output.png')

'''
participants ranked
'''

df = df_messages.groupby(by="participants").agg(
    count=("pos", "count"),
    pos=("pos", "mean"),
    neg=("neg", "mean")
)
# df["participants"] = df.index
df = df[df["count"] > 100]
df = df[~df.index.str.contains(",")]
df = df.sort_values(by="neg", ascending=False)
print(df)
df = df.sort_values(by="pos", ascending=False)
print(df)
df = df.sort_values(by="count", ascending=False)
print(df)