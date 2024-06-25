import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import os
import json
import matplotlib.pyplot as plt

# make df of al files in code_bert folder
def make_df():
    files = os.listdir("code_bert")
    df = pd.DataFrame()
    for file in files:
        with open(f"code_bert/{file}", 'r') as f:
            dataset = [json.loads(line) for line in f.readlines()]
        df = df._append(dataset)
    return df

df = make_df()
df['code_length'] = df['code'].apply(lambda x: len(x))
print(df.columns)

# group by label
label_group = df.groupby('label_name')


def save_group(grouped):
    # profile groups
    for name, group in grouped:
        profile = ProfileReport(group, title=f"Code Detection Profiling Report - {name}")
        profile.to_file(f"code_detection_profiling_report_{name}.html")
        print(profile)
#
# save_group(label_group)

difficulty_label_group = df.groupby(['label_name', 'difficulty'])
#
# save_group(difficulty_label_group)

def plot_group():
    # plot boxplot of code length each group in 1 plot
    df['label_name'] = df['label_name'].apply(lambda x: x.replace('_', ' ').capitalize())
    fig, ax = plt.subplots(figsize=(10, 8))
    # add overall boxplot as well as grouped boxplot
    df.boxplot(column=['code_length'], by=['difficulty', 'label_name'], ax=ax)

    plt.title('Lenght of generated code samples difficulty')
    # x titel
    plt.xlabel('Origin and difficulty')
    plt.suptitle('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


plot_group()

