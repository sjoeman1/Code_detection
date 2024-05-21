# get all the datasets in results and caclulate the average length of the code for gold_completion and parsed_codes
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get files in results
data = os.listdir('results')
# remove files that do not contain 207
data = [file for file in data if '207' in file]

df = pd.DataFrame(columns=['dataset', 'gold_completion', 'parsed_codes'])

# get the average length of the code for gold_completion and parsed_codes
for file in data:
    with open('results/'+ file, 'r') as f:
        dataset = [json.loads(line) for line in f.readlines()]
        # create dataframe from dataset
        dataset_df = pd.DataFrame(dataset)
        dataset_df['dataset'] = file

        #join dataset_df with df
        df = pd.concat([df, dataset_df])

# remove rows with # CANNOT PARSE
df = df[~df['parsed_codes'].str.startswith("# CANNOT")]

# print the dataset
print(df)

# calculate len of gold_completion and parsed_codes
df['gold_completion_len'] = df['gold_completion'].apply(lambda x: len(x))
df['parsed_codes_len'] = df['parsed_codes'].apply(lambda x: len(x))

# group by dataset calculate meand and std and plot errorbar
grouped = df.groupby('dataset').agg({'gold_completion_len': ['mean', 'std'], 'parsed_codes_len': ['mean', 'std']})

grouped.apply(print)

# print average len of gold_completion and parsed_codes for each dataset
for dataset in grouped.index:
    print(f"Average length of the code for gold_completion in {dataset}: {grouped.loc[dataset]['gold_completion_len']['mean']}")
    print(f"Average length of the code for parsed_codes in {dataset}: {grouped.loc[dataset]['parsed_codes_len']['mean']}")
    print()
# plot the average length of the code for gold_completion and parsed_codes
grouped.plot(kind='bar', y=['gold_completion_len', 'parsed_codes_len'], yerr=['gold_completion_len', 'parsed_codes_len'], title='Average length of the code for gold_completion and parsed_codes')

