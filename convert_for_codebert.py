# convert the jsonl files in results to be compatible with codebert
# this is done by extracting the parsed codes and the gold_completions, labeling them and putting them in a jsonl file
# split these in train, validation and test sets
# keeping the same ratio as the original dataset
# and keeping the difficulties in the test set seperate

import json
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

train_file = os.path.join("code_bert","codebert_train.jsonl")
val_file = os.path.join("code_bert", "codebert_val.jsonl")
competition_test_file = os.path.join("code_bert" , "codebert_competition_test.jsonl")
interview_test_file = os.path.join("code_bert", "codebert_interview_test.jsonl")
introductory_test_file = os.path.join("code_bert",  "codebert_introductory_test.jsonl")

human_competition = False
human_interview = False
human_introductory = False

def convert_for_codebert(input_file):
    global human_competition, human_interview, human_introductory
    with open(input_file, "r") as f:
        dataset = [json.loads(line) for line in f.readlines()]

    difficulty = input_file.split("_")[-2]

    df = pd.DataFrame(dataset)
    df = df[~df['parsed_codes'].str.startswith("# CANNOT")]


    machine_generated = df['parsed_codes']
    machine_generated_labels = [1] * len(machine_generated)

    # combine the two lists
    print(len(machine_generated))

    # human_written and machine_generated are lists of strings
    data = machine_generated
    print(data)
    data_labels = machine_generated_labels

    if difficulty == "competition" and not human_competition:
        data, data_labels = get_human_samples(data, data_labels, df)
        human_competition = True
    elif difficulty == "interview" and not human_interview:
        data, data_labels = get_human_samples(data, data_labels, df)
        human_interview = True
    elif difficulty == "introductory" and not human_introductory:
        data, data_labels = get_human_samples(data, data_labels, df)
        human_introductory = True


    labels = ['human_written', 'machine_generated']

    print(len(data), len(data_labels))

    # split data into train, validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, data_labels, shuffle=True, test_size=int(len(data)*0.15), random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, test_size=int(len(data)*0.15), random_state=42)

    # reset indices
    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    write_to_file(X_train, y_train, labels, difficulty, input_file, train_file)
    write_to_file(X_val, y_val, labels, difficulty, input_file, val_file)

    if difficulty == "competition":
        write_to_file(X_test, y_test, labels, difficulty, input_file, competition_test_file)
    elif difficulty == "interview":
        write_to_file(X_test, y_test, labels, difficulty, input_file, interview_test_file)
    elif difficulty == "introductory":
        write_to_file(X_test, y_test, labels, difficulty, input_file, introductory_test_file)


def get_human_samples(data, data_labels, df):
    human_written = df['gold_completion']
    human_written_labels = [0] * len(human_written)
    data = data._append(human_written, ignore_index=True)
    data_labels = human_written_labels + [1] * len(data)
    return data, data_labels


def write_to_file(X, y, labels, difficulty,  original_input_file, output_file):
    with open(output_file, "a") as f:
        for i in range(len(X)):
            f.write(json.dumps({"code": X[i],
                                "label": y[i],
                                "label_name": labels[y[i]],
                                "difficulty": difficulty,
                                "original_source": original_input_file}) + "\n")


for file in os.listdir("results"):
    if not file.endswith("207.jsonl"):
        continue
    input_file = os.path.join("results", file)
    convert_for_codebert(input_file)










