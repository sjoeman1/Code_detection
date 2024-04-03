# use meta features of the dataset to perform classification
import wandb
import argparse
import json
import pandas as pd
import numpy as np
import keyword
built_in_functions = dir(__builtins__)

from utils import standard_tokenizer, is_comment

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="results/gemma-7b-it-apps_interview_207.jsonl")
parser.add_argument('--classifier', default="LR", choices=['SCM', 'LR', 'MNB', 'XGB'])
args = parser.parse_args()
# args.ngram_range = tuple(args.ngram_range)
print(args)

clf_name = args.classifier
dataset_name = args.dataset.split('/')[-1].split('.')[0]
run_name = f"{dataset_name}_{clf_name}"

# load the dataset
with open(args.dataset, 'r') as f:
    dataset = [json.loads(line) for line in f.readlines()]

wandb.init(project='Code_Detection_Features', config=args, name=run_name, tags= [dataset_name.split('_')[1], dataset_name.split('_')[0]])
config = wandb.config


# create dataframe
dataset_df = pd.DataFrame(dataset)
#remove rows with # CANNOT PARSE
dataset_df = dataset_df[~dataset_df['parsed_codes'].str.startswith("# CANNOT")]
print(dataset_df)
human_written = dataset_df['gold_completion']
machine_generated = dataset_df['parsed_codes']
print(len(human_written), len(machine_generated))
human_written_labels = [0] * len(human_written)
machine_generated_labels = [1] * len(machine_generated)


def extract_features(series):
    # tokenize data
    df = series.to_frame(name='codes')
    df['tokenized'] = df['codes'].apply(lambda x: standard_tokenizer(x))
    # extract features from the data
    df['amount_of_tokens'] = df['tokenized'].apply(lambda x: len(x))

    df['amount_of_comments'] = df['tokenized'].apply(lambda x: len([word for word in x if is_comment(word)]))
    df['avg_len_comments'] = df['tokenized'].apply(lambda x: len_comments(x))
    df['space_after_comment'] = df['tokenized'].apply(
        lambda x: len([word for word in x if is_comment(word) and word.startswith('# ')]))
    df['no_space_after_comment'] = df['tokenized'].apply(
        lambda x: len([word for word in x if is_comment(word) and not word.startswith('# ')]))
    df['amount_of_code'] = df['tokenized'].apply(lambda x: len([word for word in x if not is_comment(word)]))
    df['amount_of_newlines'] = df['tokenized'].apply(lambda x: len([word for word in x if word == '#NEWLINE#']))
    df['amount_of_indents'] = df['tokenized'].apply(lambda x: len([word for word in x if word == '#INDENT#']))
    df['amount_of_defs'] = df['tokenized'].apply(lambda x: len([word for word in x if word == 'def']))
    df['avg_len_line'] = df['codes'].apply(lambda x: np.mean([len(line) for line in x.split('\n') if not line.startswith('#')]))
    # df['amount_of_builtins'] = df['tokenized'].apply(lambda x: len([word for word in x if word in built_in_functions]))
    # df['amount_of_keywords'] = df['tokenized'].apply(lambda x: len([word for word in x if keyword.iskeyword(word)]))
    # df['amount_of_soft_keywords'] = df['tokenized'].apply(lambda x: len([word for word in x if keyword.issoftkeyword(word)]))

    return df


def len_comments(tokens):
    comments = [token for token in tokens if is_comment(token)]
    sum = np.sum([len(comment) for comment in comments]) / len(comments) if len(comments) > 0 else 0
    return sum

def standardize_df(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df


human_written = extract_features(human_written)
machine_generated = extract_features(machine_generated)

# combine the two lists
# human_written and machine_generated are lists of strings
data = pd.concat([human_written, machine_generated], ignore_index=True)
print(data)
[print(i, code) for i, code in enumerate(data['codes'])]
data_labels = human_written_labels + machine_generated_labels
labels = ['human_written', 'machine_generated']
print(len(data), len(data_labels))


data_features = data.drop(columns=['codes', 'tokenized'])
# normalize the data
standardize_df(data_features)

wandb.log({'data_features': list(data_features.columns)})

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(data_features, data_labels, shuffle=True, test_size=0.2, random_state=42)

# train logistic regression on features
clf = LogisticRegression()
clf.fit(X_train, y_train)

# evaluate the model
y_pred = clf.predict(X_test)
y_probas = clf.predict_proba(X_test)
# print the classification report
report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
print(report)
wandb.log(report)
wandb.sklearn.plot_classifier(clf, X_train, X_test, y_train, y_test, y_pred, y_probas, labels, model_name=clf_name, feature_names=data_features.columns)


