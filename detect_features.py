# use meta features of the dataset to perform classification
import wandb
import argparse
import json
import pandas as pd
import numpy as np
import keyword
built_in_functions = dir(__builtins__)

from utils import standard_tokenizer, is_comment

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import xgboost as xgb

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset', default="code_bert/codebert_train.jsonl")
parser.add_argument('--test_dataset', default="code_bert/codebert_competition_test.jsonl")
parser.add_argument('--classifier', default="Feature_LR",
                    choices=['Feature_SCM', 'Feature_LR', 'Feature_MNB', 'Feature_XGB'])
args = parser.parse_args()
# args.ngram_range = tuple(args.ngram_range)
print(args)

clf_name = args.classifier
dataset_name = args.train_dataset.split('/')[-1][:-6]
run_name = f"{dataset_name}_{clf_name}"

# load the dataset
with open(args.train_dataset, 'r') as f:
    dataset = [json.loads(line) for line in f.readlines()]

wandb.init(project='Code_Detection_Features', config=args, name=run_name, tags= ['full_dataset'])
config = wandb.config


# create dataframe
dataset_df = pd.DataFrame(dataset)
print(dataset_df.columns)



def extract_features(series):
    # tokenize data
    df = series.to_frame(name='code')
    df['tokenized'] = df['code'].apply(lambda x: standard_tokenizer(x))
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
    df['avg_len_line'] = df['code'].apply(lambda x: np.mean([len(line) for line in x.split('\n') if not line.startswith('#')]))
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


data = extract_features(dataset_df['code'])

print(data.columns)
# [print(i, code) for i, code in enumerate(data['code'])]
data_labels = dataset_df['label']
labels = ['human_written', 'machine_generated']
print(len(data), len(data_labels))


data_features = data.drop(columns=['code', 'tokenized'])
print(data_features)
# normalize the data
standardize_df(data_features)

wandb.log({'data_features': list(data_features.columns)})

X_train, y_train = data_features, data_labels


match config['classifier']:
    case "Feature_SCM":
        clf = SVC(kernel='poly', C=10)
    case "Feature_LR":
        clf = LogisticRegression(max_iter=1000)
    case "Feature_MNB":
        clf = MultinomialNB()
    case "Feature_XGB":
        clf = xgb.XGBClassifier(objective='binary:logistic', max_depth=10, n_estimators=180, gamma=1)
    case _:
        print("Invalid model name")
        wandb.finish(exit_code=1)
        exit(1)


# evaluate the model
clf.fit(X_train, y_train)

# load the test dataset
with open(args.test_dataset, 'r') as f:
    test_dataset = [json.loads(line) for line in f.readlines()]

test_dataset_df = pd.DataFrame(test_dataset)
test_data = extract_features(test_dataset_df['code'])
test_data_labels = test_dataset_df['label']
test_data_features = test_data.drop(columns=['code', 'tokenized'])
standardize_df(test_data_features)
X_test, y_test = test_data_features, test_data_labels

y_pred = clf.predict(X_test)
if not config['classifier'] == "Feature_SCM":
    y_probas = clf.predict_proba(X_test)
#print the classification report
report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
print(report)
wandb.log(report)
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
wandb.sklearn.plot_learning_curve(clf, X_train, y_train)
wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
wandb.sklearn.plot_summary_metrics(clf, X_train, y_train, X_test, y_test)

if not config['classifier'] == "Feature_SCM":
    wandb.sklearn.plot_roc(y_test, y_probas, labels)
    wandb.sklearn.plot_feature_importances(clf, data_features.columns)
    features = wandb.sklearn.calculate.feature_importances(clf, data_features.columns)

    # get top and bottom 10 features out of features and plot them in wandb
    top_features = features.value.data[:10]
    bottom_features = features.value.data[-10:]
    all_features = top_features + bottom_features
    table = wandb.Table(data=all_features, columns=features.value.columns)
    chart = wandb.visualize("top_20_features", table)
    wandb.log({"top_20_features_log": chart})

