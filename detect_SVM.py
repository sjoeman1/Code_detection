import numpy as np

import wandb
import argparse
import json
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, cross_val_predict

import xgboost as xgb

from utils import tag_comment_tokenizer, hide_comment_tokenizer, tokenize_comment_tokenizer, tokenize_only_comment, \
    standard_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="results/gemma-7b-it-apps_competition_207.jsonl")
parser.add_argument('--classifier', default="MNB", choices=['SCM', 'LR', 'MNB', 'XGB'])
parser.add_argument('--max_features', default=8000, type=int)
parser.add_argument('--vectorizer', default='TfidfVectorizer', choices=['TfidfVectorizer', 'CountVectorizer'])
parser.add_argument('--ngram_range', default=(1, 4), type=tuple, nargs=2)
parser.add_argument('--tokenizer', default='tokenize_comment',
                    choices=['tokenize_comment', 'tag_comment', 'hide_comment', 'standard_tokenizer', 'comment_only'])
parser.add_argument('--cv_folds', default=5, type=int)
args = parser.parse_args()
print(args)

clf_name = args.classifier
dataset_name = args.dataset.split('/')[-1][:-6]
run_name = f"{dataset_name}_{clf_name}"

# load the dataset
with open(args.dataset, 'r') as f:
    dataset = [json.loads(line) for line in f.readlines()]

wandb.init(project='Code_Detection_CV', config=args, name=run_name, tags= [dataset_name.split('_')[1], dataset_name.split('_')[0], 'V1'])
config = wandb.config


# create dataframe
df = pd.DataFrame(dataset)
#remove rows with # CANNOT PARSE
df = df[~df['parsed_codes'].str.startswith("# CANNOT")]
print(df)
human_written = df['gold_completion']
machine_generated = df['parsed_codes']
print(len(human_written), len(machine_generated))
human_written_labels = [0] * len(human_written)
machine_generated_labels = [1] * len(machine_generated)

# combine the two lists

# human_written and machine_generated are lists of strings
data = human_written._append(machine_generated, ignore_index=True)
print(data)
data_labels = human_written_labels + machine_generated_labels
labels = ['human_written', 'machine_generated']

print(len(data), len(data_labels))

X_train, y_train = data, data_labels

# create tokenizer
match config['tokenizer']:
    case "tokenize_comment":
        tokenizer = tokenize_comment_tokenizer
    case "tag_comment":
        tokenizer = tag_comment_tokenizer
    case "hide_comment":
        tokenizer = hide_comment_tokenizer
    case "standard_tokenizer":
        tokenizer = standard_tokenizer
    case "comment_only":
        tokenizer = tokenize_only_comment
    case _:
        print("Invalid tokenizer name")

# # test the tokenizer
# train_ = df['parsed_codes'][2]
# print(train_)
# print(tokenize_comment_tokenizer(train_))

match config['vectorizer']:
    case "CountVectorizer":
        vectorizer = CountVectorizer(tokenizer=tokenizer, max_features=config.max_features, ngram_range=tuple(config.ngram_range))
    case "TfidfVectorizer":
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, max_features=config.max_features, ngram_range=tuple(config.ngram_range))
    case _:
        print("Invalid vectorizer name")
        wandb.finish(exit_code=1)
        exit(1)

X_train = vectorizer.fit_transform(X_train)
# X_test = vectorizer.transform(X_test)

match config['classifier']:
    case "SCM":
        clf = SVC(kernel='poly', C=10)
    case "LR":
        clf = LogisticRegression(max_iter=1000)
    case "MNB":
        clf = MultinomialNB()
    case "XGB":
        clf = xgb.XGBClassifier(objective='binary:logistic', max_depth=10, n_estimators=180, gamma=1)
    case _:
        print("Invalid model name")
        wandb.finish(exit_code=1)
        exit(1)

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']
scores = cross_validate(clf, X_train, y_train, scoring=scoring, cv=config.cv_folds)
# y_pred = cross_val_predict(clf, X_train, y_train, cv=config.cv_folds)


# # print the classification report
# report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
# print(report)
print(scores)

avg_scores = {key: np.mean(value) for key, value in scores.items()}
print(avg_scores)
wandb.log(avg_scores)


# wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
# wandb.sklearn.plot_learning_curve(clf, X_train, y_train)
# wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
# wandb.sklearn.plot_calibration_curve(clf, X_train, y_train, config['classifier'])
# wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
# wandb.sklearn.plot_summary_metrics(clf, X_train, y_train, X_test, y_test)
#
# if not config['classifier'] == "SCM":
#     wandb.sklearn.plot_roc(y_test, y_probas, labels)
#     wandb.sklearn.plot_feature_importances(clf, vectorizer.get_feature_names_out())

