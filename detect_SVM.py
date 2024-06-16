import numpy as np

import wandb
import argparse
import json
import pandas as pd
import sys

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

import xgboost as xgb

from utils import tag_comment_tokenizer, hide_comment_tokenizer, tokenize_comment_tokenizer, tokenize_only_comment, \
    standard_tokenizer, codebert_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset', default="code_bert/codebert_train.jsonl code_bert/codebert_val.jsonl")
# parser.add_argument('--difficulty', default="competition", choices=["introductory", "interview", "competition"])
parser.add_argument('--testset', default="code_bert/codebert_competition_test.jsonl")
parser.add_argument('--classifier', default="XGB", choices=['SCM', 'LR', 'MNB', 'XGB'])
parser.add_argument('--max_df', default=0.5, type=float)
parser.add_argument('--min_df', default=0.0155, type=float)
parser.add_argument('--vectorizer', default='TfidfVectorizer', choices=['TfidfVectorizer', 'CountVectorizer'])
parser.add_argument('--ngram_range', default=(1, 4), type=tuple, nargs=2)
parser.add_argument('--tokenizer', default='tokenize_comment',
                    choices=['tokenize_comment', 'tag_comment', 'hide_comment', 'standard_tokenizer', 'comment_only',
                             'codebert_tokenizer'])
args = parser.parse_args()
print(args)

clf_name = args.classifier
args.train_dataset = args.train_dataset.split(' ')
dataset_name = [ds.split('/')[-1][:-6] for ds in args.train_dataset]
run_name = f"{dataset_name}_{clf_name}"
print(run_name)

# load the dataset
train_dataset = []
for dataset_file in args.train_dataset:
    with open(dataset_file, 'r') as f:
        train_dataset += [json.loads(line) for line in f.readlines()]


with open(args.testset, "r") as f:
    test_dataset = [json.loads(line) for line in f.readlines()]


wandb.init(project='Code_Detection_full_dataset', config=args, name=run_name, tags=["full_dataset", "V4", "feature_importance"])
config = wandb.config

df_train = pd.DataFrame(train_dataset)
df_test = pd.DataFrame(test_dataset)

X_train = df_train['code']
y_train = df_train['label']



print(df_train['label'].value_counts())

X_test = df_test['code']
y_test = df_test['label']

print(df_test['label'].value_counts())

labels = ['human_written', 'machine_generated']
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
    case "codebert_tokenizer":
        tokenizer = codebert_tokenizer
    case _:
        print("Invalid tokenizer name")

# # test the tokenizer
# train_ = df_train['code'][6]
# print(train_)
# print(tokenizer(train_))
#
# sys.exit(1)

match config['vectorizer']:
    case "CountVectorizer":
        vectorizer = CountVectorizer(tokenizer=tokenizer, min_df=config.min_df, max_df=config.max_df,
                                     ngram_range=tuple(config.ngram_range))
    case "TfidfVectorizer":
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, min_df=config.min_df, max_df=config.max_df,
                                     ngram_range=tuple(config.ngram_range))
    case _:
        print("Invalid vectorizer name")
        wandb.finish(exit_code=1)
        exit(1)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(X_train.shape, X_test.shape)

match config['classifier']:
    case "SCM":
        clf = SVC(kernel='poly', C=10, probability=True, class_weight='balanced')
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

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_probas = clf.predict_proba(X_test)


# print the classification report
report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)


wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
wandb.sklearn.plot_learning_curve(clf, X_train, y_train)
wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
wandb.sklearn.plot_calibration_curve(clf, X_train, y_train, config['classifier'])
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
wandb.sklearn.plot_summary_metrics(clf, X_train, y_train, X_test, y_test)
wandb.sklearn.plot_roc(y_test, y_probas, labels)
report['roc_auc_score'] = roc_auc_score(y_test, y_probas[:, 1])
print(report)

if not config['classifier'] == "SCM":
    wandb.sklearn.plot_feature_importances(clf, vectorizer.get_feature_names_out())
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Get coefficients
    if config['classifier'] == "XGB":
        coefficients = clf.feature_importances_
    else:
        coefficients = clf.coef_[0]

    # Create a DataFrame with feature names and their corresponding coefficients
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': coefficients})

    # Sort by the absolute value of the coefficients
    feature_importances = feature_importances.reindex(
        feature_importances.importance.abs().sort_values(ascending=False).index)

    # Plot the top 10 features
    top_features = feature_importances.head(20)
    top_features.sort_values(by="importance", ascending=True, inplace=True)

    plt.figure(figsize=(9, 6))
    plt.barh(top_features['feature'], top_features['importance'], align='center')
    if config['classifier'] == "XGB":
        plt.title("Feature importances in the XGBoost model")
    if config['classifier'] == "LR":
        plt.title("Feature importances in the Logistic Regression model")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    wandb.log({"feature_importance": wandb.Image(plt)})

    plt.show()


wandb.log(report)

