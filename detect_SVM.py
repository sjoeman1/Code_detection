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

import xgboost as xgb

from utils import tag_comment_tokenizer, hide_comment_tokenizer, tokenize_comment_tokenizer, tokenize_only_comment, \
    standard_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="code_bert/codebert_train.jsonl")
parser.add_argument('--difficulty', default="competition", choices=["introductory", "interview", "competition"])
parser.add_argument('--classifier', default="LR", choices=['SCM', 'LR', 'MNB', 'XGB'])
parser.add_argument('--max_features', default=8000, type=int)
parser.add_argument('--vectorizer', default='TfidfVectorizer', choices=['TfidfVectorizer', 'CountVectorizer'])
parser.add_argument('--ngram_range', default=(1, 4), type=tuple, nargs=2)
parser.add_argument('--tokenizer', default='tokenize_comment',
                    choices=['tokenize_comment', 'tag_comment', 'hide_comment', 'standard_tokenizer', 'comment_only'])
args = parser.parse_args()
print(args)

clf_name = args.classifier
dataset_name = args.dataset.split('/')[-1][:-6]
run_name = f"{dataset_name}_{clf_name}"

# load train and test data
with open(args.dataset, "r") as f:
    train_dataset = [json.loads(line) for line in f.readlines()]

with open(f"code_bert/codebert_{args.difficulty}_test.jsonl", "r") as f:
    test_dataset = [json.loads(line) for line in f.readlines()]


wandb.init(project='Code_Detection_full_dataset', config=args, name=run_name)
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
X_test = vectorizer.transform(X_test)

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

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
if not config['classifier'] == "SCM":
    y_probas = clf.predict_proba(X_test)


# print the classification report
report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
print(report)
wandb.log(report)


wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
wandb.sklearn.plot_learning_curve(clf, X_train, y_train)
wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
wandb.sklearn.plot_calibration_curve(clf, X_train, y_train, config['classifier'])
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
wandb.sklearn.plot_summary_metrics(clf, X_train, y_train, X_test, y_test)

if not config['classifier'] == "SCM":
    wandb.sklearn.plot_roc(y_test, y_probas, labels)
    wandb.sklearn.plot_feature_importances(clf, vectorizer.get_feature_names_out())

