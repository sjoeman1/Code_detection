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
parser.add_argument('--dataset', default="results/gemma-7b-it-apps")
parser.add_argument('--difficulty', default="interview", choices=["introductory", "interview", "competition"])
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

# load the introductory, interview and competition datasets
with open(args.dataset + '_introductory_207.jsonl', 'r') as introductory_f:
    introductory = [json.loads(line) for line in introductory_f.readlines()]
with open(args.dataset + '_interview_207.jsonl', 'r') as interview_f:
    interview = [json.loads(line) for line in interview_f.readlines()]
with open(args.dataset + '_competition_207.jsonl', 'r') as competition_f:
    competition = [json.loads(line) for line in competition_f.readlines()]

wandb.init(project='Code_Detection_full_dataset', config=args, name=run_name)
config = wandb.config


# create dataframe for each dataset
introductory_df = pd.DataFrame(introductory)
interview_df = pd.DataFrame(interview)
competition_df = pd.DataFrame(competition)

#remove rows with # CANNOT PARSE
introductory_df = introductory_df[~introductory_df['parsed_codes'].str.startswith("# CANNOT")]
interview_df = interview_df[~interview_df['parsed_codes'].str.startswith("# CANNOT")]
competition_df = competition_df[~competition_df['parsed_codes'].str.startswith("# CANNOT")]

# print the dataset
print(introductory_df)
print(interview_df)
print(competition_df)

# create labels
labels = ['human_written', 'machine_generated']
introductory_human_written = introductory_df['gold_completion']
introductory_machine_generated = introductory_df['parsed_codes']
introductory_human_written_labels = [0] * len(introductory_human_written)
introductory_machine_generated_labels = [1] * len(introductory_machine_generated)

interview_human_written = interview_df['gold_completion']
interview_machine_generated = interview_df['parsed_codes']
interview_human_written_labels = [0] * len(interview_human_written)
interview_machine_generated_labels = [1] * len(interview_machine_generated)

competition_human_written = competition_df['gold_completion']
competition_machine_generated = competition_df['parsed_codes']
competition_human_written_labels = [0] * len(competition_human_written)
competition_machine_generated_labels = [1] * len(competition_machine_generated)

# split in train and test set
X_train_introductory, X_test_introductory, y_train_introductory, y_test_introductory = train_test_split(introductory_human_written._append(introductory_machine_generated, ignore_index=True), introductory_human_written_labels + introductory_machine_generated_labels, shuffle=True, test_size=0.2, random_state=42)
X_train_interview, X_test_interview, y_train_interview, y_test_interview = train_test_split(interview_human_written._append(interview_machine_generated, ignore_index=True), interview_human_written_labels + interview_machine_generated_labels, shuffle=True, test_size=0.2, random_state=42)
X_train_competition, X_test_competition, y_train_competition, y_test_competition = train_test_split(competition_human_written._append(competition_machine_generated, ignore_index=True), competition_human_written_labels + competition_machine_generated_labels, shuffle=True, test_size=0.2, random_state=42)

# combine the train datasets
X_train = X_train_introductory._append(X_train_interview)._append(X_train_competition, ignore_index=True)
y_train = y_train_introductory + y_train_interview + y_train_competition

if args.difficulty == 'introductory':
    X_test = X_test_introductory
    y_test = y_test_introductory
elif args.difficulty == 'interview':
    X_test = X_test_interview
    y_test = y_test_interview
else:
    X_test = X_test_competition
    y_test = y_test_competition

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

