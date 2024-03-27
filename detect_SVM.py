import wandb
import code_tokenize as ctok
import argparse
import json
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="results/gemma-7b-it-apps_introductory.jsonl")
parser.add_argument('--classifier', default="MNB", choices=['SCM', 'LR', 'MNB'])
parser.add_argument('--max_features', default=1000, type=int)
parser.add_argument('--kernel', default='linear')
parser.add_argument('--vectorizer', default='CountVectorizer', choices=['TfidfVectorizer', 'CountVectorizer'])
parser.add_argument('--ngram_range', default=(1, 4), type=tuple, nargs=2)
parser.add_argument('--hide_comments', default=True, type=bool)
args = parser.parse_args()
# args.ngram_range = tuple(args.ngram_range)
print(args)

clf_name = args.classifier
dataset_name = args.dataset.split('/')[-1].split('.')[0]
run_name = f"{dataset_name}_{clf_name}"

# load the dataset
with open(args.dataset, 'r') as f:
    dataset = [json.loads(line) for line in f.readlines()]

wandb.init(project='Code_Detection', config=args, name=run_name, tags= [dataset_name.split('_')[1], dataset_name.split('_')[0]])
config = wandb.config


# create dataframe
df = pd.DataFrame(dataset)
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

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(data, data_labels, shuffle=True, test_size=0.2, random_state=42)

# create tokenizer
def tokenizer(text):
    tokenized = ctok.tokenize(text, lang='python', syntax_error="ignore")
    # convert to string and return
    tokenized = [str(token) for token in tokenized]
    if config.hide_comments:
        for i, token in enumerate(tokenized):
            if token.startswith("#") and not (token == "#NEWLINE#" or token == "#INDENT#" or token == "#DEDENT#"):
                tokenized[i] = "#COMMENT#"
    return tokenized

# test the tokenizer
print(tokenizer(X_train[0]))

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
        clf = SVC(kernel=config.kernel)
    case "LR":
        clf = LogisticRegression()
    case "MNB":
        clf = MultinomialNB()
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

