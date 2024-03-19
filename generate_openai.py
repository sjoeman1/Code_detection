import argparse
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import tqdm
import json
import os
import torch, openai
import random
from utils import get_davinci003_response, get_chatgpt_qa_response, get_gpt4_qa_response

key1 = "" 
openai.api_key = key1

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="HumanEval.jsonl")
parser.add_argument('--model', default="gpt4") # "gpt-3.5-turbo", "gpt-4-0314",
parser.add_argument('--max_new_tokens', default=1024, type=int)
parser.add_argument('--total_instances', default=200, type=int)
parser.add_argument('--temperature', default= 0.7, type=float)
args = parser.parse_args()

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n") ]
len(data)

output_file = f"results/{args.model}_" + args.dataset + ".jsonl"

random.seed(43)
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        num_curr_outputs = len(f.read().strip().split("\n"))
else:
    num_curr_outputs = 0

print("Skipping {} instances".format(num_curr_outputs))
data = data[num_curr_outputs:args.total_instances]
print("Total instances: {}".format(len(data)))

if args.model == "davinci_003":
    openai_fn = get_davinci003_response
elif args.model == "chatgpt":
    openai_fn = get_chatgpt_qa_response
elif args.model == "gpt4":
    openai_fn = get_gpt4_qa_response
else:
    raise NotImplementedError

outputs = []
for idx, dd in tqdm.tqdm(enumerate(data), total= len(data) ): # 
    question = 'Provide me the Python3 codes for solving the question: ' + dd['prompt']
    if args.model == "davinci_003":
        question = 'Provide me the Python3 codes for solving the question. No need to make comments: ' + dd['question']
    gen_text = openai_fn(prompt_text=question, temperature=args.temperature, max_tokens=args.max_new_tokens)


    outputs.append(json.dumps({
        "question": question,
        "gold_completion": dd['canonical_solution'],
        "gen_completion": gen_text
    }))

    with open(output_file, "a") as f:
        f.write("\n".join(outputs) + "\n")
    outputs = []

with open(output_file, "a") as f:
    f.write("\n".join(outputs) + "\n")
    outputs = []
