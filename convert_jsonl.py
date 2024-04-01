# for each folder in the apps directory, combine the samples.json file with the output files to create a jsonl file
# out of the samples.json file, take the question and the solutions and put them in the jsonl file as "question" and "gold_completion"

import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gemma-7b-it",
                    choices=["CodeLlama-70b-Instruct-hf", "gemma-7b-it", "Mixtral-8x7B-Instruct-v0.1"]
                    )
parser = parser.parse_args()

for directory in os.listdir('data/apps'):
    print(directory)
    if not os.path.isdir(f'data/apps/{directory}'):
        continue
    # out of the samples.json take the question and solution
    difficulty = directory[:-6]
    with open(f'data/apps/{directory}/{difficulty}_samples.json', 'r') as json_f:
        samples = json.load(json_f)
        for sample in samples:
            question = sample['question']
            gold_completion = sample['solutions'][0]
            problem_id = sample['problem_id']
            # get the txt file in the gemma-7b-it folder with the current problem_id if it exists
            if not os.path.isfile(f'data/apps/{directory}/{parser.model_name}/{problem_id}.txt'):
                continue
            with open(f'data/apps/{directory}/{parser.model_name}/{problem_id}.txt', 'r') as txt_f:
                gen_completion = txt_f.read()

            # write the question and the gold_completion and gen completion to a jsonl file
            with open(f'data/apps/{directory}/{parser.model_name}-apps_{difficulty}_207.jsonl', 'a') as jsonl_f:
                jsonl_f.write(json.dumps({"problem_id": problem_id, "question": 'Provide me the Python3 codes for solving the question: ' + question, "gold_completion": gold_completion, "gen_completion": gen_completion}) + '\n')