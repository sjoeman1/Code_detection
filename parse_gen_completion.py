import argparse
import json
from utils import parse_code_snippet
from tqdm import tqdm

# add argument for dataset
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="results/CodeLlama-70b-Instruct-hf-apps_introductory.jsonl")
parser.add_argument('--to_parse', default="gen_completion", choices=["gen_completion", "gen_text"])
args = parser.parse_args()

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]
    if args.to_parse == "gen_text":
        for i, instance in enumerate(data):
            for key in ['human_gen_text', 'machine_gen_text']:
                data[i][key] = json.loads(instance[key])

if args.to_parse == "gen_completion":
    for i,instance in enumerate(data):
        parsed_codes = parse_code_snippet(prompt=instance['question'], raw_o=instance['gen_completion'])
        data[i]['parsed_codes'] = parsed_codes
        if parsed_codes.startswith('# CANNOT'):
            print(i+1)

    outputs = []
    for ins in data:
        outputs.append(json.dumps(ins))

    with open(f'{args.dataset}', "w") as f:
        f.write("\n".join(outputs) + "\n")

elif args.to_parse == "gen_text":
    for i, instance in tqdm(enumerate(data)):
        # load the openai chat completions and parse the code snippet
        for key in ['human_gen_text', 'machine_gen_text']:
            for j, choice in enumerate(instance[key]['choices']):
                parsed_codes = parse_code_snippet(prompt=instance['question'], raw_o=choice['message']['content'])
                data[i][key]['choices'][j]['message']['content'] = parsed_codes
                if parsed_codes.startswith('# CANNOT'):
                    print(i + 1)

    outputs = []
    for ins in data:
        outputs.append(json.dumps(ins))

    with open(f'test.jsonl', "w") as f:
        f.write("\n".join(outputs) + "\n")