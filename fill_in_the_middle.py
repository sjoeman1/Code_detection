from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json
import random, os
from utils_batch import InfillingModel
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="results/CodeLlama-70b-Instruct-hf-apps_introductory.jsonl")
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--mask_lines', default=4, type=int)
parser.add_argument("--gpu", type=str, default="2")
parser.add_argument("--model_name", type=str, default="facebook/incoder-1B")
parser.add_argument("--run", type=str, default= "1")
args = parser.parse_args()

output_file = f'results/{args.dataset.split("/")[1]}_FIM_human_{args.run}_line_{args.mask_lines}.jsonl'
with open(args.dataset, 'r') as f:
    dataset = [json.loads(line) for line in f.readlines()]
    # remove cannot parse code snippets
    dataset = [x for x in dataset if not x['parsed_codes'].startswith('# CANNOT')]

print(args)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_properties(0).total_memory)
# setting device on GPU if available, else CPU
device = torch.device('cuda')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

if args.model_name == 'facebook/incoder-6B':
    args.half = True
else:
    args.half = False
infilling_model = InfillingModel(model_name=args.model_name, cuda=True, half=args.half, device=device)


def find_all(substring, string):
    start = 0
    while True:
        start = string.find(substring, start)
        if start == -1: return
        yield start
        start += len(substring)

def mask_code(pasrsed_codes, mask_lines = args.mask_lines):
    for _ in range(mask_lines):
        positions = list(find_all(substring='\n', string=pasrsed_codes))
        if positions == []:
            positions = list(find_all(substring=':', string=pasrsed_codes))
        if len(positions) < 2:
                continue # cornor case in which there is no \n and only one ':'
        mask_start = random.choice( range(len( (positions)) -1 ) )
        mask_start_position = positions[mask_start]
        mask_end_position = positions[mask_start+1]
        pasrsed_codes = pasrsed_codes[:mask_start_position] +  '<insert>' + pasrsed_codes[mask_end_position:]
    pasrsed_codes_masked = pasrsed_codes
    return pasrsed_codes_masked #, mask_end_position - mask_start_position

def norm_inserts_num(pasrsed_code_norm):
    max_num = 0
    for i, x in enumerate(pasrsed_code_norm):
        if len(list(find_all(substring='<insert>', string=x))) > max_num:
            max_num = len(list(find_all(substring='<insert>', string=x)))
            id = i

    new_res = []
    for x in pasrsed_code_norm:
        if len( list( find_all(substring='<insert>', string=x) ) ) < max_num:
            new_res.append( pasrsed_code_norm[id] )
        else:
            new_res.append( x )
    return new_res

if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        finished = [json.loads(line) for line in f.readlines()]
    dataset = dataset[len(finished):]

###### fill_in_middle_gold and fill_in_middle_parsed ######
for idx, ins in tqdm.tqdm(enumerate(dataset), total = len(dataset)):
    for input, output in [('parsed_codes','fill_in_middle_parsed'), ('gold_completion','fill_in_middle_gold')]:
        gold_completion_all = []
        if len(ins[input]) < 2500:
            for _ in range(args.batch_size):
                gold_codes_masked = mask_code(ins[input], mask_lines=args.mask_lines)
                gold_completion_all.append(gold_codes_masked[:2500])

            gold_completion_all = norm_inserts_num(gold_completion_all)
            parts_batch = [example.split("<insert>") for example in gold_completion_all]
            fill_in_middle_gold = infilling_model.batched_infill(parts_batch, max_to_generate=16*args.mask_lines, temperature=0.7)
            ins[output] = fill_in_middle_gold
        else:
            ins[output] = ['token exceeds 2500']

    with open(output_file, 'a') as f:
        f.write(json.dumps(ins) + '\n')



