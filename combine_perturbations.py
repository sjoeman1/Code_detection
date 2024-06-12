import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="code_bert/codebert_interview_test.jsonl")
parser.add_argument('--perturbations', default=4, type=int)
parser.add_argument('--mask_lines', default=8, type=int)

args = parser.parse_args()

output_file = f'FIM/{args.dataset.split("/")[1][:-6]}_FIM_human_combined_line_{args.mask_lines}_per_{args.perturbations * 14}.jsonl'

file_paths = []
for i in range(1, 15):
    file_paths.append(
        f'FIM/{args.dataset.split("/")[1][:-6]}_FIM_human_{i}_line_{args.mask_lines}_per_{args.perturbations}.jsonl')
print(file_paths)
key = 'FIM_code'  # Replace with the actual key containing the lists

def combine_jsonl_files(file_paths, key):
    # Read all files and store the entries in a list of lists
    all_entries = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            entries = [json.loads(line) for line in file]
            all_entries.append(entries)

    # Ensure all files have the same number of entries
    num_entries = len(all_entries[0])
    for entries in all_entries:
        if len(entries) != num_entries:
            raise ValueError("All files must have the same number of entries")

    # Combine the entries
    combined_entries = []
    for i in range(num_entries):
        combined_entry = all_entries[0][i].copy()  # Start with the first file's entry
        combined_lists = []
        for entries in all_entries:
            combined_lists.extend(entries[i][key])  # Concatenate the lists
        combined_entry[key] = combined_lists
        combined_entries.append(combined_entry)

    # Write the combined entries to a new file
    with open(output_file, 'w') as f:
        for entry in combined_entries:
            f.write(json.dumps(entry) + '\n')


combine_jsonl_files(file_paths, key)
