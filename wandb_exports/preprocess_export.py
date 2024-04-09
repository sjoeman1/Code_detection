# preprocess the csv file so that the integers have 3 decimals and the names are preprocessed
import pandas as pd

df_name = "MNB_Tokenizer_wandb_export_2024-04-09T11_43_20.269+02_00.csv"
df = pd.read_csv(df_name)
print(df)
df = df.round(3)
print(df)

rename_dict = {"CodeLlama-70b-Instruct-hf-apps_competition" : "CodeLlama-70b Competition",
               "CodeLlama-70b-Instruct-hf-apps_interview": "CodeLlama-70b Interview",
               "CodeLlama-70b-Instruct-hf-apps_introductory": "CodeLlama-70b Introductory",
               "Mixtral-8x7B-Instruct-v0.1-apps_competition" : "Mixtral-8x7B Competition",
               "Mixtral-8x7B-Instruct-v0.1-apps_interview" : "Mixtral-8x7B Interview",
               "Mixtral-8x7B-Instruct-v0.1-apps_introductory" : "Mixtral-8x7B Introductory",
               "gemma-7b-it-apps_competition": "Gemma-7b Competition",
               "gemma-7b-it-apps_interview": "Gemma-7b Interview",
               "gemma-7b-it-apps_introductory": "Gemma-7b Introductory"}

# rename the names in the name column according to the dict
def rename(name):
    for key, value in rename_dict.items():
        if name.startswith(key):
            return value

df['Name'] = df['Name'].apply(rename)

# rename columns
df = df.rename({"Name": "Dataset", "macro avg.f1-score": "f1-score", "macro avg.precision": "precision", "macro avg.recall": "recall"})

print(df)
# save df
df.to_csv(df_name, index=False)