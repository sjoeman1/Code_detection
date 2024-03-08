# Code_detection

Step 1: run generate_openai.py to generate codes based on questions from humaneval dataset (https://github.com/openai/human-eval) based on one specific model version, like chatgpt-3, 3.5, 4

Step 2: run regenerate_gpt4.py to do regeneration for detection if the previous step generates codes on gpt-4. Then run load_data_gpt4.ipynb for parsing.

Step 3: run detect_detectgpt4code.ipynb for detection. Also, the commericial baselines are detect_gptzero.py, detect_openai.py. And my_detector_gpt35or4.ipynb, my_detector_whitebox.ipynb serve as baselines for DNA-GPT.