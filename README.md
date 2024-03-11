# Code_detection

Step 1: run generate_openai.py to generate codes based on questions from humaneval dataset (https://github.com/openai/human-eval) based on one specific model version, like chatgpt-3, 3.5, 4. Saved in ./results

Step 2: for baseline result on DNA-GPT, run regenerate_gpt4.py to do regeneration for detection if the previous step generates codes on gpt-4. Then run load_data_gpt4.ipynb for parsing. Saved in ./results. 

Step 3: for DetectGPT4Code result, run fill_in_the_middle.py for FIM task. You can specify dataset, FIM model version or mask_lines. Saved in ./results/. The number of FIM perturbation depends on your maximum GPU memory, so you might need to merge the results by runing fill_in_the_middle.py multiple times. For example, if fill_in_the_middle.py can only generate 4 perturbation per run, then you have to run it 10 times and combine their results together to get 40 perturbations.

Step 4: run detect_detectgpt4code.ipynb for detection. Also, the commericial baselines are detect_gptzero.py, detect_openai.py. And my_detector_gpt35or4.ipynb, my_detector_whitebox.ipynb serve as baselines for DNA-GPT.