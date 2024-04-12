import openai
import os
import numpy as np
import re
import pandas as pd
import time
import json
from datetime import datetime

DATA_PATH = "./query_data/"
quiz_info_file = pd.read_csv(os.path.join(DATA_PATH, "quiz_info_ITP1.csv"), encoding="utf-8-sig")
data_file = pd.read_excel(os.path.join(DATA_PATH, "accepted_data.xlsx"))
filename = "./error_description/description.txt"
with open(filename, "r", encoding="utf-8") as file:
    error_description = file.read()
filename = DATA_PATH + "possible_errors_per_problem.json"
with open(filename, 'r') as f:
    possible_errors_per_problem = json.load(f)


def get_quiz_info(quiz_id):
    programming_quiz = "\n[Programming Quiz]: " + quiz_info_file.loc[quiz_info_file['quiz_name'] == quiz_id, 'quiz'].values[0] + "\n"
    programming_input = "[Input]: " + quiz_info_file.loc[quiz_info_file['quiz_name'] == quiz_id, 'quiz_input'].values[0] + "\n"
    programming_output = "[Output]: " + quiz_info_file.loc[quiz_info_file['quiz_name'] == quiz_id, 'quiz_output'].values[0] + "\n"
    programming_sample_in = "[Sample Input]: " + quiz_info_file.loc[quiz_info_file['quiz_name'] == quiz_id, 'sample_input'].values[0] + "\n"
    programming_sample_out = "[Sample Output]: " + quiz_info_file.loc[quiz_info_file['quiz_name'] == quiz_id, 'sample_output'].values[0] + "\n"

    quiz_information = programming_quiz + programming_input + programming_output + programming_sample_in + programming_sample_out
    return quiz_information

class Prompt:
    def __init__(self, row):
        self.submission_code = "\n" + row["code"]
        self.logical_error = "A logical error refers to a situation where the program runs, but it operates differently from the intention.\n"
        self.problem_id = row["problem_id"].strip()
        self.quiz_information =  get_quiz_info(self.problem_id)
        self.instruction_start = "\n###\nExecute the following command, considering the common mistakes often made by beginners in programming."
        self.augmentation_format = "Modify the code that will be provided so that only a ({}) error occurs by changing just one line, then provide the entire code including the changed part. However, removing any part of the code is restricted.\n"
        self.instruction_end = "Please respond in the form of a single JSON file list in which the generated data exists, and where only 'error_type' and 'code' are the keys. For 'code', please write the entire code.\n###\n\n"

    def make_prompt(self):
        augmentation_ment = ""
        for item in possible_errors_per_problem:
            if item['problem_id'].strip() == self.problem_id:
                possible_error_list = item['possible_error']
                break
        for index, error in enumerate(possible_error_list):
            augmentation_ment += str(index+1) + ". " + self.augmentation_format.format(error)

        instruction = self.instruction_start + augmentation_ment + self.instruction_end
        code = "[Submission code]: " + self.submission_code


        return self.logical_error + error_description + self.quiz_information + code + instruction

openai.api_key = '{INSERT_YOUR_API_KEY}'

SYSTEM_PROMPT = f"""You are a senior developer with expertise in programming code.
Based on the given description and the correct code, you augment the error code that contains logical errors."""

def generate_ans(question):
    model = "gpt-3.5-turbo"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    response = openai.ChatCompletion.create(
        temperature=0.6,
        top_p=0.7,
        model=model,
        messages = messages
    )

    answer = response['choices'][0]['message']['content']
    return answer

def main():
    total = data_file.shape[0]
    results = []

    try:
        for index, row in data_file.iterrows():
            prompt = Prompt(row)
            final_prompt = prompt.make_prompt()
            print(final_prompt)

            result = generate_ans(final_prompt)
            print(str(index+1) + " / " + str(total))
            print(result)

            results.append({
                'user_id': row['user_id'],
                'judge_id': row['judge_id'],
                'problem_id': row['problem_id'],
                'code': row['code'],
                'augmented_codes': result
            })

            if index == 4: break
            
            time.sleep(3)

    finally:
        result_df = pd.DataFrame(results)
        result_df.to_excel(DATA_PATH+'/result/'+'{INSERT_SAVE_PATH}.xlsx', index=False)

main()