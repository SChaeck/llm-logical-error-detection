import openai
import os
import numpy as np
import re
import pandas as pd
import time
from datetime import datetime

DATA_PATH = "./data/query_data"
quiz_info_file = pd.read_csv(os.path.join(DATA_PATH, "quiz_info_ITP1.csv"), encoding="utf-8-sig")
data_path_list = [os.path.join(DATA_PATH, "logical_type") + "/" + file for file in os.listdir(os.path.join(DATA_PATH, "logical_type"))]
doc_path_list = [os.path.join(os.getcwd(), "memo_per_error") + "/" + file for file in os.listdir(os.path.join(os.getcwd(), "memo_per_error"))]

document_logical_error_list = []
for doc_path in doc_path_list:
    with open(doc_path, 'r', encoding='utf-8') as file:
        document_logical_error_list.append(file.read())

def get_quiz_info(quiz_id):
    text = "\n###Please determine whether the logical error in the [Submission code] is the logical error described in <Description and examples of one error among the ten types of logical errors/> by looking at the upcoming <Problem that needs to be checked/>. ###\n\n"
    programming_quiz = "[Programming Quiz]: " + quiz_info_file.loc[quiz_info_file['quiz_name'] == quiz_id, 'quiz'].values[0] + "\n"
    programming_input = "[Input]: " + quiz_info_file.loc[quiz_info_file['quiz_name'] == quiz_id, 'quiz_input'].values[0] + "\n"
    programming_output = "[Output]: " + quiz_info_file.loc[quiz_info_file['quiz_name'] == quiz_id, 'quiz_output'].values[0] + "\n"
    programming_sample_in = "[Sample Input]: " + quiz_info_file.loc[quiz_info_file['quiz_name'] == quiz_id, 'sample_input'].values[0] + "\n"
    programming_sample_out = "[Sample Output]: " + quiz_info_file.loc[quiz_info_file['quiz_name'] == quiz_id, 'sample_output'].values[0] + "\n"

    quiz_information = programming_quiz + programming_input + programming_output + programming_sample_in + programming_sample_out
    return text, quiz_information

class Prompt:
    def __init__(self, row):
        self.submission_code = "\n" + row["code"] + "\n"
        self.answer_trigger = "\nA: Let's think step by step.\n"
        self.CoT_error_code = "Q: Please look at [Submission code] written in C++ and explain it according to the flow. Print the sequence in numeric format."
        self.quiz_instruction, self.quiz_information =  get_quiz_info(row["problem_id"].strip())
        self.logical_error = "\nForget the definition and classification of logical errors you knew, and remember the definition and classification of logical errors that I define. A logical error refers to a situation where the program runs, but it operates differently from the intention. From now on, we define the classification of logical errors as Input, Output, Variable, Computation, Condition, Branching, Loop, Array/String, Function, Conceptual error. Therefore, you can determine the logical error of the program code as one of these ten classifications.\n"
        self.instruction = "\n###\nHint for finding logical errors: First, understand a specific description of one out of the ten logical errors, along with the code and situation where this logical error occurs. You then need to check whether this error occurs in the [Submission code]. To verify the error, consider whether the code can receive the value described in [Input] through the 'standard input stream' using the 'keyboard', and whether the code's algorithm operates in accordance with the intent of the [Programming Quiz]. The [Sample Input] is an example of [Input] and represents the format to be entered into the program via the 'keyboard'. As a final step, verify whether the value described in [Output] is output to the 'console' through the 'standard output stream'. The [Sample Output] is an example of [Output] and represents the format to be output by the program to the 'console'.\n###\n"

        self.ToT = """
Imagine three different experts are answering this question.
All experts will write down 1 step of their thinking,
then share it with the group.
Then all experts will go on to the next step, etc.
If any expert realises they're wrong at any point then they leave.
The question is
"""
        self.do_detection = "Q: Considering the [Programming Quiz], [Input], [Output], [Sample Input], [Sample Output], and [Submission code], could the error described in <Description and examples of one error among the ten types of logical errors/> occur in the [Submission code]?"
        
        self.baseline = "\nFirst, document the reason for your consideration, then conclude the sentence with 'Yes' or 'No'. Should the logical error be correct but fail to align with the description in <Description and examples of one error among the ten types of logical errors/>, the response should be 'No'."

        self.answer_format = "\nA: This code "

    def first_prompt(self):
        return self.CoT_error_code + self.submission_code + self.answer_trigger

    def second_prompt(self, first_prompt):
        second_prompt_list = []
        # code_flow = "The flow of [Submission code] is as follows:\n" + first_prompt
        code_flow = ""
        quiz = "\n\n<Problem that needs to be checked>" + self.quiz_instruction + self.quiz_information
        code = "[Submission code]: " + self.submission_code
        
        for document_logical_error in document_logical_error_list:
            second_prompt_list.append(self.logical_error + self.instruction + document_logical_error + quiz + code + code_flow + self.ToT + self.do_detection + self.baseline + self.answer_format)

        return second_prompt_list 
    
openai.api_key = '{INSERT_YOUR_API_KEY}'

SYSTEM_PROMPT = f"""You are a programming expert who specializes in analyzing logical errors in code. 
Based on the given conditions, you identify the parts of the code that have logical errors, and determine the type of the error."""

def generate_ans_list(question_list):
    answer_list = []
    for idx, question in enumerate(question_list):
        # if idx != 0 : 
        #     answer_list.append("NULL")
        #     continue
        answer_list.append(generate_ans(question))
        time.sleep(10)
    return answer_list
def generate_ans(question):
    model = "gpt-3.5-turbo"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    response = openai.ChatCompletion.create(
        temperature=0.2,
        top_p=0.1,
        model=model,
        messages = messages
    )

    answer = response['choices'][0]['message']['content']
    return answer

def main():
    for file in data_path_list:
        sample_data_file = pd.read_excel(file)
        print("############################" + os.path.basename(file) + "############################")
        results = [] 
        total = sample_data_file.shape[0]
        for index, row in sample_data_file.iterrows():
            prompt = Prompt(row)
            # first_prompt = prompt.first_prompt()
            # first_result = generate_ans(first_prompt)
            first_result = ''
            second_prompt_list = prompt.second_prompt(first_result)
            second_result_list = generate_ans_list(second_prompt_list)

            print(str(index+1) + " / " + str(total))
            print("label:",row["logical_error_type"])
            print("predict:",second_result_list, "\n", time.strftime('%H:%M:%S'))         
            # print("Prompt: ", "\n\n====================================================================\n\n".join(second_prompt_list[:1]))
            try:
                results.append({
                    'logical_error_type': row['logical_error_type'],
                    'code': row['code'],
                    'judge_id': row['judge_id'],
                    'problem_id': row['problem_id'],
                    '(A)': second_result_list[0],
                    '(B)': second_result_list[1],
                    '(C)': second_result_list[2],
                    '(D)': second_result_list[3],
                    '(E)': second_result_list[4],
                    '(F)': second_result_list[5],       
                    '(G)': second_result_list[6],
                    '(H)': second_result_list[7],
                    '(I)': second_result_list[8],
                    '(J)': second_result_list[9],
                })
            except: 
                results.append({
                    'logical_error_type': row['logical_error_type'],
                    'code': row['code'],
                    'judge_id': row['judge_id'],
                    'problem_id': row['problem_id'],
                    'result': ", ".join(second_result_list),
                    '(A)': "",
                    '(B)': "",
                    '(C)': "",
                    '(D)': "",
                    '(E)': "",
                    '(F)': "",
                    '(G)': "",
                    '(H)': "",
                    '(I)': "",
                    '(J)': "",
                })
            result_df = pd.DataFrame(results)
            time.sleep(10)
        result_df.to_excel(DATA_PATH+'{INSERT_SAVE_PATH}/'+os.path.basename(file)[:3]+'_result.xlsx', index=False)
main()