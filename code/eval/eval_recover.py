import os
import sys
sys.path.append('./')

from torch import sub
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import re
import json
import jsonlines
import pandas as pd
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import numpy as np
from func import *
from util import process_subwords, handlestring,sort_keywords_by_position
from alive_progress import alive_bar
from ltp import LTP
import spacy
from spacyutils import get_ents_en, get_ents_zh, add_pinyin, get_labelled_text
import ast
device = "cuda" # the device to load the model onto


#recover
instruction_head = "以下是一段文本，针对文本有一个问题，以及对应的回答：\n文本："
# instruction_question = "\n问题："
instruction_response = "\n对应的回答："
instruction_true="\n\n但由于隐私与安全考虑，该文本与问题中的某些概念或文本已经经过了替换，真实的文本与问题如下：\n真实的文本："
instruction_trueq="\n真实的问题："
instruction_tail = "\n\n你能否根据真实的新闻报道与问题，将上述回答进行改写，修改回答中的概念与文本使得修改后的回答符合真实文本中的信息，同时尽可能少地修改上述回答的结构与回答方式，如果原本的回答完全错误，则重新进行回答，如果原本的回答中的信息完全正确，则可以不进行修改，只需要给出修改后的回答。\n修改后的回答："


recover_system = "You are a helpful assistant."
recover_template = instruction_head + "{}"  + instruction_response + "{}" + instruction_true + "{}" + instruction_trueq + "{}" + instruction_tail

def replace_multiple_newlines(s):
    return re.sub(r'\n+', '\n', s)

def parse_string(s):
    pairs = []
    current_pair = ""
    bracket_level = 0

    for char in s:
        if char == ',' and bracket_level == 0:
            pairs.append(current_pair)
            current_pair = ""
        else:
            current_pair += char

            if char == '(':
                bracket_level += 1
            elif char == ')':
                bracket_level -= 1

    # Add the final pair
    if current_pair:
        pairs.append(current_pair)

    def find_middle_colon(stripped_pair):
        colons = [i for i, char in enumerate(stripped_pair) if char == ':']
        if colons:
            mid_index = len(colons) // 2
            return colons[mid_index]
        return -1

    result = []
    for pair in pairs:
        stripped_pair = pair.strip('()')
        colon_index = find_middle_colon(stripped_pair)
        
        if colon_index != -1:
            a = stripped_pair[:colon_index]
            b = stripped_pair[colon_index + 1:]
            result.append((a, b))

    return result

def replace_text(text, replacements):
    # 对替换列表按照第一个值（需要替换的字符串）的长度进行降序排序
    sorted_replacements = sorted(replacements, key=lambda x: len(x[0]), reverse=True)
    # replacements.sort(key=lambda x: len(x[0]), reverse=True)
    
    # 创建一个与原文本等长的数组，记录每个字是否已被替换
    replaced_flags = [False] * len(text)
    
    for old, new in sorted_replacements:
        if len(old)<2 or '问题' in old:
            continue
        index = 0
        while index < len(text):
            # 找到当前需要替换的字符串的位置
            pos = text.find(old, index)
            if pos == -1:
                break
            
            # 检查这个位置上的字符是否已经被替换过
            if not any(replaced_flags[pos:pos + len(old)]):
                # 替换字符串
                text = text[:pos] + new + text[pos + len(old):]
                temp = [True for _ in range(len(new))]
                replaced_flags = replaced_flags[:pos] + temp +replaced_flags[pos+len(old):]
                # # 更新替换标记
                # for i in range(pos, pos + len(new)):
                    # if i < len(replaced_flags):
                        # replaced_flags[i] = True
                index = pos + len(new)
            else:
                # 如果当前找到的位置已经被替换过，则继续向后查找
                index = pos + 1

    return text


def extract_text(input_string):
    pattern = r'给定文本：\s*(.*?)\s*\n\n需要替换的词：'
    match = re.search(pattern, input_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
def generate_select_words(raw_document, select):
    select = list(set(select))
    select = [obj for obj in select if obj!='']
    s = []
    sorted_select = sorted(select, key=lambda word: raw_document.find(word) if raw_document.find(word) != -1 else float('inf'))
    while sorted_select and raw_document.find(sorted_select[0]) != -1:
        raw_document = replace_first_occurrence(raw_document, sorted_select[0], '')
        s.append(sorted_select[0])
        sorted_select = sorted(select, key=lambda word: raw_document.find(word) if raw_document.find(word) != -1 else float('inf'))
    return s
def get_model_and_tokenizer(ckpt_dir):
    # load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    
    return model, tokenizer

def load_neighbor_embeddings():
    with open('../checkpoint/embedding_qwen/sorted_cl100_embeddings.json', 'r') as f1:
        sorted_cl100_emb = json.load(f1)
    with open('../checkpoint/embedding_qwen/sensitivity_of_embeddings.json', 'r') as f:
        sen_emb = np.array(json.load(f))
    with open('../checkpoint/embedding_qwen/token_to_vector_dict.json', 'r') as f:
        token_to_vector_dict = json.load(f)
        token_to_vector_dict = {token: np.array(vector) for token, vector in token_to_vector_dict.items()}
    return sorted_cl100_emb, sen_emb, token_to_vector_dict

def get_response(tokenizer, model, text, system):
    
    messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    # print('Generating...')

    generated_ids = model.generate(
        model_inputs.input_ids,
        # generation_config=getattr(model, 'generation_config', None),
        max_new_tokens=4096
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def main(args):
    df = pd.read_csv('./api/response_eng_gpt_gpt-4-turbo.csv',sep='\t')
    num = len(df)
    recover_model, recover_tokenizer = get_model_and_tokenizer(args.recover_ckpt_dir)
    print('Recover Model loaded.')

    trueq_list = []
    recover_list = []
    chn_pattern = '问题：'
    eng_pattern = 'Question:'
    pattern = eng_pattern
    with alive_bar(num) as bar:
        for i in range(0,num):
            trueq= df.iloc[i]['trueq']
            true_text = trueq.split(pattern)[0].strip()
            true_question = trueq.split(pattern)[1].strip()
            subq = df.iloc[i]['subq']
            sub_text = subq.split(pattern)[0].strip()
            sub_question = subq.split(pattern)[1].strip()
            subed = sub_text+'\n'+pattern+sub_question
            subr = df.iloc[i]['sub_r0']
            if type(subr)!=str:
                subr = ''

            recover_input = recover_template.format(subed, replace_multiple_newlines(subr), true_text, true_question)
            recover_output = get_response(recover_tokenizer, recover_model, recover_input, recover_system)
            
            trueq_list.append(trueq)
            recover_list.append(recover_output)
            bar()

    df_subpair = pd.DataFrame({'trueq':trueq_list,'recover':recover_list})
    df_subpair.to_csv(f'./tempfile/recover_result/recover_eng_gpt_gpt-4-turbo.csv',index=False,sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--recover_ckpt_dir', type=str, default='../exp/Recover/recover_ckpt_dir')
    args = parser.parse_args()
    main(args)
