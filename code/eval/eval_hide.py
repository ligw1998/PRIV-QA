import os
import sys
sys.path.append('./')

from torch import sub
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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


instruction_gpt = "你是一位负责保护给定文本的隐私信息的专家，请严格根据需要替换的词，将给定的文本（包括问题）中的这些词替换成对应的同类其他词（替换后词与原词含义不同），保证替换后的文本语义流畅通顺，且最大可能保护原词中含有的隐私信息，只需要给出替换前后的词对，用(a:b)代表一对替换前后的词，每两对之间用','隔开。"
sensitive_system = "以下是一段给定的新闻报道或知识型百科，你能否从该文本中找出敏感、隐私或需要进行模糊化处理的字、词、词组或短句，类型包括时间、日期、金钱、年龄、人名、数字、数量、证件信息、手机号码、重量、长度、国家、地点、职位、组织、公司、团体或国家的名称、专业领域的特定概念等，将该文本中所有满足上述条件的字、词、词组或短句写在一行中，并用,隔开，相同的词只写一次。\n给定的文本：\n"
user_input = "给定文本：%s \n\n需要替换的词：%s \n\n替换前后的词对："
# user_input = "给定的文本：%s \n\n 需要替换的词：%s \n\n 替换后的文本："
guide_in = ['问题：','回答：', ' ','根据以上文本：','问：']

importance_system = "Extract key keywords or phrases from the following text corresponding to the final question. Identify and list the most important keywords or key phrases in the text. These keywords should capture the main topics, concepts, or subjects discussed in the text and capture the essential topics with the question. Separate the keywords with comma.\\n\\nContext:"


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
        if len(old)<2:
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
    with open('./checkpoint/embedding_qwen/sorted_cl100_embeddings.json', 'r') as f1:
        sorted_cl100_emb = json.load(f1)
    with open('./checkpoint/embedding_qwen/sensitivity_of_embeddings.json', 'r') as f:
        sen_emb = np.array(json.load(f))
    with open('./checkpoint/embedding_qwen/token_to_vector_dict.json', 'r') as f:
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
        generation_config=getattr(model, 'generation_config', None),
        max_new_tokens=4096
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def get_ents(raw_input,lang,ltp,spacy_model):
    if lang=='ENG':
        target_ents = get_ents_en(raw_input, spacy_model)
    else:
        target_ents = get_ents_zh(raw_input,ltp, spacy_model)
    return target_ents

def testspacy(df):
    spacy_model_zh = spacy.load('zh_core_web_trf')
    ltp = LTP("LTP/small")
    print('Tool Model loaded')

    num = len(df)
    P = []
    R = []
    with alive_bar(num) as bar:
        for i in range(0,num):
            text= df.iloc[i]['text']
            question = df.iloc[i]['question']
            gt = df.iloc[i]['subword']
            curtext = text + " 问题：" + question
            entity_local = get_ents(curtext,'zh',ltp,spacy_model_zh)
            entity_list = list(entity_local)
            final_entity =  ','.join(entity_list)
            subwords=final_entity
            entity = handlestring(subwords)
            entity = process_subwords(entity)
            l = sort_keywords_by_position(curtext,entity).split(',')

            gt_entity = handlestring(gt)
            gt_entity = process_subwords(gt_entity)
            g = sort_keywords_by_position(curtext,gt_entity).split(',')

            TP = len(set(l) & set(g))

            # Calculate FP
            FP = len(set(l) - set(g))

            # Calculate FN
            FN = len(set(g) - set(l))

            # Calculate Precision
            Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            P.append(Precision)

            # Calculate Recall
            Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            R.append(Recall)

            bar.text(f'Precision: {Precision} | Recall: {Recall}')
            bar()

    avg_p = np.mean(P)
    avg_r = np.mean(R)
    print(f'AVG Precision: {avg_p} | AVG Recall: {avg_r}')



def main(args):
    df = pd.read_csv('./data/sensitive_compare_local.csv',sep='\t')
    num = len(df)
    hide_model, hide_tokenizer = get_model_and_tokenizer(args.hide_ckpt_dir)
    print('Hide Model loaded.')
    response_list = []
    text_list = []
    with alive_bar(num) as bar:
        for i in range(0,num):
            text= df.iloc[i]['text']
            # question = df.iloc[i]['question']
            subwords = df.iloc[i]['local']
            curtext = text
            entity = handlestring(subwords)
            entity = process_subwords(entity)
            l = sort_keywords_by_position(curtext,entity)

            hide_input = user_input%(curtext,l)
            hide_response = get_response(hide_tokenizer, hide_model, hide_input, instruction_gpt)  
            if (':' not in hide_response) or len(hide_response)<1:
                print(f'{i} th line hide model error with: {hide_response}')
                bar()
                continue
            text_list.append(curtext)
            response_list.append(hide_response)

            bar()

    df_subpair = pd.DataFrame({'text':text_list,'hidepair':response_list})
    df_subpair.to_csv('./tempfile/v6/hide_test_senslocalword.csv',index=False,sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx',type=int,default=0)
    parser.add_argument('--hide_ckpt_dir', type=str, default='../exp/Hide/hide_ckpt_dir')
    args = parser.parse_args()
    main(args)
