import os
import sys
sys.path.append('./')

from torch import sub
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
sensitive_system = "你是一个隐私保护专家，给定一段文本，需要你从该文本中找出所有的和个人隐私有关的信息，隐私信息的类别主要有下面几类：\n\n1. 个人或者公司名字\n2. 日期和时间\n3. 地点（具体地点如国家，省份，市，县，区，镇，村，街道，住所等，必须是明确的地名，不得包括模糊或通用性地点词汇）\n4. 个人信息相关（电话，证件信息，电子邮件，职位等）\n5. 敏感数字（金钱，长度，百分比等）\n\n输出要求如下：\n1. 每一类为一行进行输出，先输出类别的名称，后输出具体的包含隐私信息的词，如果不包含某类的信息，则该类输出“无”。\n2. 每一类按照词在文本中出现的先后顺序从前到后进行输出，相同的词只输出一次。\n3. 地点类词汇只需标注具体的地名，不需要包含详细职位或单位名称。例如对于“湖北省省委组织部”，只需要输出“湖北省”即可。\n4. 输出数字的时候，如果数字后有量词，需要和量词一起输出。例如“4次”，“6年”。文本："
user_input = "给定文本：%s \n\n需要替换的词：%s \n\n替换前后的词对："
guide_in = ['问题：','回答：', ' ','根据以上文本：','问：']

importance_system = "Extract key keywords or phrases from the following text corresponding to the final question. Identify and list the most important keywords or key phrases in the text. These keywords should capture the main topics, concepts, or subjects discussed in the text and capture the essential topics with the question. Separate the keywords with comma.\\n\\nContext:"


def split_text_by_punctuation(text, max_chars=300):  # max_chars可根据需求调整
    # 使用分隔符切割文本
    segments = []
    current_segment = ""

    for char in text:
        current_segment += char

        # 当长度超过最大限制时，检查最后一个字符
        if len(current_segment) >= max_chars:
            # 找到最近的分隔符
            if '。' in current_segment or '，' in current_segment:
                # 寻找最后一个分隔符
                last_punct_index = max(current_segment.rfind('。'), current_segment.rfind('，'))
                if last_punct_index != -1:
                    # 将分段切割并保存
                    segments.append(current_segment[:last_punct_index + 1])
                    # 更新当前段落为剩余部分
                    current_segment = current_segment[last_punct_index + 1:].lstrip()  # 去掉开头的空格
            else:
                # 如果没有找到分隔符，强制切割
                segments.append(current_segment)
                current_segment = ""

    # 如果还有未分割的部分，加入到段落中
    if current_segment:
        segments.append(current_segment)

    return segments

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
    if lang=='eng':
        target_ents = get_ents_en(raw_input, spacy_model)
    else:
        target_ents = get_ents_zh(raw_input,ltp, spacy_model)
    return target_ents

def testspacy(df,args):
    spacy_model_zh = spacy.load('zh_core_web_trf')
    spacy_model_en = spacy.load('en_core_web_trf')
    ltp = LTP("LTP/small")
    print('Tool Model loaded')

    num = len(df)
    a = []
    b = []
    c = []
    P = []
    R = []
    with alive_bar(num) as bar:
        for i in range(0,num):
            if args.lang=='chn':
                # if i%3!=0:
                    # bar()
                    # continue
                text= df.iloc[i]['text']
                question = df.iloc[i]['question']
                gt = df.iloc[i]['subword']
                if type(gt)!=str:
                    gt=''
                curtext = text + " 问题：" + question
                # curtext = text
                entity_local = get_ents(curtext,args.lang,ltp,spacy_model_zh)
                entity_list = list(set(list(entity_local)))
                # final_entity =  ','.join(entity_list)
                l = sort_keywords_by_position(curtext,entity_list)
                # l = process_subwords(final_entity)
                # l = sort_keywords_by_position(curtext,l)
                # l = ','.join(l)
                a.append(curtext)
                c.append(l)
                l = l.split(',')

                gt_entity = gt
                gt_entity = gt_entity.split(',')
                # gt_entity = process_subwords(gt_entity)
                g = sort_keywords_by_position(curtext,gt_entity)
                # g = ','.join(g)
                b.append(g)
                g = g.split(',')

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
            else:
                text= df.iloc[i]['text']
                gt = df.iloc[i]['subword']
                question = df.iloc[i]['question']
                if type(gt)!=str:
                    gt=''
                curtext = text + " Question: " + question
                # curtext = text
                entity_local = get_ents(curtext,args.lang,ltp,spacy_model_en)
                entity_list = list(entity_local)
                _e = []
                # for obj in entity_list:
                    # for o in obj.split(' '):
                        # if len(o)>0:
                            # _e.append(o)
                for obj in entity_list:
                    _e.append(obj)
                _e = list(set(_e))
                entity_list = _e
                final_entity =  ','.join(entity_list)
                l=final_entity
                a.append(curtext)
                c.append(l)
                l = l.split(',')

                sensitive_gt = []
                # chunk_gt = handle_sensitive_response(gt,args)
                chunk_gt = gt.split(',')
                # for obj in chunk_gt:
                    # for obj_ in obj.split(' '):
                        # if len(obj_)>0:
                            # sensitive_gt.append(obj_)
                # sensitive_gt = list(set(sensitive_gt))
                for obj in chunk_gt:
                    sensitive_gt.append(obj)
                sensitive_gt = list(set(sensitive_gt))
                # sensitive_gt = process_subwords(sensitive_gt)
                g = ','.join(sensitive_gt)
                b.append(g)
                g = g.split(',')

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
    df_new = pd.DataFrame({'text':a,'gpt4o':b,'ner':c})



def handle_sensitive_response(text,args):
    if args.lang=='chn':
        l=[]
        text= text.strip()
        response = text.split('\n')
        temp = response[0].split('个人或者公司名字：')[-1]
        if not '无' in temp:
            for obj in temp.split('，'):
                l.append(obj.strip())
        temp = response[1].split('日期和时间：')[-1]
        if not '无' in temp:
            for obj in temp.split('，'):
                l.append(obj.strip())
        temp = response[2].split('地点：')[-1]
        if not '无' in temp:
            for obj in temp.split('，'):
                l.append(obj.strip())
        temp = response[3].split('个人信息相关：')[-1]
        if not '无' in temp:
            for obj in temp.split('，'):
                l.append(obj.strip())
        temp = response[4].split('敏感数字：')[-1]
        if not '无' in temp:
            for obj in temp.split('，'):
                l.append(obj.strip())
        return l
    else:
        l=[]
        text= text.strip()
        response = text.split('\n')
        if len(response)!=5:
            return l
        temp = response[0].split('个人或者公司名字')[-1]
        if not '无' in temp:
            for obj in temp.split(', '):
                l.append(obj.strip(' :：'))
        temp = response[1].split('日期和时间')[-1]
        if not '无' in temp:
            for obj in temp.split(', '):
                l.append(obj.strip(' :：'))
        temp = response[2].split('地点')[-1]
        if not '无' in temp:
            for obj in temp.split(', '):
                l.append(obj.strip(' :：'))
        temp = response[3].split('个人信息相关')[-1]
        if not '无' in temp:
            for obj in temp.split(', '):
                l.append(obj.strip(' :：'))
        temp = response[4].split('敏感数字')[-1]
        if not '无' in temp:
            for obj in temp.split(', '):
                l.append(obj.strip(' :：'))
        return l



def main(args):
    df = pd.read_csv('./test/chn_test.csv',sep='\t')
    num = len(df)
    if args.spacy:
        testspacy(df,args)
        return
    # hide_model, hide_tokenizer = get_model_and_tokenizer(args.hide_ckpt_dir)
    # print('Hide Model loaded.')
    sensitive_model, sensitive_tokenizer = get_model_and_tokenizer(args.sensitive_ckpt_dir)
    print('Sensitive Model loaded.')
    a = []
    b = []
    c = []
    P = []
    R = []
    with alive_bar(num) as bar:
        for i in range(0,num):
            if args.lang=='chn':
                # if i%3!=0:
                    # bar()
                    # continue
                text= df.iloc[i]['text']
                question = df.iloc[i]['question']
                gt = df.iloc[i]['subword']
                curtext = text + " 问题：" + question
                text_chunks = split_text_by_punctuation(curtext,300)
                sensitive = []
                for _,chunk in  enumerate(text_chunks):
                    _subwords = get_response(sensitive_tokenizer, sensitive_model, chunk, sensitive_system)
                    chunk_l = handle_sensitive_response(_subwords,args)
                    for obj in chunk_l:
                        sensitive.append(obj)
                entity = process_subwords(sensitive)
                l = sort_keywords_by_position(curtext,entity)
                a.append(curtext)
                c.append(l)
                l = l.split(',')

                gt_entity = handlestring(gt)
                gt_entity = process_subwords(gt_entity)
                g = sort_keywords_by_position(curtext,gt_entity)
                b.append(g)
                g = g.split(',')

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

                # if Recall<0.7 and i%3==0:
                    # print(f'{i}th line {",".join(l)}\n {",".join(g)}')
                bar.text(f'Precision: {Precision} | Recall: {Recall}')
                bar()
            if args.lang=='eng':
                text= df.iloc[i]['text']
                question = df.iloc[i]['question']
                curtext = text + " Question: " + question
                gt = df.iloc[i]['subword']
                if type(gt)!=str:
                    continue
                sensitive = []
                _subwords = get_response(sensitive_tokenizer, sensitive_model, text, sensitive_system)
                chunk_l = handle_sensitive_response(_subwords,args)
                if len(chunk_l)<2:
                    print(f'{i}th line error')
                    continue
                for obj in chunk_l:
                    # for obj_ in obj.split(' '):
                        # if len(obj_)>0:
                            # sensitive.append(obj_)
                    sensitive.append(obj)
                # sensitive = process_subwords(sensitive)
                sensitive = list(set(sensitive))
                l = ','.join(sensitive)
                a.append(curtext)
                c.append(l)
                l = l.split(',')

                sensitive_gt = []
                # chunk_gt = handle_sensitive_response(gt,args)
                chunk_gt = gt.split(',')
                for obj in chunk_gt:
                    # for obj_ in obj.split(' '):
                        # if len(obj_)>0:
                            # sensitive_gt.append(obj_)
                    sensitive_gt.append(obj)
                sensitive_gt = list(set(sensitive_gt))
                # sensitive_gt = process_subwords(sensitive_gt)
                g = ','.join(sensitive_gt)
                b.append(g)
                g = g.split(',')

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

                # if Recall<0.7 and i%3==0:
                    # print(f'{i}th line {",".join(l)}\n {",".join(g)}')
                bar.text(f'Precision: {Precision} | Recall: {Recall}')
                bar()

    avg_p = np.mean(P)
    avg_r = np.mean(R)
    print(f'AVG Precision: {avg_p} | AVG Recall: {avg_r}')
    df_new = pd.DataFrame({'text':a,'gpt4o':b,'local':c})
    df_new.to_csv(f'./tempfile/sensitive_compare_local_{args.lang}.csv',sep='\t',index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx',type=int,default=0)
    parser.add_argument('--lang',type=str,default='chn')
    parser.add_argument('--sensitive_ckpt_dir', type=str, default='../exp/Sensitive_chunk_v3')
    parser.add_argument('--spacy',action='store_true')
    args = parser.parse_args()
    main(args)
