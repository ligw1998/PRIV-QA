import os
import jieba
import nltk
from nltk.lm import smoothing
import pandas as pd
import numpy as np
from rouge import Rouge
# from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate import meteor_score
from alive_progress import alive_bar
from transformers import AutoModelForCausalLM, AutoTokenizer


def calculate_meteor(response,gt_list):
    seg_response = nltk.word_tokenize(response)  # 使用nltk进行分词
    seg_gts = [nltk.word_tokenize(gt) for gt in gt_list]
    # seg_response = list(jieba.cut(response))  # 使用 jieba 进行分词
    # seg_gts = [list(jieba.cut(gt)) for gt in gt_list]  # 分词所有参考答案
    meteor = meteor_score.meteor_score(seg_gts, seg_response)
    return meteor

def calculate_bleu(response,gt_list):
    # seg_response = list(jieba.cut(response))  # 使用 jieba 进行分词
    # seg_gts = [list(jieba.cut(gt)) for gt in gt_list]  # 分词所有参考答案
    seg_response = nltk.word_tokenize(response)  # 使用nltk进行分词
    seg_gts = [nltk.word_tokenize(gt) for gt in gt_list]
    smoothing_function = SmoothingFunction().method4
    bleu = sentence_bleu(seg_gts, seg_response, smoothing_function=smoothing_function)
    return bleu

def calculate_rouge(response,gt_list):
    scores_list = []
    rouge = Rouge()
    # for gt in gt_list:
        # response_p = ' '.join(list(jieba.cut(response)))  # 可能需要用空格连接
        # gt_p = ' '.join(list(jieba.cut(gt)))  # 可能需要用空格连接
        # scores = rouge.get_scores(response_p, gt_p)
        # scores_list.append((scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']))
    for gt in gt_list:
        response_p = ' '.join(nltk.word_tokenize(response))  # 使用nltk分词
        gt_p = ' '.join(nltk.word_tokenize(gt))
        scores = rouge.get_scores(response_p, gt_p)
        scores_list.append((scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']))
    
    # 提取每种 ROUGE 分数的均值
    rouge_1 = np.mean([score[0] for score in scores_list])
    rouge_2 = np.mean([score[1] for score in scores_list])
    rouge_l = np.mean([score[2] for score in scores_list])
    
    return rouge_1, rouge_2, rouge_l


def main():
    df_gt = pd.read_csv('./api/response_eng_gpt_gpt-4-turbo.csv',sep='\t')
    df_recover = pd.read_csv('./temp_file/recover_result/recover_eng_gpt_gpt-4-turbo.csv',sep='\t')
    num = len(df_gt)
    bleu_full=[]
    meteor_full=[]
    rouge_1_full=[]
    rouge_2_full=[]
    rouge_l_full = []
    j=0
    total= 0
    chn_pattern = '问题：'
    eng_pattern = 'Question:'
    pattern = eng_pattern
    with alive_bar(num) as bar:
        for i in range(0,num):
            tq = df_gt.iloc[i]['trueq'].split(pattern)
            tq = tq[0].strip()+' '+pattern+tq[1].strip()
            gt = []
            for r in [df_gt.iloc[i]['true_r0'], df_gt.iloc[i]['true_r1'], df_gt.iloc[i]['true_r2']]:
                if pd.notna(r):  # 检查是否不是 NaN
                    gt.append(r)
            if len(gt)==0:
                continue
            tqr = df_recover.iloc[j]['trueq'].split(pattern)
            tqr  = tqr[0].strip()+' '+pattern+tqr[1].strip()
            # tqr = df_recover.iloc[j]['ori'].strip()+ ' '+pattern+df_recover.iloc[j]['sensitiveq'].strip()
            while (tq!=tqr):
                j+=1
                tqr = df_recover.iloc[j]['trueq'].split(pattern)
                tqr  = tqr[0].strip()+' '+pattern+tqr[1].strip()
                # tqr = df_recover.iloc[j]['trueq'].strip()
            recover = df_recover.iloc[j]['recover'].strip()
            # recover = df_recover.iloc[j]['sub_answer']
            if type(recover)!=str or len(recover)<1:
                continue
            recover = recover.strip()
            bleu = calculate_bleu(recover,gt)
            meteor = calculate_meteor(recover,gt)
            rouge_1,rouge_2,rouge_l = calculate_rouge(recover,gt)
            bleu_full.append(bleu)
            meteor_full.append(meteor)
            rouge_1_full.append(rouge_1)
            rouge_2_full.append(rouge_2)
            rouge_l_full.append(rouge_l)
            total+=1
            bar()
    print(f'AVG bleu: {np.mean(bleu_full)} \n AVG meteor: {np.mean(meteor_full)} \n AVG ROUGE -1: {np.mean(rouge_1_full)} | -2: {np.mean(rouge_2_full)} | -l: {np.mean(rouge_l_full)} ')
    print(f'Total num: {total}')
            


if __name__ == "__main__":
    main()
