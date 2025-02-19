import os
import jieba
import nltk
from nltk.lm import smoothing
import pandas as pd
import numpy as np
# from rouge import Rouge
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate import meteor_score
from alive_progress import alive_bar
from transformers import AutoModelForCausalLM, AutoTokenizer


def calculate_meteor(response,gt_list):
    # seg_response = nltk.word_tokenize(response)  # 使用nltk进行分词
    # seg_gts = [nltk.word_tokenize(gt) for gt in gt_list]
    seg_response = list(jieba.cut(response))  # 使用 jieba 进行分词
    seg_gts = [list(jieba.cut(gt)) for gt in gt_list]  # 分词所有参考答案
    meteor = meteor_score.meteor_score(seg_gts, seg_response)
    return meteor

def calculate_bleu(response,gt_list):
    seg_response = list(jieba.cut(response))  # 使用 jieba 进行分词
    seg_gts = [list(jieba.cut(gt)) for gt in gt_list]  # 分词所有参考答案
    # seg_response = nltk.word_tokenize(response)  # 使用nltk进行分词
    # seg_gts = [nltk.word_tokenize(gt) for gt in gt_list]
    smoothing_function = SmoothingFunction().method4
    bleu = sentence_bleu(seg_gts, seg_response, smoothing_function=smoothing_function)
    return bleu

def calculate_rouge(response,gt_list):
    scores_list = []
    rouge = Rouge()
    for gt in gt_list:
        response_p = ' '.join(list(jieba.cut(response)))  # 可能需要用空格连接
        gt_p = ' '.join(list(jieba.cut(gt)))  # 可能需要用空格连接
        scores = rouge.get_scores(response_p, gt_p)
        scores_list.append((scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']))
    # for gt in gt_list:
        # response_p = ' '.join(nltk.word_tokenize(response))  # 使用nltk分词
        # gt_p = ' '.join(nltk.word_tokenize(gt))
        # scores = rouge.get_scores(response_p, gt_p)
        # scores_list.append((scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']))
    
    # 提取每种 ROUGE 分数的均值
    rouge_1 = np.mean([score[0] for score in scores_list])
    rouge_2 = np.mean([score[1] for score in scores_list])
    rouge_l = np.mean([score[2] for score in scores_list])
    
    return rouge_1, rouge_2, rouge_l


def main():
    df_gt = pd.read_csv('./temp_file/recover_test/chn_qa.csv',sep='\t')
    num = len(df_gt)
    bleu_full=[]
    meteor_full=[]
    rouge_1_full=[]
    rouge_2_full=[]
    rouge_l_full = []
    total= 0
    with alive_bar(num) as bar:
        for i in range(0,num):
            sub = df_gt.iloc[i]['subq']
            gt = df_gt.iloc[i]['trueq']
            if type(sub)!=str or type(gt)!=str:
                continue
            sub = sub.strip()
            gt = gt.strip()
            if len(sub)<1 or len(gt)<1:
                continue
            bleu = calculate_bleu(sub,[gt])
            meteor = calculate_meteor(sub,[gt])
            rouge_1,rouge_2,rouge_l = calculate_rouge(sub,[gt])
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
