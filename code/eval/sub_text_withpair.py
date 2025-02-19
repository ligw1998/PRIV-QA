import os
import sys
sys.path.append('./')

from torch import sub
import re
import json
import jsonlines
import pandas as pd
import random
import argparse
import numpy as np
from func import *
from util import process_subwords, handlestring,sort_keywords_by_position
from alive_progress import alive_bar
from ltp import LTP
import spacy
from spacyutils import get_ents_en, get_ents_zh, add_pinyin, get_labelled_text
import ast


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
    sorted_replacements = sorted(replacements, key=lambda x: len(x[0]), reverse=True)
    # replacements.sort(key=lambda x: len(x[0]), reverse=True)
    
    replaced_flags = [False] * len(text)
    
    for old, new in sorted_replacements:
        if len(old)<2 or '问题' in old:
            continue
        index = 0
        while index < len(text):
            pos = text.find(old, index)
            if pos == -1:
                break
            
            if not any(replaced_flags[pos:pos + len(old)]):
                text = text[:pos] + new + text[pos + len(old):]
                temp = [True for _ in range(len(new))]
                replaced_flags = replaced_flags[:pos] + temp +replaced_flags[pos+len(old):]
                # for i in range(pos, pos + len(new)):
                    # if i < len(replaced_flags):
                        # replaced_flags[i] = True
                index = pos + len(new)
            else:
                index = pos + 1

    return text

def main():
    df_chn = pd.read_csv('./temp_file/hide_test_chn.csv',sep='\t')
    df_eng = pd.read_csv('./temp_file/hide_test_eng.csv',sep='\t')
    df = df_eng
    a = []
    b = []
    c = []
    with alive_bar(len(df)) as bar:
        for i in range(0,len(df)):
            text = df.iloc[i]['text']
            subpair = df.iloc[i]['hidepair']
            sublist = parse_string(subpair)
            subed = replace_text(text, sublist)
            a.append(text)
            b.append(subpair)
            c.append(subed)
            bar()

    df_new = pd.DataFrame({'text':a,'hidepair':b,'subed':c})
    df_new.to_csv('./temp_file/sub_test_eng.csv',sep='\t',index=False)



if __name__ == "__main__":
    main()
