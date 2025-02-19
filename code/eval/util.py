import pandas as pd
import numpy as np
from alive_progress import alive_bar
import re
import random

def is_all_english(s):
    pattern = r'^[A-Za-z0-9\s!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]*$'
    return re.match(pattern, s) is not None

def clean_and_filter(strings):
    return [s.strip().replace('\n', '') for s in strings if (len(s.replace('\n', '')) <= 15 or (is_all_english(s) and len(s)<=25) )]

def process_subwords(a):
    clean_a = clean_and_filter(a)
    union_set = set(clean_a)
    union_list = sorted(union_set, key=len)
    temp_list = []
    remaining_parts = []
    for s in union_list:
        if len(s)<=1:
            continue
        temp_list.append(s)
        for other in temp_list:
            if s != other:
                if other in s:
                    if (s.startswith(other) or s.endswith(other)):
                        if s.startswith(other):
                            remaining_part = s[len(other):]
                        else:
                            remaining_part = s[:-len(other)]
                        if len(remaining_part)>1:
                            remaining_parts.append(remaining_part)
                        break
    
    temp_list.extend(remaining_parts)
    temp_list_ = sorted(set(temp_list), key=len)
    
    def contains_digit(s):
        return any(char.isdigit() for char in s)

    final_list = []
    for i, s in enumerate(temp_list_):
        if contains_digit(s):
            final_list.append(s)
        elif len(s)==1:
            continue
        else:
            if not any(s != other and other in s for other in final_list):
                final_list.append(s)

    return final_list

def process_strings(a, b):
    clean_a = clean_and_filter(a)
    clean_b = clean_and_filter(b)
    
    union_set = set(clean_a + clean_b)
    
    union_list = sorted(union_set, key=len)
    
    def contains_digit(s):
        return any(char.isdigit() for char in s)

    final_list = []
    for i, s in enumerate(union_list):
        if contains_digit(s):
            final_list.append(s)
        elif len(s)==1:
            continue
        else:
            if not any(s != other and other in s for other in final_list):
                final_list.append(s)

    return final_list

def handlestring(cur_entity):
    if pd.isnull(cur_entity):
        cur_entity_ = []
    elif cur_entity:
        cur_entity_ = cur_entity.split(',')
    else:
        cur_entity_ = []
    return cur_entity_

def sort_keywords_by_position(text, key):
    # key = key.split(',')
    keyword_positions = []
    for keyword in key:
        pos = text.find(keyword)
        if pos != -1:  
            keyword_positions.append((keyword, pos))
            
    sorted_keywords = sorted(keyword_positions, key=lambda x: (x[1], len(x[0])))
    
    sorted_key = [kw[0] for kw in sorted_keywords]
    sorted_key = ','.join(sorted_key)
    
    return sorted_key

