import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime as dt
import spacy
import re
from collections import defaultdict
import os
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Extract entities
def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    money_ent = defaultdict(int)
    percent_ent = defaultdict(int)
    number_ent = defaultdict(int)
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            money_ent[ent.text] += 1
        elif ent.label_ == "PERCENT":
            percent_ent[ent.text] += 1
        elif ent.label_ == "CARDINAL":
            number_ent[ent.text] += 1
    return money_ent, percent_ent, number_ent

def check_questions(sentence):
    if re.search(r'\?$', sentence.strip()):
        return 1
    if re.search(r'^(who|what|where|when|why|how)\s', sentence.strip().lower()):
        return 1
    return 0

def get_n2w_ratio(n_words, text):
    money_ent, percent_ent, number_ent = extract_entities(text)
    number_ent_total = len(money_ent) + len(percent_ent) + len(number_ent)
    return number_ent_total / n_words

def get_number_words(text):
    return len(text.split())

def get_number_sents(text):
    n_sent = 0
    n_quest = 0
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for sent in doc.sents:
        n_sent += 1
        n_quest += check_questions(sent.text)
    return n_sent, n_quest

def get_attribute(text_block):
    n_words = get_number_words(text_block)
    n_sents, n_questions = get_number_sents(text_block)
    num_2_word_ratio = get_n2w_ratio(n_words, text_block)
    return n_words, n_sents, num_2_word_ratio, n_questions

def extract_attr_sent(data_sent):
    for row in data_sent.itertuples():
        if row.speaker_type == 'Corporate Participant' and row.type == 'presentation':
            n_words, n_sents, num_2_word_ratio, _ = get_attribute(row.text)
            data_sent.at[row.Index, 'num_words_present'] = n_words
            data_sent.at[row.Index, 'num_sents_present'] = n_sents
            data_sent.at[row.Index, 'num_2_word_ratio_present'] = num_2_word_ratio
        elif row.speaker_type == 'Corporate Participant' and row.type == 'qna':
            n_words, n_sents, num_2_word_ratio, _ = get_attribute(row.text)
            data_sent.at[row.Index, 'num_words_qna'] = n_words
            data_sent.at[row.Index, 'num_sents_qna']  = n_sents
            data_sent.at[row.Index, 'num_2_word_ratio_qna'] = num_2_word_ratio
        elif row.speaker_type == 'Conference Participant':
            _, _, _, num_quests = get_attribute(row.text)
            data_sent.at[row.Index, 'num_questions_asked'] = num_quests
    return data_sent

def make_att_df(sent_level_df):
    df = extract_attr_sent(sent_level_df)
    result = (
       df
       .groupby(['company_name', 'date'])
       .agg(
          total_num_words_present = ('num_words_present', 'sum'),
          total_num_sents_present = ('num_sents_present', 'sum'),
          avg_num_2_word_ratio_present = ('num_2_word_ratio_present', 'mean'),
          total_num_questions = ('num_questions_asked', 'sum'),
          total_num_words_qna = ('num_words_qna', 'sum'),
          total_num_sents_qna = ('num_sents_qna', 'sum'),
          avg_num_2_word_ratio_qna = ('num_2_word_ratio_qna', 'mean')
        )
        .reset_index()
    )
    return result

# Compute similarities
load_dotenv()
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", token = ACCESS_TOKEN)
model_summary = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", token = ACCESS_TOKEN)
model_score = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate(context: str):
    model_summary.to(device)
    prompt = f"""
        I would like you to read the following text and summarize it into a concise abstract paragraph.
        Aim to retain the most important points, providing a coherent and readable summary that could
        help a person understand the main points of the discussion without needing to read the entire text.
        Please avoid unnecessary details or tangential points.
        Context: {context}.
    """
    
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize = False,
        add_generation_prompt = True,
    )
    
    inputs = tokenizer.encode(formatted_prompt, add_special_tokens = False, return_tensors = "pt").to(device)
    with torch.no_grad():
        outputs = model_summary.generate(input_ids = inputs, max_new_tokens = 250, do_sample = False,)
    response = tokenizer.decode(outputs[0], skip_special_tokens = False)
    response = response[len(formatted_prompt) :]
    response = response.replace("<eos>", "")
    return response

def compute_similarity(sent1, sent2):
    embedding_1= model_score.encode(sent1, convert_to_tensor = True)
    embedding_2 = model_score.encode(sent2, convert_to_tensor = True)
    cos_sim = util.pytorch_cos_sim(embedding_1, embedding_2)
    return cos_sim.item()

def extract_similarity_score(data):
    for row in data.itertuples():
       if not pd.isnull(row.speaker_role):
          if (row.speaker_type == 'Corporate Participant') and (row.type == 'presentation') and not re.search(r'\bIR\b', row.speaker_role):
             text_block = row.text
             summary = generate(context=text_block)
             data.at[row.Index, 'summary'] = summary
    
    result2 = (
        data.dropna(subset = ['summary']) 
        .groupby(['company_name', 'date'])
        .agg({'summary': ' '.join})
        .reset_index()
        .sort_values(by = ['company_name', 'date'])
    )

    for company in result2['company_name'].unique():
        company_df = result2[result2['company_name'] == company]
        for i in range(1, len(company_df)):
            similarity_scores = []
            for j in range(max(0, i - 3), i):
                similarity = compute_similarity(company_df['summary'].iloc[i], company_df['summary'].iloc[j])
                similarity_scores.append(similarity)
            for k in range(3 - len(similarity_scores)):
                similarity_scores.append(None)
            result2.loc[company_df.index[i], 'similarity_to_1_previous'] = similarity_scores[0]
            result2.loc[company_df.index[i], 'similarity_to_2_previous'] = similarity_scores[1]
            result2.loc[company_df.index[i], 'similarity_to_3_previous'] = similarity_scores[2]
    return result2
