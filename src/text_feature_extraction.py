import pandas as pd
import textstat
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from bertopic.representation import KeyBERTInspired
import spacy



# Calculate readability metrics
def calculate_readability_metrics(df, text_var, keep_all = False):

    # Define readability functions
    readability_functions = {
        'automated_readability_index': textstat.automated_readability_index,
        'coleman_liau_index': textstat.coleman_liau_index,
        'dale_chall_readability_score': textstat.dale_chall_readability_score,
        'flesch_reading_ease': textstat.flesch_reading_ease,
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade,
        'gunning_fog': textstat.gunning_fog,
        'smog_index': textstat.smog_index,
    }

    # Evaluate readability metrics
    for metric, function in readability_functions.items():
        df[f'r_{metric}_{text_var}'] = df[text_var].apply(function)

    # Calculate overall readability
    metric_columns = [f'r_{metric}_{text_var}' for metric in readability_functions]
    df[f'readability_overall_{text_var}'] = df[metric_columns].mean(axis = 1)

    # Drop columns (if required)
    if not keep_all:
        df.drop(metric_columns, axis=1, inplace=True)
        if not keep_all:
            df.drop([text_var], axis=1, inplace=True)

    # end
    return df

# Create readability features
def create_readability_features(df):

    # Aggregate data by speaker
    grouped = df.groupby(['company_name', 'date', 'speaker_type', 'type'])['text'].apply(' '.join).reset_index()

    # Pivot data to wide format
    pivot_table = grouped.pivot(index = ['company_name', 'date'], columns = ['speaker_type', 'type'], values = 'text').reset_index()
    pivot_table = pivot_table.fillna('')
    pivot_table.columns = ['company_name', 'date', 'conf_qna', 'corp_pres', 'corp_qna', 'op_pres', 'op_qna']

    # Calculate readability by speaker type
    for var in ['conf_qna', 'corp_pres', 'corp_qna', 'op_pres', 'op_qna']:
        pivot_table = calculate_readability_metrics(pivot_table, text_var = var)

    # end
    return pivot_table



# Load the spaCy model
nlp = spacy.load('en_core_web_sm')
nltk.download('punkt')

# Redact text for topic models
def redact_text_with_spacy(text):

    # Process text
    doc = nlp(text)
    redacted_text = text
    sorted_entities = sorted(doc.ents, key = lambda ent: ent.start_char, reverse = True)

    # Replace entities with their label
    for ent in sorted_entities:
        if ent.label_ not in ['ORDINAL', 'CARDINAL']:
            redacted_text = redacted_text[:ent.start_char] + '[' + ent.label_ + ']' + redacted_text[ent.end_char:]

    return redacted_text, text



def create_sentence_topics(df):

    # Redact text and store original text
    df['redacted_text'], df['original_text'] = zip(*df['text'].apply(redact_text_with_spacy))

    # Subset data for corporate participant text
    subset_df = df[df['speaker_type'] == "Corporate Participant"]

    # Tokenize text into sentences for both redacted and original texts
    subset_df['sentences'] = subset_df['redacted_text'].apply(sent_tokenize)
    subset_df['original_sentences'] = subset_df['original_text'].apply(sent_tokenize)

    # Flatten list of lists into a single list of sentences along with their original texts and IDs
    all_sentences = [(sent, orig, full_text, ID) for full_text, sents, origs, ID in zip(subset_df['original_text'], subset_df['sentences'], subset_df['original_sentences'], subset_df['ID']) for sent, orig in zip(sents, origs)]

    # Remove duplicates
    unique_sentences = list(set(all_sentences))

    # Split the tuples for processing
    redacted_only = [sent[0] for sent in unique_sentences]
    original_only = [sent[1] for sent in unique_sentences]
    full_text_only = [sent[2] for sent in unique_sentences]
    ids_only = [sent[3] for sent in unique_sentences]

    # Define and fit the topic model
    topic_model = BERTopic(representation_model = "KeyBERTInspired", min_topic_size = 100, nr_topics = 'auto')
    topics, probabilities = topic_model.fit_transform(redacted_only)

    # Create DataFrame with topic mappings
    sentence_df = pd.DataFrame({
        'Sentence': redacted_only,
        'Topic': topics,
        'Original_Sentence': original_only,
        'Original_Text': full_text_only,
        'ID': ids_only
    })

    # Include topic names
    topic_info = topic_model.get_topic_info()
    topic_info.rename(columns = {'Name': 'Topic_Name'}, inplace = True)
    topic_info['Topic'] = topic_info.index - 1
    sentence_df = pd.merge(sentence_df, topic_info[['Topic', 'Topic_Name']], on = 'Topic', how = 'left')

    # Finalze output
    selected_df = sentence_df.loc[:, ['ID', 'Sentence', 'Original_Sentence', 'Topic', 'Topic_Name']]
    sorted_df = selected_df.sort_values(by = 'ID')

    return sorted_df