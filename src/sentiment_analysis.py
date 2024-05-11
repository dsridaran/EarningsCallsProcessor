import pandas as pd
import numpy as np
import transformers
from tqdm import tqdm


model_name = 'ProsusAI/finbert'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)


def get_sentiment_sentence_bert(text, already_sentenced=False):
    if not already_sentenced:
        sentences = text.split('.')
        list_sentiment = []
        for sentence in sentences:
            tokenized = tokenizer(sentence, return_tensors='pt')
            output = model(**tokenized)
            scores = output.logits.softmax(dim=1).detach().numpy()
            list_sentiment.append(scores[0])
        return np.mean(list_sentiment, axis=0)
    else:
        tokenized = tokenizer(text, return_tensors='pt')
        output = model(**tokenized)
        scores = output.logits.softmax(dim=1).detach().numpy()
        return scores[0]


def get_sentiment_bert(data, col='text'):
    texts = data[col]
    positives = []
    negatives = []
    neutrals = []
    polarities = []
    for i in tqdm(range(len(data))):
        text = texts[i]
        positive, negative, neutral = get_sentiment_sentence_bert(text)
        positives.append(positive)
        negatives.append(negative)
        neutrals.append(neutral)
        polarity = (positive - negative) / (positive + negative + neutral)
        polarities.append(polarity)

    
    # new column for sentiment
    data['positive_sentiment_bert'] = positives
    data['negative_sentiment_bert'] = negatives
    data['neutral_sentiment_bert'] = neutrals
    data['polarity_bert'] = polarities
    return data

words = ['margin', 'cost', 'revenue', 'earnings', 'growth', 'debt', 'dividend', 'cashflow']

def get_sentiment_topic(data):
    texts = data['text'].apply(lambda x: x.lower())
    positives = {}
    negatives = {}
    neutrals = {}
    polarities = {}
    for word in words:
        positives[word] = []
        negatives[word] = []
        neutrals[word] = []
        polarities[word] = []
    for i in tqdm(range(len(data))):
        text = texts[i]
        for word in words:

            if word in text:
                positive, negative, neutral = get_sentiment_sentence_bert(text)
                polarity = (positive - negative) / (positive + negative + neutral)
            else:
                positive, negative, neutral = -1, -1, -1
                polarity = -1
            positives[word].append(positive)
            negatives[word].append(negative)
            neutrals[word].append(neutral)
            polarities[word].append(polarity)
    for word in words:
        data[f'positive_sentiment_bert_{word}'] = positives[word]
        data[f'negative_sentiment_bert_{word}'] = negatives[word]
        data[f'neutral_sentiment_bert_{word}'] = neutrals[word]
        data[f'polarity_bert_{word}'] = polarities[word]
    return data


global_sentiment_cols = ['positive_sentiment_bert', 'negative_sentiment_bert', 'neutral_sentiment_bert', 'polarity_bert']

topic_sentiment_cols = [f'positive_sentiment_bert_{word}' for word in words] + [f'negative_sentiment_bert_{word}' for word in words] + [f'neutral_sentiment_bert_{word}' for word in words] + [f'polarity_bert_{word}' for word in words]


def aggregate_sentiment(data):

    # company
    def mean_company_sentiment(col):
        return col[data['speaker_type'] == 'Corporate Participant'].mean()
    
    def std_company_sentiment(col):
        return col[data['speaker_type'] == 'Corporate Participant'].std()
    
    def min_company_sentiment(col):
        return col[data['speaker_type'] == 'Corporate Participant'].min()
    
    def max_company_sentiment(col):
        return col[data['speaker_type'] == 'Corporate Participant'].max()
    
    def median_company_sentiment(col):
        return col[data['speaker_type'] == 'Corporate Participant'].median()

    # company and presentation
    def mean_company_presentation_sentiment(col):
        return col[(data['speaker_type'] == 'Corporate Participant') & (data['type'] == 'presentation')].mean()
    
    def std_company_presentation_sentiment(col):
        return col[(data['speaker_type'] == 'Corporate Participant') & (data['type'] == 'presentation')].std()
    
    def min_company_presentation_sentiment(col):
        return col[(data['speaker_type'] == 'Corporate Participant') & (data['type'] == 'presentation')].min()
    
    def max_company_presentation_sentiment(col):
        return col[(data['speaker_type'] == 'Corporate Participant') & (data['type'] == 'presentation')].max()
    
    def median_company_presentation_sentiment(col):
        return col[(data['speaker_type'] == 'Corporate Participant') & (data['type'] == 'presentation')].median()
    

    # company and qna
    def mean_company_qna_sentiment(col):
        return col[(data['speaker_type'] == 'Corporate Participant') & (data['type'] == 'qna')].mean()
    
    def std_company_qna_sentiment(col):
        return col[(data['speaker_type'] == 'Corporate Participant') & (data['type'] == 'qna')].std()
    
    def min_company_qna_sentiment(col):
        return col[(data['speaker_type'] == 'Corporate Participant') & (data['type'] == 'qna')].min()
    
    def max_company_qna_sentiment(col):
        return col[(data['speaker_type'] == 'Corporate Participant') & (data['type'] == 'qna')].max()
    
    def median_company_qna_sentiment(col):
        return col[(data['speaker_type'] == 'Corporate Participant') & (data['type'] == 'qna')].median()
    

    # analyst overall
    def mean_analyst_sentiment(col):
        return col[data['speaker_type'] == 'Conference Participant'].mean()
    
    def std_analyst_sentiment(col):
        return col[data['speaker_type'] == 'Conference Participant'].std()
    
    def min_analyst_sentiment(col):
        return col[data['speaker_type'] == 'Conference Participant'].min()
    
    def max_analyst_sentiment(col):
        return col[data['speaker_type'] == 'Conference Participant'].max()
    
    def median_analyst_sentiment(col):
        return col[data['speaker_type'] == 'Conference Participant'].median()
    

    def mean_topic_sentiment(col):
        if len(col[col != -1]) == 0:
            return -1
        return col[col != -1].mean()

    # for each transcript, average global sentiment, and average sentiment per section and per speaker
    aggregations = dict()
    for col in global_sentiment_cols:
        aggregations[col] = ['mean', 'std', 'median', 'min', 'max',
                            mean_company_sentiment, std_company_sentiment, min_company_sentiment, max_company_sentiment, median_company_sentiment,
                            mean_company_presentation_sentiment, std_company_presentation_sentiment, min_company_presentation_sentiment, max_company_presentation_sentiment, median_company_presentation_sentiment,
                            mean_company_qna_sentiment, std_company_qna_sentiment, min_company_qna_sentiment, max_company_qna_sentiment, median_company_qna_sentiment,
                            mean_analyst_sentiment, std_analyst_sentiment, min_analyst_sentiment, max_analyst_sentiment, median_analyst_sentiment]

    for col in topic_sentiment_cols:
        aggregations[col] = [mean_topic_sentiment]

    data = data.groupby(['company_name', 'date'])[global_sentiment_cols + topic_sentiment_cols].agg(
        aggregations,
    )
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

    return data



def topic_modeling_sentiment(sentence_topics, output):
    topics = sentence_topics['Topic'].unique()

    # Initialize new columns for each topic with default values -1
    for topic in topics:
        sentence_topics[f'Topic_{topic}_positive'] = -1
        sentence_topics[f'Topic_{topic}_negative'] = -1
        sentence_topics[f'Topic_{topic}_neutral'] = -1

    # Populate the columns based on the topic of the sentence
    for index, row in sentence_topics.iterrows():
        topic = row['Topic']
        if topic != -1:
            sentence_topics.at[index, f'Topic_{topic}_positive'] = row['positive_sentiment_bert']
            sentence_topics.at[index, f'Topic_{topic}_negative'] = row['negative_sentiment_bert']
            sentence_topics.at[index, f'Topic_{topic}_neutral'] = row['neutral_sentiment_bert']

    new_sentence_topics = sentence_topics.drop(['positive_sentiment_bert', 'negative_sentiment_bert', 'neutral_sentiment_bert', 'polarity_bert'], axis=1)
    # drop rows with topic -1
    new_sentence_topics = new_sentence_topics[new_sentence_topics['Topic'] != -1]

    new_sentence_topics = pd.merge(new_sentence_topics, output, on='ID', how='left')

    topics_columns = [f'Topic_{topic}_positive' for topic in topics] + [f'Topic_{topic}_negative' for topic in topics] + [f'Topic_{topic}_neutral' for topic in topics]

    def mean_topic_sentiment(col):
        if len(col[col != -1]) == 0:
            return -1
        return col[col != -1].mean()

    new_sentence_topics = new_sentence_topics.groupby(['company_name', 'date'])[topics_columns].agg(mean_topic_sentiment)

    new_sentence_topics = new_sentence_topics.reset_index()

    new_sentence_topics = new_sentence_topics.drop(['Topic_-1_positive', 'Topic_-1_negative', 'Topic_-1_neutral'], axis=1)

    return new_sentence_topics


def merge_sentiment_dataframes(sentiment1, sentiment2):
    total = pd.merge(sentiment1, sentiment2, on=['company_name', 'date'], how='left')
    return total