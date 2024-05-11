import librosa
import numpy as np
import os
from transformers import ClapModel, ClapProcessor
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import torch
import torch.nn as nn


def librosa_process(y, sr=16000):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return tempo, chroma_stft.mean(), rmse.mean(), spec_cent.mean(), spec_bw.mean(), rolloff.mean(), zcr.mean(), mfcc.mean()


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)


def wave2vec_process(y):
    y_processed = processor(y, sampling_rate=16000)
    y_processed = y_processed['input_values'][0]
    y_processed = y_processed.reshape(1, -1)
    y_processed = torch.from_numpy(y_processed).to(device)

    with torch.no_grad():
            y_processed, sent = model(y_processed)

    # convert to numpy
    y_processed = y_processed.detach().cpu().numpy()
    sent = sent.detach().cpu().numpy()

    return sent


model_clap = ClapModel.from_pretrained("laion/larger_clap_general")
processor_clap = ClapProcessor.from_pretrained("laion/larger_clap_general")


def clap_process(y):
    inputs = processor_clap(audios=y, return_tensors="pt")
    audio_embed = model_clap.get_audio_features(**inputs)
    return audio_embed



# function to extract company and date from a format like 2020-Jul-30-AAPL.OQ-139668219181
def extract_company_date(file):
    date = file.split('-')[0] + '-' + file.split('-')[1] + '-' + file.split('-')[2]
    company = file.split('-')[3]
    return company, date


def get_features(path, librosa_ = True, wave2vec = False, embeddings=False):
    ''' Get features for each segment of the audio file'''
    clap_features = {}
    features = {}
    for i in range(1, 16):
        if librosa_:
            features[f'company_{i}'] = []
            features[f'date_{i}'] = []
            features[f'tempo_segment_{i}'] = []
            features[f'chroma_stft_segment_{i}'] = []
            features[f'rmse_segment_{i}'] = []
            features[f'spec_cent_segment_{i}'] = []
            features[f'spec_bw_segment_{i}'] = []
            features[f'rolloff_segment_{i}'] = []
            features[f'zcr_segment_{i}'] = []
            features[f'mfcc_segment_{i}'] = []

        if wave2vec:
            features[f'arousal_segment_{i}'] = []
            features[f'valence_segment_{i}'] = []
            features[f'dominance_segment_{i}'] = []
        
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mp3'):
                company, date = extract_company_date(root.split('/')[-1])
                # extract date from root
                y, _ = librosa.load(root + '/' + file, sr=16000)
                if librosa_:
                    tempo, chroma, rmse, spec_cent, spec_bw, rolloff, zcr, mfcc = librosa_process(y)
                    segment = int(file.split('_')[-1][:-4])
                    features[f'company_{segment}'].append(company)
                    features[f'date_{segment}'].append(date)
                    features[f'tempo_segment_{segment}'].append(tempo)
                    features[f'chroma_stft_segment_{segment}'].append(chroma)
                    features[f'rmse_segment_{segment}'].append(rmse)
                    features[f'spec_cent_segment_{segment}'].append(spec_cent)
                    features[f'spec_bw_segment_{segment}'].append(spec_bw)
                    features[f'rolloff_segment_{segment}'].append(rolloff)
                    features[f'zcr_segment_{segment}'].append(zcr)
                    features[f'mfcc_segment_{segment}'].append(mfcc)
                if wave2vec:
                    arousal, valence, dominance = wave2vec_process(y)
                    features[f'arousal_segment_{segment}'].append(arousal)
                    features[f'valence_segment_{segment}'].append(valence)
                    features[f'dominance_segment_{segment}'].append(dominance)
                if embeddings:
                    clap_features[(company, date, segment)] = clap_process(y)
    return features, clap_features
                        


    