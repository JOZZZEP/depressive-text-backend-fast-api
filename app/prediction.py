import re
import numpy as np
import os
import string
import pickle
import pandas as pd
import io
import nltk

#run at frist time
# nltk.download('punkt', download_dir="../nltk/")
# nltk.download('stopwords', download_dir="../nltk/")
# nltk.download('wordnet', download_dir="../nltk/")
# nltk.download('averaged_perceptron_tagger', download_dir="../nltk/")
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#for local
# nltk.data.path.append('../nltk/')

#for docker
nltk.data.path.append('./nltk/')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

#for local
# tokenizer = pickle.load(open('../tokenizer/tokenizer.pkl', 'rb'))
# lstm_model = load_model("../model/lstm_model.h5")
# cnn_model = load_model("../model/cnn_model.h5")

#for docker
tokenizer = pickle.load(open('tokenizer/tokenizer.pkl', 'rb'))
lstm_model = load_model("model/lstm_model.h5")
cnn_model = load_model("model/cnn_model.h5")


predict_class = {
    0: "Happy",
    1: "Depressive Sadness",
    2: "Loss of Interest",
    3: "Appetite",
    4: "Sleep",
    5: "Thinking",
    6: "Guilt",
    7: "Tired",
    8: "Movement",
    9: "Suicidal"
}

def preprocess_text(text):
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    # Lowercasing
    text = text.lower()

    # Remove tag HTML
    text = re.sub(r'<[^\s]*>', ' ', text)

    # Remove Mentions and Hashtag
    text = re.sub(r'(rt\s|rt)?[@]\w+\s*', ' ', text, flags=re.IGNORECASE)

    # Remove URL
    text = re.sub(r'https?:[^\s]+', ' ', text)

    # Remove Number
    text = re.sub(r'\d+', ' ', text)

    # Remove Punctuation
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

    # Remove Special Letters
    text = re.sub(r'[\n\t\b\r]', '', text)

    # Remove non English
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Word Tokenization
    tokens = word_tokenize(text)

    # Lemma
    lemmatizer = WordNetLemmatizer()
    tagged_tokens = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_tokens]

    # Remove Stop Words
    stop_words = list(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

def lstm_predict(text):
    text_prepro = preprocess_text(text)
    X = pad_sequences(tokenizer.texts_to_sequences([text_prepro]), maxlen=300)
    test_predict = lstm_model.predict(X,verbose = 0)
    return int(np.argmax(test_predict))

def cnn_predict(text):
    text_prepro = preprocess_text(text)
    X = pad_sequences(tokenizer.texts_to_sequences([text_prepro]), maxlen=300)
    test_predict = cnn_model.predict(X,verbose = 0)
    return int(np.argmax(test_predict))

def predict(text, predict_func):
    class_num = predict_func(text)
    return {"text": text, "prediction": class_num, "className": predict_class[class_num]}

def predict_csv(file_name, content, predict_func):
    df = pd.read_csv(io.StringIO(content.decode('utf-8')), header=None)
    total_row = len(df)
    df.columns = ['text']
    df['prediction'] = df['text'].apply(predict_func)
    grouped_df = df.groupby('prediction')['text'].apply(list).reset_index()
    grouped_df['class_name'] =  grouped_df['prediction'].map(predict_class)
    grouped_df['amount'] = grouped_df['text'].apply(len)
    grouped_df['percent'] = (grouped_df['amount'] / total_row * 100).round(2)
    grouped_df = grouped_df.sort_values(by='percent', ascending=False) 
    result = grouped_df.to_dict(orient='records')
    return {"fileName":file_name, "totalRow": total_row, "result":result}