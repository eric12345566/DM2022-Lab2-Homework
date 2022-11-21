import os
from tensorflow.python.client import device_lib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import spacy
import spacy.cli
from tqdm import tqdm
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Embedding
from keras.initializers import Constant


print("----------------- GPU -----------------")

os.environ['CUDA_VISIBLE_DEVICES']='0'
print(device_lib.list_local_devices())

print("----------------- GPU End -----------------")

print("<Info> Load Data...")

emotion_df = pd.read_csv('emotion.csv')
di_df = pd.read_csv('data_identification.csv')
sample_sub_df = pd.read_csv('sampleSubmission.csv')
tweets_df = pd.read_pickle("tweets_df.pkl")

t_data = pd.merge(tweets_df, emotion_df)
t_data = t_data.sample(n=700000)
print("<Info> Load OK!")


# print("<Info> Word2Vec Model loading..")

# model = Word2Vec.load("wiki-lemma-100D-phrase")
# model.wv.save_word2vec_format('keras_word2vec.txt', binary=False)
# print("<Info> Word2Vec Model load OK!")


print("<Info> load the whole embedding into memory...")

# load the whole embedding into memory
embeddings_index = dict()
with open('keras_word2vec.txt') as f:
  for line in f:
    values = line.split()
    # 只接受長度為 101 的向量 (word + 100d embedding)
    if len(values) != 101:
      print(values)
      continue
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

print("<Info> embedding load ok!")


print("<Info> data text word2vec transfer (Take much time)...")
spacy.cli.download("en_core_web_md")
nlp = spacy.load('en_core_web_md', disable=["ner", "parser"])

review_lines = list()
reviews = t_data['text'].values.tolist()

for review in tqdm(reviews):   
    doc = nlp(review)
    words = [word.lemma_ for word in doc]
    review_lines.append(words)

print("<Info> data text word2vec transfer ok!")

print("<Info> save the review_lines data...")

np.save("review_lines", review_lines)

print("<Info> save the review_lines data ok!")

sentiment = t_data['emotion'].values

label_encoder = LabelEncoder()
label_encoder.fit(sentiment)
print('check label: ', label_encoder.classes_)
print('\n## Before convert')
print('y_train[0:4]:\n', sentiment[0:4])
print('\ny_train.shape: ', sentiment.shape)

def label_encode(le, labels):
    enc = le.transform(labels)
    return np_utils.to_categorical(enc)

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)

sentiment_oh = label_encode(label_encoder, sentiment)

print('\n\n## After convert')
print('y_train[0:4]:\n', sentiment_oh[0:4])
print('\ny_train.shape: ', sentiment_oh.shape)


sentence_lengths = t_data['text'].apply(lambda x: len(x.split()))
quantiles = np.percentile(sentence_lengths, [25, 50, 75])

print('The minimal word count in a movie review: ', min(sentence_lengths))
print('The first quantile (25%) of word count in a movie review:', quantiles[0])
print('The second quantile (50%) of word count in a movie review:', quantiles[1])
print('The third quantile (75%) of word count in a movie review:', quantiles[2])
print('The maximum word count in a movie review: ', max(sentence_lengths))


# 初始化 tokenizer
# 將文字轉換成 2D integer tensor
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(review_lines)
sequences = tokenizer_obj.texts_to_sequences(review_lines)

# 因為model的input必須一樣長，所以要使用padding，句子短於512的都會補0，長於512的會被truncate
word_index = tokenizer_obj.word_index
print('Found %s unique tokens.' % len(word_index))
review_pad = pad_sequences(sequences, maxlen=50)

print('Shape of review tensor:', review_pad.shape)
print('Shape of sentiment tensor:', sentiment_oh.shape)


vocab_size = len(word_index) + 1
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer_obj.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


# define model
gru_model = Sequential()
e = Embedding(vocab_size, 100, embeddings_initializer=Constant(embedding_matrix), trainable=False)
gru_model.add(e)
gru_model.add(GRU(64, dropout = 0.2, recurrent_dropout = 0.2))
gru_model.add(Dense(8, activation='softmax'))
# compile the model
gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# summarize the model
print(gru_model.summary())


# Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tflogs")
csv_logger = CSVLogger('logs/training_log_lstm_full.csv')
callbacks = [ModelCheckpoint(filepath='./model_lstm_full/model_lstm_full_{epoch}.h5', verbose=1, save_best_only=True), csv_logger, tensorboard_callback]

# fit the model
gru_model.fit(
    review_pad, sentiment_oh, 
    batch_size=128, 
    epochs=100, 
    validation_split=0.1, 
    verbose=1, 
    callbacks=callbacks)
