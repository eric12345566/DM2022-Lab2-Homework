import os
from tensorflow.python.client import device_lib
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import ReLU, Softmax, BatchNormalization
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


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

print("<Info> Load OK!")

print("<Info> Tokenizing...")
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(t_data['text'])

print("<Info> Tokenize OK")

print("<Info> get_sequences...")
def get_sequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=50, padding='post')
    return padded_sequences

padded_train_sequences = get_sequences(tokenizer, t_data['text'])

print("<Info> get_sequences OK")

X_train, X_test, y_train, y_test = train_test_split(padded_train_sequences, t_data['emotion'] ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)

print("x_train.shape: ", X_train.shape)
print("y_train.shape: ", y_train.shape)
print("x_test.shape: ", X_test.shape)
print("y_test.shape: ", y_test.shape)


label_encoder = LabelEncoder()
label_encoder.fit(y_train)
print('check label: ', label_encoder.classes_)
print('\n## Before convert')
print('y_train[0:4]:\n', y_train[0:4])
print('\ny_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

def label_encode(le, labels):
    enc = le.transform(labels)
    return np_utils.to_categorical(enc)

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)

y_train = label_encode(label_encoder, y_train)
y_test = label_encode(label_encoder, y_test)

print('\n\n## After convert')
print('y_train[0:4]:\n', y_train[0:4])
print('\ny_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)


model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=50),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(8, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

csv_logger = CSVLogger('logs/training_log.csv')
callbacks = [ModelCheckpoint(filepath='./model/model_{epoch}.h5', verbose=1, save_best_only=True), csv_logger]


h = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    batch_size=32,
    epochs=25,
    callbacks=callbacks,
)

eval = model.evaluate(X_test, y_test)
