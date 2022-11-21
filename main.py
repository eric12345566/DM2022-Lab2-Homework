## GPU Setup
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

# GPU
print("----------------- GPU -----------------")

os.environ['CUDA_VISIBLE_DEVICES']='0'
print(device_lib.list_local_devices())

print("----------------- GPU End -----------------")

print("Load Data...")

emotion_df = pd.read_csv('emotion.csv')
di_df = pd.read_csv('data_identification.csv')
sample_sub_df = pd.read_csv('sampleSubmission.csv')

with open('tweets_DM.json') as f:
    tweets_df = pd.json_normalize(json.loads(line)['_source']['tweet'] for line in f)

t_data = pd.merge(tweets_df, emotion_df)

print("Load OK!")

print("Download nltk...")

nltk.download('punkt')

print("Download ok...")

t_data_sample = t_data.sample(n=300000)

print("BOW_vectorizer & train_test_split Runing...")

# build analyzers (bag-of-words)
BOW_vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize) 

# apply analyzer to training data
BOW_vectorizer.fit(t_data_sample['text'])

X_train, X_test, y_train, y_test = train_test_split(t_data_sample['text'], t_data_sample['emotion'] ,
                                   random_state=104, 
                                   test_size=0.25, 
                                   shuffle=True)

X_train = BOW_vectorizer.transform(X_train)
X_test = BOW_vectorizer.transform(X_test)

print("x_train.shape: ", X_train.shape)
print("y_train.shape: ", y_train.shape)
print("x_test.shape: ", X_test.shape)
print("y_test.shape: ", y_test.shape)

print("BOW_vectorizer & train_test_split Finish!")


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


input_shape = X_train.shape[1]
print('input_shape: ', input_shape)

output_shape = len(label_encoder.classes_)
print('output_shape: ', output_shape)


# input layer
model_input = Input(shape=(input_shape, ))  # 500
X = model_input

# 1st hidden layer
X_W1 = Dense(units=2048)(X)  # 2048
H1 = ReLU()(X_W1)
H1 = BatchNormalization()(H1)
# 2nd hidden layer
H1_W2 = Dense(units=1024)(H1)  # 1024
H2 = ReLU()(H1_W2)
H2 = BatchNormalization()(H2)

# 2nd hidden layer
H2_W3 = Dense(units=512)(H2)  # 512
H3 = ReLU()(H2_W3)
H3 = BatchNormalization()(H3)

# 2nd hidden layer
H3_W4 = Dense(units=128)(H3)  # 128
H4 = ReLU()(H3_W4)
H4 = BatchNormalization()(H4)

# 2nd hidden layer
H4_W5 = Dense(units=64)(H4)  # 64
H5 = ReLU()(H4_W5)
H5 = BatchNormalization()(H5)

# output layer
H5_W6 = Dense(units=output_shape)(H5)  # 8
H6 = Softmax()(H5_W6)

model_output = H6

# create model
model = Model(inputs=[model_input], outputs=[model_output])

# loss function & optimizer
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# show model construction
model.summary()

csv_logger = CSVLogger('logs/training_log.csv')

# training setting
epochs = 50
batch_size = 128

callbacks = [ModelCheckpoint(filepath='./model/model_{epoch}.h5', save_weights_only=True, verbose=1, save_best_only=True), csv_logger]

# training!
history = model.fit(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    callbacks=callbacks,
                    validation_data = (X_test, y_test))
print('training finish')

pred_result = model.predict(X_test, batch_size=128)
pred_result = label_decode(label_encoder, pred_result)
print('testing accuracy: {}'.format(round(accuracy_score(label_decode(label_encoder, y_test), pred_result), 2)))
