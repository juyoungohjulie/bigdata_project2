import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.layers import Embedding, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from model import Emotion_Classifier
from preprocess import data_loader
from save_figs import save_figs_train_distribution, save_figs_val_distribution, save_figs_test_distribution, ensure_dir, save_figs_all, save_figs_confusion_matrix
from drop_love_sur import drop_love_sur_train, drop_love_sur_val, drop_love_sur_test
import seaborn as sns
import os
from predict import predict
import random
random.seed(42)

drop_label = 'y'
# Load data
df, val_df, ts_df = data_loader()

# Print the unique labels and their counts for each dataset
print(df['label'].unique())
print(df.label.value_counts())

# Save original data distribution for each dataset files
save_figs_train_distribution(df)
save_figs_val_distribution(val_df)
save_figs_test_distribution(ts_df)


# Choose whether to drop the love/surprise label based on the user choice
# drop_label = input("Do you want to drop the love/surprise label? (y/n): ")

if drop_label == 'y':
    df = drop_love_sur_train(df)
    val_df = drop_love_sur_val(val_df)
    ts_df = drop_love_sur_test(ts_df)

    save_figs_train_distribution(df, drop_label)
    save_figs_val_distribution(val_df, drop_label)
    save_figs_test_distribution(ts_df, drop_label)

#  data preprocessing

# 1. Split data into x and y (text and label)
train_text = df['sentence']
train_label = df['label']

val_text = val_df['sentence']
val_label = val_df['label']

test_text = ts_df['sentence']
test_label = ts_df['label']

#. 2. Encode the given data for each train, validation, and test set
encoder = LabelEncoder()
train_label = encoder.fit_transform(train_label)
val_label = encoder.transform(val_label)
test_label = encoder.transform(test_label)

# 3. Tokenize the given data for each train, validation, and test set
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_text)

sequences = tokenizer.texts_to_sequences(train_text)

train_x = pad_sequences(sequences, maxlen=50)
train_y = to_categorical(train_label)

sequences = tokenizer.texts_to_sequences(val_text)
val_x = pad_sequences(sequences, maxlen=50)
val_y = to_categorical(val_label)

sequences = tokenizer.texts_to_sequences(test_text)
test_x = pad_sequences(sequences, maxlen=50)
test_y = to_categorical(test_label)

# Build the CNN model
max_words = 10000
max_len = 50
embedding_dim = 32

num_classes = train_y.shape[1]
# output_layer = Dense(num_classes, activation='softmax')(dropout)

# Final model
# model = Model(inputs=input_layer, outputs=output_layer)
model = Emotion_Classifier(max_words, max_len, embedding_dim, num_classes)

model.compile(optimizer='adamax',
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall()])

print(model.summary()) # print the whole model structure

batch_size = 256
epochs = 25
# history = model.fit([tr_x, tr_x], tr_y, epochs=epochs, batch_size=batch_size,
#                     validation_data=([val_x, val_x], val_y))

# Train the model
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                    validation_data=(val_x, val_y))

# Evaluate and visualize the model in various formats
(loss, accuracy, percision, recall) = model.evaluate(train_x, train_y)
print(f'Loss: {round(loss, 2)}, Accuracy: {round(accuracy, 2)}, Precision: {round(percision, 2)}, Recall: {round(recall, 2)}')

(loss, accuracy, percision, recall) = model.evaluate(test_x, test_y)
print(f'Loss: {round(loss, 2)}, Accuracy: {round(accuracy, 2)}, Precision: {round(percision, 2)}, Recall: {round(recall, 2)}')

train_acc = history.history['accuracy']
train_loss = history.history['loss']
train_per = history.history['precision']
train_recall = history.history['recall']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
val_per = history.history['val_precision']
val_recall = history.history['val_recall']

index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
index_precision = np.argmax(val_per)
per_highest = val_per[index_precision]
index_recall = np.argmax(val_recall)
recall_highest = val_recall[index_recall]


Epochs = [i + 1 for i in range(len(train_acc))]
loss_label = f'Best epoch = {str(index_loss + 1)}'
acc_label = f'Best epoch = {str(index_acc + 1)}'
per_label = f'Best epoch = {str(index_precision + 1)}'
recall_label = f'Best epoch = {str(index_recall + 1)}'


dir_path = ensure_dir()
save_figs_all(Epochs, train_loss, val_loss, index_loss, val_lowest, loss_label, train_acc, val_acc, index_acc, acc_highest, acc_label, train_per, val_per, index_precision, per_highest, per_label, train_recall, val_recall, index_recall, recall_highest, recall_label)


###################################### confusion matrix ######################################
y_true=[]
for i in range(len(test_y)):
    x = np.argmax(test_y[i]) 
    y_true.append(x)

preds = model.predict(test_x)  
y_pred = np.argmax(preds, axis=1)

save_figs_confusion_matrix(y_true, y_pred)

# print the confusion matrix 
clr = classification_report(y_true, y_pred)
print(clr)

import pickle
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# model.save('nlp.h5')
model.save_weights('model2.weights.h5')
txt = 'I am so nervous until I finish this project'

predict(txt, 'model2.weights.h5', 'tokenizer.pkl')

txt = 'He laughed aloud, unable to hide his excitement.'
predict(txt, 'model2.weights.h5', 'tokenizer.pkl')

txt = 'Her hands trembled as the door creaked open.'
predict(txt, 'model2.weights.h5', 'tokenizer.pkl')

txt = 'He slammed the book shut, his face burning with rage.'
predict(txt, 'model2.weights.h5', 'tokenizer.pkl')

###################################### get_emotion_mappings ######################################
import pandas as pd

def get_emotion_mappings(file_paths):
    all_emotions = set()
    for file_path in file_paths:
        try:
            
            df = pd.read_csv(file_path, sep=';', header=None, names=['text', 'emotion'])
            all_emotions.update(df['emotion'].unique())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    sorted_emotions = sorted(list(all_emotions))
    
    emotion_to_id = {emotion: i for i, emotion in enumerate(sorted_emotions)}
    id_to_emotion = {i: emotion for i, emotion in enumerate(sorted_emotions)}
    
    return emotion_to_id, id_to_emotion, len(sorted_emotions)

file_paths = ['train.txt', 'val.txt', 'test.txt'] 

emotion_to_id, id_to_emotion, num_classes = get_emotion_mappings(file_paths)

print("Emotion to ID Mapping:")
print(emotion_to_id)
print("\nID to Emotion Mapping:")
print(id_to_emotion)
print(f"\nNumber of unique classes: {num_classes}")

# docker run -itd \
# 	-v /mnt/d/bigdata/:/workspace/bigData \
#   	--ipc=host \
#   	--gpus all \
#   	--name bigdata \
# 	tensorflow/tensorflow:latest-gpu

# seaborn pandas scikit-learn