from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Embedding, Conv1D, BatchNormalization,
                                    ReLU, Dropout, GlobalMaxPooling1D, Concatenate, Dense)
import tensorflow as tf

class Emotion_Classifier(tf.keras.Model):
    def __init__(self, max_words, max_len, embedding_dim, num_classes):
        super(Emotion_Classifier, self).__init__()
        
        # Input layer
        self.input_layer = Input(shape=(max_len,))
        
        # Branch 1
        self.branch1 = Sequential([
            Embedding(max_words, embedding_dim, input_length=max_len),
            Conv1D(64, 3, padding='same'),  
            BatchNormalization(),
            ReLU(),
            Dropout(0.5),
            GlobalMaxPooling1D()
        ])
        
        # Branch 2
        self.branch2 = Sequential([
            Embedding(max_words, embedding_dim, input_length=max_len),
            Conv1D(64, 3, padding='same'),
            BatchNormalization(),
            ReLU(),
            Dropout(0.5),
            GlobalMaxPooling1D()
        ])
        
        # Concatenate layer
        self.concat = Concatenate()
        
        # Fully connected layer
        self.dense = Dense(128, activation='relu')
        self.dropout = Dropout(0.3)
        self.output_layer = Dense(num_classes, activation='softmax')
        
        # Build model
        self.build_model()
        
    def build_model(self):
        out1 = self.branch1(self.input_layer)
        out2 = self.branch2(self.input_layer)
        concatenated = self.concat([out1, out2])
        hidden = self.dense(concatenated)
        dropout = self.dropout(hidden)
        outputs = self.output_layer(dropout)
        
        # Final model generation
        self.model = Model(inputs=self.input_layer, outputs=outputs)
        
    def call(self, inputs, training=None):
        # Call model
        return self.model(inputs, training=training)
    
    def summary(self):
        # Print model summary
        return self.model.summary()