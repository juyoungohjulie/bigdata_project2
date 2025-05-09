# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import (Input, Embedding, Conv1D, BatchNormalization,
#                                      ReLU, Dropout, GlobalMaxPooling1D, Concatenate, Dense)

# class Emotion_Classifier(nn.Module):
#     def __init__(self, num_classes):
#         super(Emotion_Classifier, self).__init__()
#         # Input layer
#         input_layer = Input(shape=(max_len,))  # max_len 정의되어 있어야 함

#         # Branch 1
#         branch1 = Sequential()
#         branch1.add(Embedding(max_words, embedding_dim, input_length=max_len))  # input_length 제거
#         branch1.add(Conv1D(64, 3, padding='same', activation='relu'))
#         branch1.add(BatchNormalization())
#         branch1.add(ReLU())
#         branch1.add(Dropout(0.5))
#         branch1.add(GlobalMaxPooling1D())
#         out1 = branch1(input_layer)  # 명시적으로 호출

#         # Branch 2
#         branch2 = Sequential()
#         branch2.add(Embedding(max_words, embedding_dim))
#         branch2.add(Conv1D(64, 3, padding='same', activation='relu'))
#         branch2.add(BatchNormalization())
#         branch2.add(ReLU())
#         branch2.add(Dropout(0.5))
#         branch2.add(GlobalMaxPooling1D())
#         out2 = branch2(input_layer)

#         # Concatenate outputs
#         concatenated = Concatenate()([out1, out2])

#         # Fully connected layers
#         hid_layer = Dense(128, activation='relu')(concatenated)
#         dropout = Dropout(0.3)(hid_layer)

#         num_classes = tr_y.shape[1]
#         output_layer = Dense(num_classes, activation='softmax')(dropout)

    
#     def forward(self, input_ids, attention_mask, labels=None):
#         return self.bert(input_ids, attention_mask=attention_mask, labels=labels)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Embedding, Conv1D, BatchNormalization,
                                    ReLU, Dropout, GlobalMaxPooling1D, Concatenate, Dense)
import tensorflow as tf

class Emotion_Classifier(tf.keras.Model):
    def __init__(self, max_words, max_len, embedding_dim, num_classes):
        super(Emotion_Classifier, self).__init__()
        
        # 입력 레이어 정의
        self.input_layer = Input(shape=(max_len,))
        
        # 브랜치 1 정의
        self.branch1 = Sequential([
            Embedding(max_words, embedding_dim, input_length=max_len),
            Conv1D(64, 3, padding='same'),  # activation은 나중에 ReLU 레이어에서 적용
            BatchNormalization(),
            ReLU(),
            Dropout(0.5),
            GlobalMaxPooling1D()
        ])
        
        # 브랜치 2 정의
        self.branch2 = Sequential([
            Embedding(max_words, embedding_dim, input_length=max_len),
            Conv1D(64, 3, padding='same'),
            BatchNormalization(),
            ReLU(),
            Dropout(0.5),
            GlobalMaxPooling1D()
        ])
        
        # 연결 레이어
        self.concat = Concatenate()
        
        # 완전 연결 레이어
        self.dense = Dense(128, activation='relu')
        self.dropout = Dropout(0.3)
        self.output_layer = Dense(num_classes, activation='softmax')
        
        # 모델 초기화 (build 메서드를 호출)
        self.build_model()
        
    def build_model(self):
        # 모델 구성
        out1 = self.branch1(self.input_layer)
        out2 = self.branch2(self.input_layer)
        concatenated = self.concat([out1, out2])
        hidden = self.dense(concatenated)
        dropout = self.dropout(hidden)
        outputs = self.output_layer(dropout)
        
        # 최종 모델 생성 및 저장
        self.model = Model(inputs=self.input_layer, outputs=outputs)
        
    def call(self, inputs, training=None):
        # 모델 호출 시 forward pass 정의
        return self.model(inputs, training=training)
    
    def summary(self):
        # 모델 요약 정보 출력
        return self.model.summary()