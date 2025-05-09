from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pickle
import os
from model import Emotion_Classifier
import tensorflow as tf # TensorFlow 임포트 추가

def predict(text, weights_path, token_path, save_path=None):

    
    # 모델 재생성
    max_words = 10000  # 원래 사용한 값으로 설정
    max_len = 50
    embedding_dim = 32
    num_classes = 4
    
    model = Emotion_Classifier(max_words, max_len, embedding_dim, num_classes)
    
    # 모델 빌드: 입력 형태를 지정합니다. (batch_size, sequence_length)
    # 여기서 max_len은 50입니다.
    model.build(input_shape=(None, max_len)) 
    
    model.compile(optimizer='adamax', loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    
    # 가중치만 로드
    model.load_weights(weights_path)
    
    with open(token_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    sequences = tokenizer.texts_to_sequences([text])
    x_new = pad_sequences(sequences, maxlen=max_len) # maxlen 사용 일관성
    predictions = model.predict(x_new)
    
    emotions = {0: 'anger', 1: 'fear', 2: 'joy', 3:'sadness'}
    # emotions = {'anger': 0, 'fear': 1, 'joy': 2, 'love': 3, 'sadness': 4, 'surprise': 5}
    
    # 결과 그래프 생성
    plt.figure(figsize=(10, 6))
    label = list(emotions.values())
    probs = list(predictions[0])
    labels = label
    bars = plt.barh(labels, probs)
    plt.xlabel('Probability', fontsize=15)
    plt.title(f'Emotion Prediction: "{text}"', fontsize=16)
    ax = plt.gca()
    ax.bar_label(bars, fmt = '%.2f')
    
    # 그래프 저장
    if save_path is None:
        dir_path = 'predictions'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        safe_text = ''.join(c if c.isalnum() else '_' for c in text[:20])
        file_path = os.path.join(dir_path, f'prediction_{safe_text}.png')
    else:
        file_path = save_path
    
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"예측 결과가 저장되었습니다: {os.path.abspath(file_path)}")
    
    plt.show()
    
    return predictions, emotions