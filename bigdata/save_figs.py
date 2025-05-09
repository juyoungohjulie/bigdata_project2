from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 저장할 디렉토리 생성
def ensure_dir(directory='figures'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def save_figs_train_distribution(df, drop_label='n'):
    dir_path = ensure_dir()
    file_path = os.path.join(dir_path, 'emotion_train_distribution.png')
    
    label_counts = df['label'].value_counts()
    light_colors = sns.husl_palette(n_colors=len(label_counts))
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=light_colors)
    print("drop_label: ", drop_label)
    if drop_label == 'y':
        file_path = os.path.join(dir_path, 'emotion_train_distribution_drop.png')
        plt.title('Emotion Train Distribution (Dropped Love/Surprise)')
    else:
        file_path = os.path.join(dir_path, 'emotion_train_distribution.png')
        plt.title('Emotion Train Distribution')
    
    plt.savefig(file_path)
    print(f"파일이 저장되었습니다: {os.path.abspath(file_path)}")
    plt.close()  # plt.show() 대신 plt.close() 사용
    return 

def save_figs_val_distribution(val_df, drop_label='n'):
    dir_path = ensure_dir()
    file_path = os.path.join(dir_path, 'emotion_valid_distribution.png')
    
    label_counts = val_df['label'].value_counts()
    light_colors = sns.husl_palette(n_colors=len(label_counts))
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=light_colors)
    if drop_label == 'y':
        file_path = os.path.join(dir_path, 'emotion_valid_distribution_drop.png')
        plt.title('Emotion Valid Distribution (Dropped Love/Surprise)')
    else:
        file_path = os.path.join(dir_path, 'emotion_valid_distribution.png')
        plt.title('Emotion Valid Distribution')
    
    plt.savefig(file_path)
    print(f"파일이 저장되었습니다: {os.path.abspath(file_path)}")
    plt.close()
    return

def save_figs_test_distribution(ts_df, drop_label='n'):
    dir_path = ensure_dir()
    file_path = os.path.join(dir_path, 'emotion_test_distribution.png')
    
    # Count label distributions
    label_counts = ts_df['label'].value_counts()
    light_colors = sns.husl_palette(n_colors=len(label_counts))
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=light_colors)
    if drop_label == 'y':
        file_path = os.path.join(dir_path, 'emotion_test_distribution_drop.png')
        plt.title('Emotion Test Distribution (Dropped Love/Surprise)')
    else:
        file_path = os.path.join(dir_path, 'emotion_test_distribution.png')
        plt.title('Emotion Test Distribution')
    
    plt.savefig(file_path)
    print(f"파일이 저장되었습니다: {os.path.abspath(file_path)}")
    plt.close()
    return




def save_figs_loss(Epochs, train_loss, val_loss, index_loss, val_lowest, loss_label):
    dir_path = ensure_dir()
    # Loss 그래프 - 새 figure 생성
    plt.figure(figsize=(10, 6))
    plt.style.use('fivethirtyeight')
    plt.plot(Epochs, train_loss, 'r', label='Train loss')
    plt.plot(Epochs, val_loss, 'g', label='Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dir_path, 'loss_metrics.png'))
    plt.close()
    return

def save_figs_accuracy(Epochs, train_acc, val_acc, index_acc, acc_highest, acc_label):
    dir_path = ensure_dir()
    # Accuracy 그래프 - 새 figure 생성
    plt.figure(figsize=(10, 6))
    plt.style.use('fivethirtyeight')
    plt.plot(Epochs, train_acc, 'r', label='Train Accuracy')
    plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dir_path, 'accuracy_metrics.png'))
    plt.close()
    return

def save_figs_precision(Epochs, train_per, val_per, index_precision, per_highest, per_label):
    dir_path = ensure_dir()

    # Precision 그래프 - 새 figure 생성
    plt.figure(figsize=(10, 6))
    plt.style.use('fivethirtyeight')
    plt.plot(Epochs, train_per, 'r', label='Precision')
    plt.plot(Epochs, val_per, 'g', label='Validation Precision')
    plt.scatter(index_precision + 1, per_highest, s=150, c='blue', label=per_label)
    plt.title('Precision and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dir_path, 'precision_metrics.png'))
    plt.close()
    return

def save_figs_recall(Epochs, train_recall, val_recall, index_recall, recall_highest, recall_label):
    dir_path = ensure_dir()
    # Recall 그래프 - 새 figure 생성
    plt.figure(figsize=(10, 6))
    plt.style.use('fivethirtyeight')
    plt.plot(Epochs, train_recall, 'r', label='Recall')
    plt.plot(Epochs, val_recall, 'g', label='Validation Recall')
    plt.scatter(index_recall + 1, recall_highest, s=150, c='blue', label=recall_label)
    plt.title('Recall and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dir_path, 'recall_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return

def save_figs_all(Epochs, train_loss, val_loss, index_loss, val_lowest, loss_label, train_acc, val_acc, index_acc, acc_highest, acc_label, train_per, val_per, index_precision, per_highest, per_label, train_recall, val_recall, index_recall, recall_highest, recall_label):
    # 전체 그래프 - 새 figure 생성
    dir_path = ensure_dir()
    plt.figure(figsize=(20, 12))
    plt.style.use('fivethirtyeight')

    plt.subplot(2, 2, 1)
    plt.plot(Epochs, train_loss, 'r', label='Train loss')
    plt.plot(Epochs, val_loss, 'g', label='Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(Epochs, train_acc, 'r', label='Train Accuracy')
    plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(Epochs, train_per, 'r', label='Precision')
    plt.plot(Epochs, val_per, 'g', label='Validation Precision')
    plt.scatter(index_precision + 1, per_highest, s=150, c='blue', label=per_label)
    plt.title('Precision and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(Epochs, train_recall, 'r', label='Recall')
    plt.plot(Epochs, val_recall, 'g', label='Validation Recall')
    plt.scatter(index_recall + 1, recall_highest, s=150, c='blue', label=recall_label)
    plt.title('Recall and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)

    plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
    plt.savefig(os.path.join(dir_path, 'all_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()  # 또는 plt.close()    
    print(f"loss, accuracy, precision, recall 파일이 저장되었습니다: {os.path.abspath(os.path.join(dir_path, 'all_metrics.png'))}")
    return

def save_figs_confusion_matrix(y_true, y_pred):
    plt.figure(figsize=(8,6))
    emotions = {0: 'anger', 1: 'fear', 2: 'joy', 3:'sadness'}
    emotions = list(emotions.values())
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # 히트맵 저장
    dir_path = ensure_dir()  # 이미 정의된 ensure_dir 함수 사용
    file_path = os.path.join(dir_path, 'confusion_matrix.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"혼동 행렬이 저장되었습니다: {os.path.abspath(file_path)}")
    return

    # file_path = os.path.join(dir_path, 'emotion_train_distribution.png')

# # Loss 그래프 - 새 figure 생성
# plt.figure(figsize=(10, 6))
# plt.style.use('fivethirtyeight')
# plt.plot(Epochs, train_loss, 'r', label='Train loss')
# plt.plot(Epochs, val_loss, 'g', label='Validation loss')
# plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
# plt.title('Train and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(dir_path, 'loss_metrics.png'), dpi=300, bbox_inches='tight')
# plt.close()

# # Accuracy 그래프 - 새 figure 생성
# plt.figure(figsize=(10, 6))
# plt.style.use('fivethirtyeight')
# plt.plot(Epochs, train_acc, 'r', label='Train Accuracy')
# plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
# plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
# plt.title('Train and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(dir_path, 'accuracy_metrics.png'), dpi=300, bbox_inches='tight')
# plt.close()

# # Precision 그래프 - 새 figure 생성
# plt.figure(figsize=(10, 6))
# plt.style.use('fivethirtyeight')
# plt.plot(Epochs, train_per, 'r', label='Precision')
# plt.plot(Epochs, val_per, 'g', label='Validation Precision')
# plt.scatter(index_precision + 1, per_highest, s=150, c='blue', label=per_label)
# plt.title('Precision and Validation Precision')
# plt.xlabel('Epochs')
# plt.ylabel('Precision')
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(dir_path, 'precision_metrics.png'), dpi=300, bbox_inches='tight')
# plt.close()

# # Recall 그래프 - 새 figure 생성
# plt.figure(figsize=(10, 6))
# plt.style.use('fivethirtyeight')
# plt.plot(Epochs, train_recall, 'r', label='Recall')
# plt.plot(Epochs, val_recall, 'g', label='Validation Recall')
# plt.scatter(index_recall + 1, recall_highest, s=150, c='blue', label=recall_label)
# plt.title('Recall and Validation Recall')
# plt.xlabel('Epochs')
# plt.ylabel('Recall')
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(dir_path, 'recall_metrics.png'), dpi=300, bbox_inches='tight')
# plt.close()

# # 전체 그래프 - 새 figure 생성
# plt.figure(figsize=(20, 12))
# plt.style.use('fivethirtyeight')

# plt.subplot(2, 2, 1)
# plt.plot(Epochs, train_loss, 'r', label='Train loss')
# plt.plot(Epochs, val_loss, 'g', label='Validation loss')
# plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# plt.subplot(2, 2, 2)
# plt.plot(Epochs, train_acc, 'r', label='Train Accuracy')
# plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
# plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
# plt.title('Train and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)

# plt.subplot(2, 2, 3)
# plt.plot(Epochs, train_per, 'r', label='Precision')
# plt.plot(Epochs, val_per, 'g', label='Validation Precision')
# plt.scatter(index_precision + 1, per_highest, s=150, c='blue', label=per_label)
# plt.title('Precision and Validation Precision')
# plt.xlabel('Epochs')
# plt.ylabel('Precision')
# plt.legend()
# plt.grid(True)

# plt.subplot(2, 2, 4)
# plt.plot(Epochs, train_recall, 'r', label='Recall')
# plt.plot(Epochs, val_recall, 'g', label='Validation Recall')
# plt.scatter(index_recall + 1, recall_highest, s=150, c='blue', label=recall_label)
# plt.title('Recall and Validation Recall')
# plt.xlabel('Epochs')
# plt.ylabel('Recall')
# plt.legend()
# plt.grid(True)

# plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
# plt.savefig(os.path.join(dir_path, 'all_metrics.png'), dpi=300, bbox_inches='tight')
# plt.show()  # 또는 plt.close()