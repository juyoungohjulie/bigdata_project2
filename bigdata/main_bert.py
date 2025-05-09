import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read datasets
df_train = pd.read_csv('train.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv('val.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv('test.txt', names=['Text', 'Emotion'], sep=';')

def analyze_dataset(df, name="Dataset"):
    print(f"\n=== {name} ===\n")

    # Dimensions
    print(f"Shape: {df.shape}")

    # Aperçu
    print("\nAperçu:")
    print(df.head(3))

    # Valeurs manquantes
    print("\nMissing values:")
    print(df.isnull().sum())

    # Duplicats
    num_duplicates = df.duplicated().sum()
    print(f"\nDuplicated rows: {num_duplicates}")

    # Colonnes et types
    print("\nDtypes:")
    print(df.dtypes)

    # Statistiques descriptives
    print("\nDescription:")
    print(df.describe(include="all"))

    # Distribution des émotions si la colonne existe
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x="Emotion", order=df["Emotion"].value_counts().index,palette='Set1')
    plt.title(f"{name} - Distribution des émotions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Analyse des 3 datasets
analyze_dataset(df_train, "Train")
analyze_dataset(df_test, "Test")
analyze_dataset(df_val, "Validation")

##########################2. DistilBert MODEL : Training and Evaluation##########################
# Create a mapping from emotion names to numerical indices
emotion_labels = df_train['Emotion'].unique()
label_mapping = {emotion: idx for idx, emotion in enumerate(emotion_labels)}

# Convert string labels to numerical indices
df_train['Emotion'] = df_train['Emotion'].map(label_mapping)
df_val['Emotion'] = df_val['Emotion'].map(label_mapping)
df_test['Emotion'] = df_test['Emotion'].map(label_mapping)

# Create a BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Preprocess the text data
train_texts = df_train['Text'].tolist()
val_texts = df_val['Text'].tolist()
test_texts = df_test['Text'].tolist()

train_labels = df_train['Emotion'].tolist()
val_labels = df_val['Emotion'].tolist()
test_labels = df_test['Emotion'].tolist()


# Convert text data to BERT input format
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Create dataset instances
train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)
test_dataset = EmotionDataset(test_encodings, test_labels)


# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pre-trained BERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(train_labels)))

# Set the device (GPU or CPU)
device = torch.device('cpu')
model.to(device)

# Define the training loop
def train(model, device, loader, optimizer, epoch):
    model.train()
    total_loss = 0

    for batch in loader:

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(loader)}')

    # Define the evaluation loop
def evaluate(model, device, loader):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / len(loader.dataset)
    return accuracy

    # Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(5):
    train(model, device, train_loader, optimizer, epoch)
    val_acc = evaluate(model, device, val_loader)
    print(f'Epoch {epoch+1}, Val Acc: {val_acc:.4f}')

# Evaluate the model on the test set
test_acc = evaluate(model, device, test_loader)
print(f'Test Acc: {test_acc:.4f}')

# Save the model's state dictionary

torch.save(model.state_dict(), 'model.pth')


# Initialize the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(train_labels)))

model.load_state_dict(torch.load('model.pth'))