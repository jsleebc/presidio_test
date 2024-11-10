import pandas as pd
import torch
import io
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os
from pathlib import Path

class CustomDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=16):
        self.data = pd.read_csv(csv_file, encoding='cp949')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = self.data['label'].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        label_idx = torch.tensor(self.labels.tolist().index(label))

        combined_text = f"{text} "

        encoding = self.tokenizer(combined_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask, label_idx

def evaluate_model(model, dataloader, criterion):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_samples += len(labels)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / num_samples

    return accuracy, precision, recall, f1, avg_loss

def save_model_in_parts(model, save_dir, part_size_mb=95):
    """모델을 여러 부분으로 나눠서 저장"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 모델의 state_dict를 바이트로 변환
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_bytes = buffer.getvalue()
    
    # 부분 크기 계산 (MB를 바이트로 변환)
    part_size = part_size_mb * 1024 * 1024
    total_parts = (len(model_bytes) + part_size - 1) // part_size
    
    print(f"Total model size: {len(model_bytes)/1024/1024:.2f}MB")
    print(f"Splitting into {total_parts} parts")
    
    # 모델을 여러 부분으로 나눠서 저장
    for i in range(total_parts):
        start_idx = i * part_size
        end_idx = min((i + 1) * part_size, len(model_bytes))
        part_bytes = model_bytes[start_idx:end_idx]
        
        part_path = save_dir / f"model_part_{i}.pt"
        with open(part_path, 'wb') as f:
            f.write(part_bytes)
        
        print(f"Saved part {i+1}/{total_parts}: {len(part_bytes)/1024/1024:.2f}MB")
    
    # 부분 정보 저장
    info = {
        'num_parts': total_parts,
        'total_size': len(model_bytes),
        'part_size': part_size
    }
    torch.save(info, save_dir / 'model_info.pt')

# 메인 코드
if __name__ == "__main__":
    # 토크나이저 초기화
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 데이터셋 생성
    csv_file_path = "train_name.csv"
    dataset = CustomDataset(csv_file=csv_file_path, tokenizer=tokenizer)
    batch_size = 16
    shuffle = True
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # 데이터 분할
    train_data, val_data = train_test_split(dataset, test_size=0.4, random_state=93)
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=93)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 모델 초기화
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(dataset.labels))

    # 디바이스 설정
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)

    # 학습 설정
    criterions = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-7, no_deprecation_warning=True)

    # 학습 히스토리 저장용
    train_accuracy_history = []
    train_f1_history = []
    val_accuracy_history = []
    val_f1_history = []

    # 학습 루프
    num_epochs = 30
    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        acc, pre, rec, f, val_loss = evaluate_model(model, val_dataloader, criterions)
        train_accuracy, train_f1, _1, _2, _3 = evaluate_model(model, train_dataloader, criterions)
        train_accuracy_history.append(train_accuracy)
        train_f1_history.append(train_f1)
        val_accuracy_history.append(acc)
        val_f1_history.append(f)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        print(f"Validation Accuracy: {acc:.4f}, Validation Precision: {pre:.4f}, Validation Recall: {rec:.4f}, Validation F1: {f:.4f}")

        # 최고 성능 모델 저장
        if f > best_val_f1:
            best_val_f1 = f
            print(f"New best F1 score: {f:.4f}. Saving model...")
            
            # 저장 디렉토리 생성
            save_dir = Path('saved_model')
            save_dir.mkdir(exist_ok=True, parents=True)
            
            # 모델을 분할하여 저장
            save_model_in_parts(model, save_dir)
            
            # 토크나이저와 레이블은 그대로 저장
            tokenizer.save_pretrained(str(save_dir))
            torch.save(dataset.labels, save_dir / 'labels.pt')

    # 학습 결과 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_accuracy_history, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracy_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_f1_history, label='Training F1 Score')
    plt.plot(range(1, num_epochs + 1), val_f1_history, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Training and Validation F1 Score')

    plt.tight_layout()
    plt.show()

    print(f"Training completed! Best validation F1 score: {best_val_f1:.4f}")
    print(f"Model saved in directory: saved_model")