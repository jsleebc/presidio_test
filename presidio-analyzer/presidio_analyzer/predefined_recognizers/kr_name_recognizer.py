import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os

class CustomDataset(Dataset):
    def __init__(self, csv_files, tokenizer, max_length=16):
        # 여러 CSV 파일을 결합
        dataframes = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, encoding='cp949')
            dataframes.append(df)
        self.data = pd.concat(dataframes, ignore_index=True)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = self.data['label'].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        label_idx = torch.tensor(self.labels.tolist().index(label))
        
        encoding = self.tokenizer(str(text), 
                                padding='max_length', 
                                truncation=True, 
                                max_length=self.max_length, 
                                return_tensors='pt')
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask, label_idx

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_samples += len(labels)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / num_samples

    return accuracy, precision, recall, f1, avg_loss

def main():
    # 파일 경로 설정
    csv_files = ['tests/train_name.csv', 'tests/korean_name.csv']
    
    # BERT 토크나이저 초기화
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 데이터셋 생성
    dataset = CustomDataset(csv_files=csv_files, tokenizer=tokenizer)
    
    # 데이터 분할
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # 데이터로더 설정
    batch_size = 16
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # 모델 초기화
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                        num_labels=len(dataset.labels))
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 학습 설정
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    # 학습
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for input_ids, attention_mask, labels in train_dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # 검증
        acc, pre, rec, f1, val_loss = evaluate_model(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Validation - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # 모델 저장
    save_dir = 'saved_model'
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # 레이블 정보 저장
    torch.save(dataset.labels, os.path.join(save_dir, 'labels.pt'))

if __name__ == "__main__":
    main()