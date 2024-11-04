import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

class NameRecognizer:
    def __init__(self, model_path='saved_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델과 토크나이저 불러오기
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.labels = torch.load(os.path.join(model_path, 'labels.pt'))
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        # 텍스트 전처리
        encoding = self.tokenizer(str(text),
                                padding='max_length',
                                truncation=True,
                                max_length=16,
                                return_tensors='pt')
        
        # 예측
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, dim=1)
        
        return self.labels[predicted.item()]

def main():
    # 모델 불러오기
    recognizer = NameRecognizer()
    
    # 테스트할 이름들
    test_names = [
        "김철수",
        "이영희",
        "John Smith",
        "Mary Johnson"
    ]
    
    # 예측
    for name in test_names:
        prediction = recognizer.predict(name)
        print(f"이름: {name} -> 예측: {prediction}")

if __name__ == "__main__":
    main()