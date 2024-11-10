import pandas as pd
import torch
import io
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import os
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

class TestDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=16):
        try:
            self.data = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            self.data = pd.read_csv(csv_file, encoding='cp949')
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        
        encoding = self.tokenizer(
            str(text),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'text': text,
            'label': label
        }

def load_model_from_parts(model_path):
    """분할된 모델 파일들을 불러와서 하나로 합침"""
    model_path = Path(model_path)
    
    # 모델 정보 불러오기
    info_path = model_path / 'model_info.pt'
    if not info_path.exists():
        raise FileNotFoundError(f"Model info file not found: {info_path}")
    
    info = torch.load(info_path)
    total_parts = info['num_parts']
    
    # 모든 부분을 순서대로 합치기
    model_bytes = bytearray()
    for i in range(total_parts):
        part_path = model_path / f"model_part_{i}.pt"
        if not part_path.exists():
            raise FileNotFoundError(f"Model part file not found: {part_path}")
            
        with open(part_path, 'rb') as f:
            part_bytes = f.read()
            model_bytes.extend(part_bytes)
    
    # 바이트를 state_dict로 변환
    buffer = io.BytesIO(model_bytes)
    state_dict = torch.load(buffer)
    
    return state_dict

class KoreanNameRecognizer:
    def __init__(self):
        self.model_path = Path('tests/saved_model')
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 토크나이저와 레이블 불러오기
        self.tokenizer = BertTokenizer.from_pretrained(str(self.model_path))
        self.labels = torch.load(self.model_path / 'labels.pt')
        
        # 모델 초기화 및 분할된 가중치 불러오기
        print("Loading model parts...")
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=len(self.labels)
        )
        
        # 분할된 모델 파일들을 불러와서 합치기
        state_dict = load_model_from_parts(self.model_path)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def predict_batch(self, csv_file):
        """CSV 파일의 데이터에 대해 배치 예측을 수행합니다."""
        dataset = TestDataset(csv_file, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_predictions = []
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            texts = batch['text']
            labels = batch['label']
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                
                for idx, (pred, text, label) in enumerate(zip(predictions, texts, labels)):
                    confidence = probabilities[idx][pred].item()
                    pred_label = self.labels[pred]
                    
                    all_predictions.append({
                        'text': text,
                        'actual_label': label,
                        'predicted_label': pred_label,
                        'correct_prediction': label == pred_label,
                        'confidence': confidence
                    })
        
        return all_predictions, None

def evaluate_korean_names(test_file: str = 'name_test_cases.csv'):
    """한국 이름 인식기 평가"""
    try:
        test_dir = Path("tests")
        test_filepath = test_dir / test_file
        
        print("Recognizer 초기화 중...")
        recognizer = KoreanNameRecognizer()
        
        print("평가 시작...")
        predictions, _ = recognizer.predict_batch(test_filepath)
        
        # 예측 결과에서 실제값과 예측값 추출
        y_true = []
        y_pred = []
        for p in predictions:
            # 실제 레이블과 예측 레이블을 직접 비교
            y_true.append(p['actual_label'])
            y_pred.append(p['predicted_label'])
        
        # 메트릭 계산
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average='macro')),
            "recall": float(recall_score(y_true, y_pred, average='macro')),
            "f1": float(f1_score(y_true, y_pred, average='macro'))
        }
        
        # 오류 분석
        error_analysis = analyze_errors(predictions)
        
        # 결과 출력
        print_evaluation_results(metrics, error_analysis)
        
        # 결과 저장
        save_evaluation_results(metrics, error_analysis, predictions, test_dir)
        
        return metrics, error_analysis, predictions
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None

def analyze_errors(results):
    """오류 분석"""
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r['correct_prediction'])
    total_errors = total_samples - correct_predictions
    
    confidence_stats = {
        'high_confidence': sum(1 for r in results if r['confidence'] > 0.9),
        'medium_confidence': sum(1 for r in results if 0.5 <= r['confidence'] <= 0.9),
        'low_confidence': sum(1 for r in results if r['confidence'] < 0.5)
    }
    
    misclassified = [r for r in results if not r['correct_prediction']]
    error_types = {}
    for error in misclassified:
        error_type = f"{error['actual_label']}->{error['predicted_label']}"
        error_types[error_type] = error_types.get(error_type, 0) + 1
    
    return {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "confidence_distribution": confidence_stats,
        "error_types": error_types
    }

def print_evaluation_results(metrics, error_analysis):
    """결과 출력"""
    print("\n=== 평가 결과 ===")
    print(f"정확도 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"정밀도 (Precision): {metrics['precision']:.4f}")
    print(f"재현율 (Recall): {metrics['recall']:.4f}")
    print(f"F1 점수: {metrics['f1']:.4f}")

    print("\n=== 오류 분석 ===")
    print(f"총 샘플 수: {error_analysis['total_samples']}")
    print(f"총 오류 수: {error_analysis['total_errors']}")
    
    print("\n신뢰도 분포:")
    for conf_level, count in error_analysis['confidence_distribution'].items():
        print(f"- {conf_level}: {count}개")
    
    print("\n오류 유형별 분포:")
    for error_type, count in error_analysis['error_types'].items():
        print(f"- {error_type}: {count}개")

def save_evaluation_results(metrics, error_analysis, detailed_results, test_dir):
    """평가 결과 저장"""
    result_filepath = test_dir / 'evaluation_name_results.json'
    
    # tensor를 일반 파이썬 타입으로 변환
    processed_results = []
    for result in detailed_results:
        processed_result = {
            'text': str(result['text']),
            'actual_label': str(result['actual_label']),
            'predicted_label': str(result['predicted_label']),
            'correct_prediction': bool(result['correct_prediction']),
            'confidence': float(result['confidence'])
        }
        processed_results.append(processed_result)
    
    save_results = {
        "metrics": metrics,
        "error_analysis": error_analysis,
        "detailed_results": processed_results
    }
    
    with open(result_filepath, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n상세 평가 결과가 '{result_filepath}'에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    try:
        test_dir = Path("tests")
        test_dir.mkdir(exist_ok=True)
        evaluate_korean_names()
    except Exception as e:
        print(f"Error: 평가 중 오류가 발생했습니다 - {str(e)}")

if __name__ == "__main__":
    main()