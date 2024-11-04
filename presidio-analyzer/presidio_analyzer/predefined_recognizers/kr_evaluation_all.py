import os
import json
import pandas as pd
from typing import Dict, List
import glob

def load_evaluation_results(test_dir: str = "tests") -> Dict[str, Dict]:
    """
    tests 디렉토리에서 모든 evaluation 결과를 로드
    """
    results = {}
    
    # evaluation_*.json 파일 찾기
    pattern = os.path.join(test_dir, "evaluation_*.json")
    result_files = glob.glob(pattern)
    
    for file_path in result_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 파일명에서 recognizer 타입 추출 (예: email, creditcard 등)
                recognizer_type = os.path.basename(file_path).split('_')[1].split('.')[0]
                results[recognizer_type] = json.load(f)
        except Exception as e:
            print(f"Warning: {file_path} 로드 중 오류 발생 - {str(e)}")
    
    return results

def calculate_average_metrics(results: Dict[str, Dict]) -> Dict:
    """
    모든 recognizer의 평균 메트릭 계산
    """
    metrics_list = []
    
    for recognizer_type, result in results.items():
        metrics = result.get('metrics', {})
        metrics['recognizer'] = recognizer_type
        metrics_list.append(metrics)
    
    # DataFrame으로 변환하여 계산
    df = pd.DataFrame(metrics_list)
    
    # 평균 계산 (recognizer 컬럼 제외)
    numeric_columns = ['accuracy', 'precision', 'recall', 'f1']
    averages = df[numeric_columns].mean()
    
    return {
        'individual_metrics': metrics_list,
        'average_metrics': averages.to_dict()
    }

def format_metrics_table(metrics: Dict) -> str:
    """
    메트릭을 보기 좋은 테이블 형식으로 포매팅
    """
    individual = metrics['individual_metrics']
    avg = metrics['average_metrics']
    
    # 테이블 헤더
    table = "\n=== 개별 Recognizer 성능 ===\n"
    table += f"{'Recognizer':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}\n"
    table += "-" * 60 + "\n"
    
    # 개별 recognizer 결과
    for item in individual:
        table += f"{item['recognizer']:<15} {item['accuracy']:>10.4f} {item['precision']:>10.4f} {item['recall']:>10.4f} {item['f1']:>10.4f}\n"
    
    # 평균 메트릭
    table += "=" * 60 + "\n"
    table += f"{'평균':<15} {avg['accuracy']:>10.4f} {avg['precision']:>10.4f} {avg['recall']:>10.4f} {avg['f1']:>10.4f}\n"
    
    return table

def save_summary_results(metrics: Dict, output_file: str = "evaluation_summary.json"):
    """
    종합 결과를 JSON 파일로 저장
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

def main():
    try:
        # 결과 로드
        results = load_evaluation_results()
        
        if not results:
            print("Warning: 평가 결과 파일을 찾을 수 없습니다.")
            return
        
        # 평균 메트릭 계산
        metrics = calculate_average_metrics(results)
        
        # 결과 출력
        print("\n=== 평가 결과 종합 ===")
        print(format_metrics_table(metrics))
        
        # 결과 저장
        save_summary_results(metrics)
        print(f"\n종합 평가 결과가 'evaluation_summary.json'에 저장되었습니다.")
        
    except Exception as e:
        print(f"Error: 평가 중 오류가 발생했습니다 - {str(e)}")

if __name__ == "__main__":
    main()