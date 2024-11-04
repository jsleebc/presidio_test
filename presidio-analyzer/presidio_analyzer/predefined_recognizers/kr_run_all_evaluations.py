import os
import glob
import subprocess
import json
import pandas as pd
import time
from typing import Dict, List

def find_test_files(directory: str = ".") -> List[str]:
    """
    kr_test_*.py 형식의 모든 테스트 파일을 찾습니다
    """
    pattern = os.path.join(directory, "kr_test_*.py")
    return glob.glob(pattern)

def run_test_files(test_files: List[str]):
    """
    각 테스트 파일을 실행합니다
    """
    print("\n=== 테스트 실행 시작 ===")
    
    for file in test_files:
        test_name = os.path.basename(file).replace('.py', '')
        print(f"\n실행 중: {test_name}")
        
        try:
            # subprocess로 파일 실행
            result = subprocess.run(
                ['python', file],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            # 실행 결과 출력
            print(result.stdout)
            
            if result.stderr:
                print(f"경고/에러: {result.stderr}")
                
        except Exception as e:
            print(f"Error: {test_name} 실행 중 오류 발생 - {str(e)}")
        
        # 각 테스트 사이에 잠시 대기 (NLP 모델 로딩 시간 고려)
        time.sleep(2)

def calculate_average_metrics(test_dir: str = "tests") -> Dict:
    """
    모든 evaluation 결과를 로드하고 평균을 계산합니다
    """
    results = {}
    pattern = os.path.join(test_dir, "evaluation_*.json")
    result_files = glob.glob(pattern)
    
    for file_path in result_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                recognizer_type = os.path.basename(file_path).split('_')[1].split('.')[0]
                results[recognizer_type] = json.load(f)
        except Exception as e:
            print(f"Warning: {file_path} 로드 중 오류 발생 - {str(e)}")
    
    # 메트릭 수집 및 평균 계산
    metrics_list = []
    for recognizer_type, result in results.items():
        metrics = result.get('metrics', {})
        metrics['recognizer'] = recognizer_type
        metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
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
    
    table = "\n=== 전체 평가 결과 ===\n"
    table += f"{'Recognizer':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}\n"
    table += "-" * 60 + "\n"
    
    for item in individual:
        table += f"{item['recognizer']:<15} {item['accuracy']:>10.4f} {item['precision']:>10.4f} {item['recall']:>10.4f} {item['f1']:>10.4f}\n"
    
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
        # tests 디렉토리 생성
        os.makedirs("tests", exist_ok=True)
        
        # 테스트 파일 찾기
        test_files = find_test_files()
        
        if not test_files:
            print("Error: kr_test_*.py 형식의 테스트 파일을 찾을 수 없습니다.")
            return
            
        print(f"발견된 테스트 파일: {[os.path.basename(f) for f in test_files]}")
        
        # 모든 테스트 실행
        run_test_files(test_files)
        
        # 결과 수집 전 잠시 대기
        time.sleep(2)
        
        # 평균 메트릭 계산
        metrics = calculate_average_metrics()
        
        # 결과 출력
        print(format_metrics_table(metrics))
        
        # 결과 저장
        save_summary_results(metrics)
        print(f"\n종합 평가 결과가 'evaluation_summary.json'에 저장되었습니다.")
        
    except Exception as e:
        print(f"Error: 실행 중 오류가 발생했습니다 - {str(e)}")

if __name__ == "__main__":
    main()