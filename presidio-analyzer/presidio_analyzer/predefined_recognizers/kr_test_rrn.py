from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from kr_rrn_recognizer import KRRRNRecognizer
import json
import os
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_rrn_recognizer(test_file: str = 'rrn_test_cases_1000.json'):
    """주민등록번호 인식기 평가"""
    # 테스트 파일 로드
    test_dir = "tests"
    test_filepath = os.path.join(test_dir, test_file)
    
    print("테스트 케이스 로딩 중...")
    with open(test_filepath, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        test_cases = test_data['test_cases']

    # Analyzer 초기화
    print("Analyzer 초기화 중...")
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "ko", "model_name": "ko_core_news_lg"}]
    }
    
    provider = NlpEngineProvider(nlp_configuration=configuration)
    analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=["ko"]
    )
    
    # Recognizer 등록
    recognizer = KRRRNRecognizer()
    analyzer.registry.add_recognizer(recognizer)

    print("평가 시작...")
    total_cases = len(test_cases)
    y_true = []
    y_pred = []
    detailed_results = []

    for i, case in enumerate(test_cases, 1):
        if i % 100 == 0:
            print(f"진행 중... {i}/{total_cases}")

        text = case['text']
        is_valid = case['is_valid']
        rrn = case['rrn']
        clean_rrn = re.sub(r'[\s\-_.]', '', rrn)

        # 1단계: 패턴 매칭
        results = analyzer.analyze(
            text=text,
            language="ko",
            entities=["KR_RRN"]
        )

        # 2단계: 검증
        found_valid_rrn = False
        detected_rrn = None
        validation_details = {
            "pattern_match": False,
            "format_valid": False,
            "date_valid": False,
            "checksum_valid": False,
            "score": 0.0
        }

        if results:
            result = results[0]
            detected_text = text[result.start:result.end]
            detected_clean = re.sub(r'[\s\-_.]', '', ''.join(filter(str.isdigit, detected_text)))
            
            validation_details["pattern_match"] = True
            validation_details["score"] = result.score

            if detected_clean == clean_rrn:
                detected_rrn = detected_clean
                
                # 직접 검증 수행
                format_valid = recognizer._validate_format(clean_rrn)
                validation_details["format_valid"] = format_valid

                if format_valid:
                    date_valid = recognizer._validate_date(clean_rrn)
                    validation_details["date_valid"] = date_valid

                    checksum_valid = recognizer._validate_checksum(clean_rrn)
                    validation_details["checksum_valid"] = checksum_valid

                    if date_valid and checksum_valid:
                        found_valid_rrn = True

        # 결과 저장
        y_true.append(is_valid)
        y_pred.append(found_valid_rrn)

        # 상세 결과 저장
        result_detail = {
            "text": text,
            "rrn": rrn,
            "actual_valid": is_valid,
            "predicted_valid": found_valid_rrn,
            "correct_prediction": is_valid == found_valid_rrn,
            "detected_rrn": detected_rrn,
            "validation_details": validation_details
        }

        if case.get('invalid_type'):
            result_detail["error_type"] = case['invalid_type']

        detailed_results.append(result_detail)

    # 메트릭 계산
    metrics = calculate_metrics(y_true, y_pred)
    
    # 오류 분석
    error_analysis = analyze_errors(detailed_results)
    
    # 결과 출력
    print_evaluation_results(metrics, error_analysis)
    
    # 결과 저장
    save_evaluation_results(metrics, error_analysis, detailed_results, test_dir)

    return metrics, error_analysis, detailed_results

def calculate_metrics(y_true, y_pred):
    """성능 메트릭 계산"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

def analyze_errors(results):
    """오류 분석"""
    error_types = {}
    validation_errors = {
        "pattern_match": 0,
        "format": 0,
        "date": 0,
        "checksum": 0
    }
    total_errors = 0
    
    for result in results:
        if not result["correct_prediction"]:
            total_errors += 1
            error_type = result.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # 검증 단계별 오류 집계
            details = result["validation_details"]
            if not details["pattern_match"]:
                validation_errors["pattern_match"] += 1
            elif not details["format_valid"]:
                validation_errors["format"] += 1
            elif not details["date_valid"]:
                validation_errors["date"] += 1
            elif not details["checksum_valid"]:
                validation_errors["checksum"] += 1

    return {
        "total_errors": total_errors,
        "error_types": error_types,
        "validation_errors": validation_errors
    }

def print_evaluation_results(metrics, error_analysis):
    """결과 출력"""
    print("\n=== 평가 결과 ===")
    print(f"정확도 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"정밀도 (Precision): {metrics['precision']:.4f}")
    print(f"재현율 (Recall): {metrics['recall']:.4f}")
    print(f"F1 점수: {metrics['f1']:.4f}")

    print("\n=== 오류 분석 ===")
    print(f"총 오류 수: {error_analysis['total_errors']}")
    
    if error_analysis['error_types']:
        print("\n오류 유형별 분포:")
        for error_type, count in error_analysis['error_types'].items():
            print(f"- {error_type}: {count}개")
    
    print("\n검증 단계별 오류:")
    for stage, count in error_analysis['validation_errors'].items():
        print(f"- {stage}: {count}개")

def save_evaluation_results(metrics, error_analysis, detailed_results, test_dir):
    """평가 결과 저장"""
    result_filepath = os.path.join(test_dir, 'evaluation_rrn_results.json')
    
    save_results = {
        "metrics": metrics,
        "error_analysis": error_analysis,
        "detailed_results": detailed_results
    }
    
    with open(result_filepath, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n상세 평가 결과가 '{result_filepath}'에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    try:
        test_dir = "tests"
        os.makedirs(test_dir, exist_ok=True)
        evaluate_rrn_recognizer()
    except Exception as e:
        print(f"Error: 평가 중 오류가 발생했습니다 - {str(e)}")

if __name__ == "__main__":
    main()