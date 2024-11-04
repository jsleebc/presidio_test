from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from kr_phone_recognizer import KRPhoneRecognizer
import json
import os
import re
from typing import Dict, List

def evaluate_phone_recognizer(test_file: str = 'phone_test_cases_1000.json'):
    """
    전화번호 인식기 평가
    """
    test_dir = "tests"
    test_filepath = os.path.join(test_dir, test_file)

    print("테스트 케이스 로딩 중...")
    with open(test_filepath, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        test_cases = test_data['test_cases']

    print("Analyzer 초기화 중...")
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "ko", "model_name": "ko_core_news_lg"},
            {"lang_code": "en", "model_name": "en_core_web_lg"},
        ],
    }

    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["ko", "en"]
    )

    recognizer = KRPhoneRecognizer()
    analyzer.registry.add_recognizer(recognizer)

    print("평가 시작...")
    total_cases = len(test_cases)
    y_true = []
    y_pred = []
    error_logs = []
    detailed_results = []

    for i, case in enumerate(test_cases, 1):
        if i % 100 == 0:
            print(f"진행 중... {i}/{total_cases}")

        text = case['text']
        is_valid = case['is_valid']
        phone_number = case['phone_number']

        # 전화번호 정규화
        normalized_number = re.sub(r'[\s\-_+]', '', phone_number)
        if normalized_number.startswith('82'):
            normalized_number = '0' + normalized_number[2:]

        # Presidio 분석 실행
        results = analyzer.analyze(
            text=text,
            language="ko",
            entities=["KR_PHONE"]
        )

        # 분석 결과 처리
        found_valid_phone = False
        detected_phone = None
        score = 0.0

        if results:
            result = results[0]  # 첫 번째 결과 사용
            detected_text = text[result.start:result.end]
            
            # Recognizer를 통한 직접 검증
            format_valid = recognizer._validate_format(normalized_number)
            prefix_valid = recognizer._validate_prefix(normalized_number)
            length_valid = recognizer._validate_length(normalized_number)
            
            phone_valid = format_valid and prefix_valid and length_valid

            if phone_valid:
                score = result.score
                detected_phone = detected_text
                found_valid_phone = result.score >= 0.6

            # 검증 세부 정보 저장
            validation_details = {
                "format_valid": format_valid,
                "prefix_valid": prefix_valid,
                "length_valid": length_valid,
                "normalized_number": normalized_number,
                "original_number": phone_number
            }
        else:
            validation_details = {
                "format_valid": False,
                "prefix_valid": False,
                "length_valid": False,
                "normalized_number": normalized_number,
                "original_number": phone_number
            }

        # 결과 저장
        y_true.append(is_valid)
        y_pred.append(found_valid_phone)

        # 상세 결과 저장
        result_detail = {
            "text": text,
            "phone_number": phone_number,
            "normalized_number": normalized_number,
            "actual_valid": is_valid,
            "predicted_valid": found_valid_phone,
            "correct_prediction": is_valid == found_valid_phone,
            "detected_phone": detected_phone,
            "score": score
        }

        if case.get('invalid_type'):
            result_detail["type"] = case['invalid_type']

        if not is_valid == found_valid_phone:
            error_logs.append({
                "case": result_detail,
                "validation_details": validation_details,
                "error_type": case.get('invalid_type', 'unknown'),
                "text_context": {
                    "before": text[:result.start] if results else "",
                    "matched": detected_text if results else "",
                    "after": text[result.end:] if results else ""
                }
            })

        detailed_results.append(result_detail)

    # 메트릭 계산
    metrics = calculate_metrics(y_true, y_pred)
    
    # 오류 분석
    error_analysis = analyze_errors(detailed_results)
    
    # 결과 출력
    print_evaluation_results(metrics, error_analysis)

    # 결과 저장
    result_filepath = os.path.join(test_dir, 'evaluation_phone_results.json')
    save_results = {
        "metrics": metrics,
        "error_analysis": {
            "total_errors": error_analysis["total_errors"],
            "error_types": error_analysis["error_types"],
            "error_ratios": {k: float(v) for k, v in error_analysis["error_ratios"].items()}
        },
        "detailed_results": detailed_results,
        "error_logs": error_logs
    }

    with open(result_filepath, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)

    print(f"\n상세 평가 결과가 '{result_filepath}'에 저장되었습니다.")
    return metrics, error_analysis, detailed_results

def calculate_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict:
    """평가 메트릭 계산"""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0
    }

def analyze_errors(results: List[Dict]) -> Dict:
    """오류 유형별 분석"""
    error_types = {}
    total_errors = 0
    
    for result in results:
        if not result["correct_prediction"]:
            total_errors += 1
            error_type = result.get("type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1

    error_ratios = {
        error_type: count / total_errors
        for error_type, count in error_types.items()
    } if total_errors > 0 else {}

    return {
        "total_errors": total_errors,
        "error_types": error_types,
        "error_ratios": error_ratios
    }

def print_evaluation_results(metrics: Dict, error_analysis: Dict):
    """결과 출력"""
    print("\n=== 상세 평가 결과 ===")
    print(f"정확도 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"정밀도 (Precision): {metrics['precision']:.4f}")
    print(f"재현율 (Recall): {metrics['recall']:.4f}")
    print(f"F1 점수: {metrics['f1']:.4f}")
    
    print(f"True Positive: {metrics['true_positive']}")
    print(f"True Negative: {metrics['true_negative']}")
    print(f"False Positive: {metrics['false_positive']}")
    print(f"False Negative: {metrics['false_negative']}")

    print("\n=== 오류 분석 ===")
    print(f"총 오류 수: {error_analysis['total_errors']}")
    
    if error_analysis['error_types']:
        print("\n오류 유형별 분포:")
        for error_type, count in error_analysis['error_types'].items():
            ratio = error_analysis['error_ratios'][error_type]
            print(f"- {error_type}: {count}개 ({ratio:.2%})")

def main():
    try:
        test_dir = "tests"
        os.makedirs(test_dir, exist_ok=True)
        metrics, error_analysis, detailed_results = evaluate_phone_recognizer()
    except Exception as e:
        print(f"Error: 평가 중 오류가 발생했습니다 - {str(e)}")

if __name__ == "__main__":
    main()