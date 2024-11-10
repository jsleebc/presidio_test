from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from kr_creditcard_recognizer import KRCreditCardRecognizer
import json
import os

def evaluate_creditcard_recognizer(test_file: str = 'creditcard_test_cases_1000.json'):
    """
    신용카드 번호 인식기 평가
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

    recognizer = KRCreditCardRecognizer()
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
        card_number = case['card_number']

        # Presidio 분석 실행
        results = analyzer.analyze(
            text=text,
            language="ko",
            entities=["KR_CREDIT_CARD"]
        )

        # 분석 결과 처리

        found_valid_card = False
        detected_card = None
        score = 0.0

        if results:
            result = results[0]  # 첫 번째 결과 사용
            detected_text = text[result.start:result.end]
            
            # 오탐 여부 확인을 위한 검증
            format_valid = recognizer._validate_format(card_number)
            checksum_valid = recognizer._validate_checksum(card_number)
            issuer_valid = recognizer._validate_issuer(card_number)
            
            score = result.score
            detected_card = detected_text
            
            # 실제로는 유효하지 않은 카드번호인데 유효하다고 판단한 경우를 오탐으로 체크
            if not is_valid and score >= 0.6 and (format_valid or checksum_valid or issuer_valid):
                found_valid_card = True
            # 실제로 유효한 카드번호이고 정상적으로 감지한 경우
            elif is_valid and score >= 0.6 and format_valid and checksum_valid and issuer_valid:
                found_valid_card = True

        # 결과 저장
        y_true.append(is_valid)
        y_pred.append(found_valid_card)

        # 상세 결과 저장
        result_detail = {
            "text": text,
            "card_number": card_number,
            "actual_valid": is_valid,
            "predicted_valid": found_valid_card,
            "correct_prediction": is_valid == found_valid_card,
            "detected_card": detected_card,
            "score": score,
            "validation_details": {
                "format_valid": format_valid if results else False,
                "checksum_valid": checksum_valid if results else False,
                "issuer_valid": issuer_valid if results else False
            }
        }

        detailed_results.append(result_detail)

    # 메트릭 계산
    metrics = calculate_metrics(y_true, y_pred)
    
    # 오류 분석
    error_analysis = analyze_errors(detailed_results)
    
    # 결과 출력
    print_evaluation_results(metrics, error_analysis)

    # 결과 저장
    result_filepath = os.path.join(test_dir, 'evaluation_creditcard_results.json')
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

def calculate_metrics(y_true, y_pred):
    """평가 메트릭 계산"""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

def analyze_errors(results):
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
            ratio = error_analysis['error_ratios'][error_type]
            print(f"- {error_type}: {count}개 ({ratio:.2%})")

def main():
    try:
        test_dir = "tests"
        os.makedirs(test_dir, exist_ok=True)
        metrics, error_analysis, detailed_results = evaluate_creditcard_recognizer()
    except Exception as e:
        print(f"Error: 평가 중 오류가 발생했습니다 - {str(e)}")

if __name__ == "__main__":
    main()