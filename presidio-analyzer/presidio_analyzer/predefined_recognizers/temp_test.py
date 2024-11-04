from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from kr_rrn_recognizer import KRRRNRecognizer
import json
import os
import re

def analyze_false_positives():
    """오탐(False Positive) 케이스 상세 분석"""
    
    # 평가 결과 파일 로드
    test_dir = "tests"
    result_filepath = os.path.join(test_dir, 'evaluation_rrn_results.json')
    
    print("평가 결과 파일 로딩 중...")
    with open(result_filepath, 'r', encoding='utf-8') as f:
        evaluation_results = json.load(f)
        detailed_results = evaluation_results['detailed_results']

    # Recognizer 초기화
    recognizer = KRRRNRecognizer()
    
    # 오탐 케이스 필터링
    false_positives = [
        result for result in detailed_results 
        if result['predicted_valid'] and not result['actual_valid']
    ]
    
    print(f"\n=== 오탐 분석 결과 ===")
    print(f"총 오탐 케이스 수: {len(false_positives)}")
    
    # 오류 유형별 분류
    error_types = {}
    for case in false_positives:
        error_type = case.get('error_type', 'unknown')
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(case)
    
    # 각 오류 유형별 상세 분석
    print("\n=== 오류 유형별 상세 분석 ===")
    
    for error_type, cases in error_types.items():
        print(f"\n## {error_type} 유형 (총 {len(cases)}건)")
        
        for i, case in enumerate(cases, 1):
            rrn = case['rrn']
            clean_rrn = re.sub(r'[\s\-_.]', '', rrn)
            
            print(f"\n케이스 {i}:")
            print(f"입력 텍스트: {case['text']}")
            print(f"주민등록번호: {rrn}")
            
            # 상세 검증 결과
            print("\n검증 단계별 결과:")
            
            # 1. 형식 검증
            format_valid = recognizer._validate_format(clean_rrn)
            print(f"1. 형식 검증: {'통과' if format_valid else '실패'}")
            
            if format_valid:
                # 2. 날짜 검증
                date_valid = recognizer._validate_date(clean_rrn)
                print(f"2. 날짜 검증: {'통과' if date_valid else '실패'}")
                
                if date_valid:
                    # 연도/성별 코드 분석
                    year = int(clean_rrn[:2])
                    gender_code = int(clean_rrn[6])
                    
                    if gender_code in [1, 2, 5, 6]:
                        year += 1900
                    else:  # 3, 4, 7, 8
                        year += 2000
                        
                    print(f"   - 추정 출생년도: {year}")
                    print(f"   - 성별 코드: {gender_code}")
                
                # 3. 체크섬 검증
                print("\n3. 체크섬 계산 과정:")
                multipliers = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
                total = 0
                
                for j in range(12):
                    value = int(clean_rrn[j])
                    subtotal = value * multipliers[j]
                    total += subtotal
                    print(f"   위치 {j}: {value} × {multipliers[j]} = {subtotal}")
                
                expected_checksum = (11 - (total % 11)) % 10
                actual_checksum = int(clean_rrn[-1])
                
                print(f"\n   총합: {total}")
                print(f"   예상 체크섬: {expected_checksum}")
                print(f"   실제 체크섬: {actual_checksum}")
                print(f"   체크섬 검증: {'통과' if expected_checksum == actual_checksum else '실패'}")
            
            # 오류 원인 분석
            print("\n추정 오류 원인:")
            if error_type == "invalid_date":
                print("- 유효하지 않은 날짜 형식")
                print(f"- 월: {clean_rrn[2:4]}")
                print(f"- 일: {clean_rrn[4:6]}")
            elif error_type == "invalid_gender_year":
                print("- 연도와 성별코드 불일치")
                print(f"- 연도: {year}")
                print(f"- 성별코드: {gender_code}")
            
            print("\n" + "="*70)
    
    # 종합 분석 결과
    print("\n=== 종합 분석 ===")
    print("오류 유형별 발생 빈도:")
    for error_type, cases in error_types.items():
        print(f"- {error_type}: {len(cases)}건 ({len(cases)/len(false_positives)*100:.1f}%)")
    
    # 결과 저장
    analysis_results = {
        "total_false_positives": len(false_positives),
        "error_types": {
            error_type: {
                "count": len(cases),
                "percentage": len(cases)/len(false_positives)*100,
                "cases": cases
            }
            for error_type, cases in error_types.items()
        }
    }
    
    analysis_filepath = os.path.join(test_dir, 'rrn_false_positives_analysis.json')
    with open(analysis_filepath, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n상세 분석 결과가 '{analysis_filepath}'에 저장되었습니다.")

if __name__ == "__main__":
    analyze_false_positives()