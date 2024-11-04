# Presidio Test Project

이 프로젝트는 Presidio를 활용한 다양한 개인정보 인식기의 성능을 테스트하고 평가하는 도구입니다.

## 설치 방법
### 1. Anaconda 설치
1. [Anaconda 공식 웹사이트](https://www.anaconda.com/products/distribution)

### 2. 가상환경 생성 및 활성화

# 가상환경 생성 (Python 3.8 기준)
conda create -n presidio python=3.9.15

# 가상환경 활성화
conda activate presidio
```

### 3. 필수 라이브러리 설치
# requirements.txt 설치
pip install -r requirements.txt
```

### 1. 개별 테스트 실행
각각의 인식기 테스트를 개별적으로 실행할 수 있습니다:
```bash
# 신용카드 번호 인식기 테스트
python kr_test_creditcard.py

# 이메일 인식기 테스트
python kr_test_email.py

# 운전면허번호 인식기 테스트
python kr_test_driverlicense.py
```

### 2. 전체 테스트 실행 및 결과 종합
모든 테스트를 한 번에 실행하고 결과를 종합하여 볼 수 있습니다:
```bash
python run_all_evaluations.py
```

## 결과 확인

- 각 테스트의 결과는 `tests` 디렉토리 아래에 JSON 파일로 저장됩니다.
- 전체 평가 결과는 `evaluation_summary.json` 파일에서 확인할 수 있습니다.
- 결과에는 다음과 같은 메트릭이 포함됩니다:
  - Accuracy (정확도)
  - Precision (정밀도)
  - Recall (재현율)
  - F1 Score