from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import InPassportRecognizer

# 1. AnalyzerEngine 생성
analyzer = AnalyzerEngine()

# 2. 여권 인식기 등록
passport_recognizer = InPassportRecognizer()
analyzer.registry.add_recognizer(passport_recognizer)

# 3. 감지할 텍스트 정의
text = "My passport number is X12345678."

# 4. 민감 정보 감지
results = analyzer.analyze(text=text, language="en")

# 5. 결과 출력
for result in results:
    print(f"Entity: {result.entity_type}, Start: {result.start}, End: {result.end}, Score: {result.score}")
