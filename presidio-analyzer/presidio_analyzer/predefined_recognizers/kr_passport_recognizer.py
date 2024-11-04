from typing import Optional, List
from presidio_analyzer import Pattern, PatternRecognizer
import re

class KRPassportRecognizer(PatternRecognizer):
    """
    개선된 한국 여권번호 인식기
    """
    
    PATTERNS = [
        Pattern(
            "Passport-Basic",
            r"[MR][0-9A-Z]\d{7}",  # 기본 패턴
            0.6,
        ),
        Pattern(
            "Passport-With-Space",
            r"[MR]\s*[0-9A-Z]\s*\d{1,7}",  # 공백이 있을 수 있는 패턴
            0.5,
        ),
        Pattern(
            "Passport-With-Separators",
            r"[MR][-\s]?[0-9A-Z][-\s]?\d{1,7}",  # 구분자가 있을 수 있는 패턴
            0.4,
        )
    ]
    
    CONTEXT = [
        "여권번호", "여권", "PASSPORT", "passport", "여권 번호",
        "여권발급번호", "여권 발급 번호", "여권정보", "여권 정보",
        "여권 발급", "PASSPORT NUMBER", "Passport Number",
        "Korean Passport", "대한민국 여권", "번호"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "KR_PASSPORT",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_match):
        """검증 로직"""
        result = super().validate_result(pattern_match)
        if not result:
            return result

        try:
            # 매칭된 텍스트 추출
            matched_text = pattern_match.matched_text.strip()
            passport_number = self._extract_passport_number(matched_text)
            
            if not passport_number:
                result.score = 0
                return None
                
            # 기본 형식 검증
            if not self._validate_format(passport_number):
                result.score = 0
                return None
                
            # 체크섬 검증
            if not self._validate_checksum(passport_number):
                pattern_match.score = 0
                return None
            
            # 모든 검증을 통과한 경우
            pattern_match.score = 0.95
            return result
            
        except Exception as e:
            return None
    
    def validate(self, pattern: str, start: int, end: int) -> bool:
        """추가 검증 메서드"""
        try:
            passport_number = self._extract_passport_number(pattern)
            if not passport_number:
                return False
            
            return self._validate_format(passport_number) and self._validate_checksum(passport_number)
        except Exception:
            return False

    def _extract_passport_number(self, text: str) -> Optional[str]:
        """텍스트에서 여권번호 추출"""
        text = re.sub(r'[\s\-_]', '', text)
        pattern = r'[MR][0-9A-Z]\d{7}'
        match = re.search(pattern, text)
        return match.group() if match else None

    def _validate_format(self, passport_number: str) -> bool:
        """형식 검증"""
        if len(passport_number) != 9:
            return False
            
        if passport_number[0] not in ['M', 'R']:
            return False
            
        second_char = passport_number[1]
        if not (second_char.isdigit() or (second_char.isupper() and second_char.isalpha())):
            return False
            
        if not all(c.isdigit() for c in passport_number[2:]):
            return False
            
        return True

    def _validate_checksum(self, number: str) -> bool:
        """체크섬 검증"""
        try:
            weights = [7, 3, 1, 7, 3, 1, 7, 3, 1]
            total = 0
            
            # 첫 번째 문자 처리 (M=22, R=27)
            first_char_values = {'M': 22, 'R': 27}
            first_value = first_char_values[number[0]]
            total += first_value * weights[0]
            
            # 두 번째 문자 처리
            second_char = number[1]
            if second_char.isdigit():
                second_value = int(second_char)
            else:
                second_value = ord(second_char) - ord('A') + 10
            total += second_value * weights[1]
            
            # 나머지 숫자 처리
            for i in range(2, 8):
                total += int(number[i]) * weights[i]
            
            # 체크섬 계산
            expected_checksum = (10 - (total % 10)) % 10
            actual_checksum = int(number[-1])
            
            return expected_checksum == actual_checksum
            
        except Exception:
            return False