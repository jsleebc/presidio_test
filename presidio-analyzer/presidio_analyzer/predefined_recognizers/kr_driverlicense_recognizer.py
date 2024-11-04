from typing import Optional, List
from presidio_analyzer import Pattern, PatternRecognizer
import re

class KRDriverLicenseRecognizer(PatternRecognizer):
    """한국 운전면허번호 인식기"""
    
    PATTERNS = [
        Pattern(
            "DriverLicense-Basic",
            r"\d{2}-\d{2}-\d{6}-\d{2}",  # 기본 패턴 (11-22-333333-44)
            0.7,
        ),
        Pattern(
            "DriverLicense-NoSeparator",
            r"\d{2}\d{2}\d{6}\d{2}",  # 구분자 없는 패턴
            0.6,
        ),
        Pattern(
            "DriverLicense-Flexible",
            r"\d{2}[-\s_.]\d{2}[-\s_.]\d{6}[-\s_.]\d{2}",  # 유연한 구분자
            0.5,
        ),
        Pattern(
            "DriverLicense-WithText",
            r"(?:운전면허|면허)(?:증)?(?:\s)?(?:번호)?(?:\s)?:?\s*(?:\d{2}[-\s_.]\d{2}[-\s_.]\d{6}[-\s_.]\d{2}|\d{12})",
            0.4,
        )
    ]
    
    CONTEXT = [
        "운전면허번호", "면허번호", "운전면허", "면허증번호",
        "운전면허증", "면허증", "운전면허증번호", "운전면허 번호",
        "면허", "운전", "Driver License", "License Number",
        "Driver's License", "License", "면허정보", "운전면허정보"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "KR_DRIVER_LICENSE",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )
        
        # 유효한 지역코드와 발급년도 초기화
        self.valid_regions = set(str(i).zfill(2) for i in range(11, 29))  # 11-28
        self.valid_years = set(str(i).zfill(2) for i in range(100))  # 00-99

    def validate_result(self, pattern_match):
        """검증 로직"""
        result = super().validate_result(pattern_match)
        if not result:
            return result

        try:
            # 매칭된 텍스트에서 면허번호 추출
            matched_text = pattern_match.matched_text.strip()
            license_number = self._extract_license_number(matched_text)
            
            if not license_number:
                result.score = 0
                return None
                
            # 기본 형식 검증
            if not self._validate_format(license_number):
                result.score = 0
                return None

            # 개별 컴포넌트 검증
            components = license_number.split('-')
            if len(components) != 4:
                result.score = 0
                return None

            region, year, serial, check = components
            
            # 지역코드 검증
            if region not in self.valid_regions:
                result.score = 0
                return None
                
            # 발급년도 검증
            if year not in self.valid_years:
                result.score = 0
                return None
                
            # 일련번호 검증
            if not (len(serial) == 6 and serial.isdigit()):
                result.score = 0
                return None
                
            # 확인번호 검증
            if not (len(check) == 2 and check.isdigit()):
                result.score = 0
                return None

            # 모든 검증 통과
            pattern_match.score = 0.95
            return result

        except Exception:
            return None

    def validate(self, pattern: str, start: int, end: int) -> bool:
        """추가 검증 메서드"""
        try:
            license_number = self._extract_license_number(pattern)
            if not license_number:
                return False
            
            # 기본 형식 검증
            if not self._validate_format(license_number):
                return False

            # 개별 컴포넌트 검증
            components = license_number.split('-')
            if len(components) != 4:
                return False

            region, year, serial, check = components
            
            return (
                region in self.valid_regions and
                year in self.valid_years and
                len(serial) == 6 and serial.isdigit() and
                len(check) == 2 and check.isdigit()
            )
            
        except Exception:
            return False

    def _extract_license_number(self, text: str) -> Optional[str]:
        """운전면허번호 추출 및 정규화"""
        # 불필요한 문자 제거
        text = re.sub(r'[^\d\-\s_.]', '', text)
        
        # 구분자로 분리된 패턴 찾기
        pattern1 = r'(\d{2})[-\s_.]?(\d{2})[-\s_.]?(\d{6})[-\s_.]?(\d{2})'
        match = re.search(pattern1, text)
        
        if not match:
            # 연속된 숫자 패턴 찾기
            pattern2 = r'(\d{2})(\d{2})(\d{6})(\d{2})'
            match = re.search(pattern2, text)
        
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}-{match.group(4)}"
            
        return None

    def _validate_format(self, license_number: str) -> bool:
        """기본 형식 검증"""
        pattern = r'^\d{2}-\d{2}-\d{6}-\d{2}$'
        return bool(re.match(pattern, license_number))