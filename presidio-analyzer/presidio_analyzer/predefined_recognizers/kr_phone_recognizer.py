from typing import Optional, List
from presidio_analyzer import Pattern, PatternRecognizer
import re

class KRPhoneRecognizer(PatternRecognizer):
    """
    한국 전화번호 인식기
    """
    
    PATTERNS = [
        Pattern(
            "Mobile-Basic",
            r"01[0-9]-\d{3,4}-\d{4}", 
            0.7
        ),
        Pattern(
            "Mobile-No-Separator",
            r"01[0-9]\d{7,8}",
            0.6
        ),
        Pattern(
            "Landline-Basic",
            r"0[2-6][1-5]-\d{3,4}-\d{4}",
            0.7
        ),
        Pattern(
            "Landline-No-Separator",
            r"0[2-6][1-5]\d{7,8}",
            0.6
        ),
        Pattern(
            "Phone-With-Spaces",
            r"(01[0-9]|0[2-6][1-5])\s*\d{3,4}\s*\d{4}",
            0.5
        ),
        Pattern(
            "Mobile-International",
            r"\+82-?10-?\d{3,4}-?\d{4}",
            0.5
        )
    ]

    CONTEXT = [
        "전화번호", "전화", "연락처", "휴대폰", "휴대전화",
        "핸드폰", "폰번호", "연락망", "연락가능", "연락바람",
        "phone", "PHONE", "전화", "통화", "통화번호",
        "Phone Number", "Contact", "Tel", "전화문의",
        "휴대폰번호", "연락주세요", "전화주세요", "문의전화",
        "대표번호", "고객센터", "상담전화", "유선번호",
        "지역번호", "발신번호", "수신번호", "팩스번호"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "KR_PHONE",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )
        
        # 유효한 지역번호/통신사번호 초기화
        self.mobile_prefixes = ['010', '011', '016', '017', '018', '019']
        self.landline_prefixes = {
            '02': '서울',
            '031': '경기', '032': '인천', '033': '강원',
            '041': '충남', '042': '대전', '043': '충북', '044': '세종',
            '051': '부산', '052': '울산', '053': '대구', '054': '경북',
            '055': '경남', '061': '전남', '062': '광주', '063': '전북',
            '064': '제주'
        }

    def validate_result(self, pattern_match):
        """검증 로직"""
        result = super().validate_result(pattern_match)
        if not result:
            return result

        try:
            # 매칭된 텍스트 추출
            matched_text = pattern_match.matched_text.strip()
            phone_number = self._extract_phone_number(matched_text)
            
            if not phone_number:
                result.score = 0
                return None

            # 전화번호 정규화 (구분자 제거)
            normalized_number = re.sub(r'[\s\-_+]', '', phone_number)
            
            # 국제번호 형식 처리
            if normalized_number.startswith('82'):
                normalized_number = '0' + normalized_number[2:]

            # 기본 형식 검증
            if not self._validate_format(normalized_number):
                result.score = 0
                return None

            # 접두사 검증
            if not self._validate_prefix(normalized_number):
                result.score = 0
                return None

            # 길이 검증
            if not self._validate_length(normalized_number):
                result.score = 0
                return None

            # 모든 검증을 통과한 경우
            pattern_match.score = 0.95
            return result

        except Exception as e:
            return None

    def validate(self, pattern: str, start: int, end: int) -> bool:
        """추가 검증 메서드"""
        try:
            phone_number = self._extract_phone_number(pattern)
            if not phone_number:
                return False
                
            normalized_number = re.sub(r'[\s\-_+]', '', phone_number)
            if normalized_number.startswith('82'):
                normalized_number = '0' + normalized_number[2:]
                
            return (self._validate_format(normalized_number) and 
                   self._validate_prefix(normalized_number) and 
                   self._validate_length(normalized_number))
        except Exception:
            return False

    def _extract_phone_number(self, text: str) -> Optional[str]:
        """전화번호 추출"""
        # 국제번호 형식 패턴
        international_pattern = r'\+82-?(?:10|[2-6][1-5])-?\d{3,4}-?\d{4}'
        # 일반 전화번호 패턴
        domestic_pattern = r'(0\d{1,2})-?(\d{3,4})-?(\d{4})'
        
        # 패턴 매칭 시도
        match = re.search(f'({international_pattern}|{domestic_pattern})', text)
        if not match:
            return None
            
        return match.group()

    def _validate_format(self, number: str) -> bool:
        """형식 검증"""
        if not number.isdigit():
            return False
            
        if len(number) < 9 or len(number) > 11:
            return False
            
        if not number.startswith('0'):
            return False
            
        return True

    def _validate_prefix(self, number: str) -> bool:
        """접두사 검증"""
        # 휴대폰 번호
        if number.startswith('01'):
            return number[:3] in self.mobile_prefixes
            
        # 서울 지역번호
        if number.startswith('02'):
            return True
            
        # 그 외 지역번호
        if number.startswith('0'):
            area_code = number[:3]
            return area_code in self.landline_prefixes
            
        return False

    def _validate_length(self, number: str) -> bool:
        """길이 검증"""
        # 서울 지역번호(02)
        if number.startswith('02'):
            return len(number) in [9, 10]  # 2(지역) + 3or4(국번) + 4(번호)
            
        # 그 외 지역번호나 휴대폰 번호
        return len(number) in [10, 11]  # 3(지역/통신사) + 3or4(국번) + 4(번호)