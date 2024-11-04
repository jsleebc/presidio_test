from typing import Optional, List
from presidio_analyzer import Pattern, PatternRecognizer
import re

class KRCreditCardRecognizer(PatternRecognizer):
    """
    한국 신용카드 번호 인식기
    """
    
    PATTERNS = [
        Pattern(
            "CreditCard-Basic",
            r"(?:9[1-5]|4|5[1-5])\d{14}", # 기본 패턴
            0.6,
        ),
        Pattern(
            "CreditCard-With-Space",
            r"(?:9[1-5]|4|5[1-5])\s*\d{3,4}\s*\d{4}\s*\d{4}", # 공백 포함 패턴
            0.5,
        ),
        Pattern(
            "CreditCard-With-Separators",
            r"(?:9[1-5]|4|5[1-5])[-\s]?\d{3,4}[-\s]?\d{4}[-\s]?\d{4}", # 구분자 포함 패턴
            0.4,
        )
    ]

    CONTEXT = [
        "카드번호", "카드", "신용카드", "체크카드", "카드 번호",
        "결제카드", "카드정보", "카드 정보", "결제수단",
        "CARD", "card", "Credit Card", "Debit Card",
        "Card Number", "카드발급", "카드발급번호",
        "BC카드", "삼성카드", "현대카드", "신한카드", "국민카드",
        "롯데카드", "카드사", "VISA", "MasterCard"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "KR_CREDIT_CARD",
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
            card_number = self._extract_card_number(matched_text)
            
            if not card_number:
                result.score = 0
                return None

            # 기본 형식 검증
            if not self._validate_format(card_number):
                result.score = 0
                return None

            # 체크섬 검증
            if not self._validate_checksum(card_number):
                pattern_match.score = 0
                return None

            # 카드 발급기관 검증
            if not self._validate_issuer(card_number):
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
            card_number = self._extract_card_number(pattern)
            if not card_number:
                return False
            return (self._validate_format(card_number) and 
                   self._validate_checksum(card_number) and 
                   self._validate_issuer(card_number))
        except Exception:
            return False

    def _extract_card_number(self, text: str) -> Optional[str]:
        """텍스트에서 카드번호 추출"""
        text = re.sub(r'[\s\-_]', '', text)
        pattern = r'(?:9[1-5]|4|5[1-5])\d{14}'
        match = re.search(pattern, text)
        return match.group() if match else None

    def _validate_format(self, card_number: str) -> bool:
        """형식 검증"""
        if len(card_number) != 16:
            return False
            
        # 카드 발급기관 prefix 검증
        valid_prefixes = {
            'BC카드': ['94', '95'],
            '삼성카드': ['94'],
            '현대카드': ['95'],
            '신한카드': ['91'],
            'KB국민': ['93'],
            '롯데카드': ['92'],
            'VISA': ['4'],
            'MasterCard': ['51', '52', '53', '54', '55']
        }
        
        prefix_valid = False
        for prefixes in valid_prefixes.values():
            for prefix in prefixes:
                if card_number.startswith(prefix):
                    prefix_valid = True
                    break
            if prefix_valid:
                break
                
        if not prefix_valid:
            return False

        # 모든 문자가 숫자인지 검증
        if not card_number.isdigit():
            return False

        return True

    def _validate_checksum(self, number: str) -> bool:
        """Luhn 알고리즘을 사용한 체크섬 검증"""
        try:
            digits = [int(d) for d in number]
            for i in range(len(digits) - 2, -1, -2):
                digits[i] *= 2
                if digits[i] > 9:
                    digits[i] -= 9
            
            total = sum(digits)
            return total % 10 == 0
        except Exception:
            return False

    def _validate_issuer(self, card_number: str) -> bool:
        """카드 발급기관 검증"""
        issuer_prefixes = {
            'domestic': ['91', '92', '93', '94', '95'],  # 국내 카드사
            'visa': ['4'],  # VISA
            'mastercard': ['51', '52', '53', '54', '55']  # MasterCard
        }
        
        for prefixes in issuer_prefixes.values():
            for prefix in prefixes:
                if card_number.startswith(prefix):
                    return True
        return False