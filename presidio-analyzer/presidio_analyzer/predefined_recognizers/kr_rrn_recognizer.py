from typing import Optional, List
from presidio_analyzer import Pattern, PatternRecognizer
import re
from datetime import datetime

class KRRRNRecognizer(PatternRecognizer):
    """한국 주민등록번호 인식기"""
    
    PATTERNS = [
        Pattern(
            "RRN-Full",
            r"(?:주민\s*등록\s*번호|주민\s*번호)[\s:\(\{\[\-]*(\d{6}[\s\-]?\d{7})",
            0.6,
        ),
        Pattern(
            "RRN-Context",
            r"(?:RRN|RRN\s*NUMBER|Korean\s*RRN)[\s:\(\{\[\-]*(\d{6}[\s\-]?\d{7})",
            0.6,
        ),
        Pattern(
            "RRN-Brackets",
            r"[\(\{\[]\d{6}[\s\-]?\d{7}[\)\}\]]",
            0.6,
        ),
        Pattern(
            "RRN-Simple",
            r"\d{6}[\s\-]?\d{7}",
            0.5,
        )
    ]

    CONTEXT = [
        "주민", "주민등록", "주민등록번호", "RRN", 
        "resident registration", "주민번호"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "KR_RRN",
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
        result = super().validate_result(pattern_match)
        if not result:
            return result

        try:
            rrn = ''.join(filter(str.isdigit, pattern_match.matched_text))
            
            if not self._validate_format(rrn):
                return None
                
            if not self._validate_date(rrn):
                return None
                
            if not self._validate_gender_year(rrn):
                return None
                
            if not self._validate_checksum(rrn):
                return None

            return result
        except Exception:
            return None

    def _validate_format(self, rrn: str) -> bool:
        """형식 검증"""
        if len(rrn) != 13 or not rrn.isdigit():
            return False

        gender_code = int(rrn[6])
        if gender_code < 1 or gender_code > 8:  # 9,0은 더이상 사용되지 않음
            return False

        region_code = int(rrn[7:9])
        if region_code < 0 or region_code > 95:
            return False

        return True

    def _validate_date(self, rrn: str) -> bool:
        """날짜 검증"""
        try:
            year = int(rrn[:2])
            month = int(rrn[2:4])
            day = int(rrn[4:6])
            gender_code = int(rrn[6])

            # 연도 계산
            if gender_code in [1, 2, 5, 6]:
                year += 1900
            else:  # 3, 4, 7, 8
                year += 2000

            # 기본 유효성 검사
            if not (1 <= month <= 12):
                return False

            # 월별 마지막 날짜
            days_in_month = {
                1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
            }

            # 윤년 처리
            if month == 2 and ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0):
                days_in_month[2] = 29

            if day < 1 or day > days_in_month[month]:
                return False

            # 미래 날짜 검증
            birth_date = datetime(year, month, day)
            if birth_date > datetime.now():
                return False

            return True
        except ValueError:
            return False

    def _validate_gender_year(self, rrn: str) -> bool:
        """성별/연도 조합 검증"""
        try:
            year = int(rrn[:2])
            gender_code = int(rrn[6])
            current_year = datetime.now().year

            # 2000년대 출생자 (00~24)
            if year <= int(str(current_year)[2:]):
                if gender_code not in [3, 4, 7, 8]:  # 2000년대 성별코드
                    return False
            # 1900년대 출생자
            else:
                if gender_code not in [1, 2, 5, 6]:  # 1900년대 성별코드
                    return False

            return True
        except ValueError:
            return False

    def _validate_checksum(self, rrn: str) -> bool:
        """체크섬 검증"""
        try:
            multipliers = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
            checksum = sum(int(rrn[i]) * multipliers[i] for i in range(12))
            expected_checksum = (11 - (checksum % 11)) % 10
            return expected_checksum == int(rrn[-1])
        except Exception:
            return False