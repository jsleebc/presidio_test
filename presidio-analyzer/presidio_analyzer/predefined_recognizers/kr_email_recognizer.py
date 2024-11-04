from typing import Optional, List
from presidio_analyzer import Pattern, PatternRecognizer
import re
from email_validator import validate_email, EmailNotValidError
from tld import get_tld
from tld.utils import get_tld_names

class KREmailRecognizer(PatternRecognizer):
    """한국 이메일 인식기"""
    
    PATTERNS = [
        Pattern(
            "Email-Basic",
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # 기본 이메일 패턴
            0.6,
        ),
        Pattern(
            "Email-With-Korean-Domain",
            r"[a-zA-Z0-9._%+-]+@(?:naver\.com|daum\.net|kakao\.com|gmail\.com|nate\.com|hanmail\.net)",  # 한국 주요 도메인
            0.7,
        ),
        Pattern(
            "Email-With-Context",
            r"(?:이메일|메일|[Ee]mail|[Mm]ail)(?:\s*[:：]?\s*)((?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}))",  # 컨텍스트 포함
            0.5,
        ),
        Pattern(
            "Email-Flexible",
            r"[a-zA-Z0-9][a-zA-Z0-9._%+-]{0,63}@(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}",  # 유연한 패턴
            0.4,
        )
    ]
    
    CONTEXT = [
        "이메일", "메일", "이메일주소", "이메일 주소", "메일주소",
        "email", "mail", "e-mail", "email address", "mail address",
        "보내는곳", "받는곳", "수신", "발신", "연락처", "contact",
        "이메일로", "메일로", "이메일:", "메일:", "Email:", "Mail:"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "ko",
        supported_entity: str = "EMAIL_ADDRESS",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )
        
        # 한국 주요 도메인 초기화
        self.common_kr_domains = {
            "naver.com", "daum.net", "kakao.com", "hanmail.net",
            "gmail.com", "nate.com", "outlook.kr"
        }

    def validate_result(self, pattern_match):
        """검증 로직"""
        result = super().validate_result(pattern_match)
        if not result:
            return result

        try:
            # 매칭된 텍스트 추출
            matched_text = pattern_match.matched_text.strip()
            email = self._extract_email(matched_text)
            
            if not email:
                result.score = 0
                return None

            # 이메일 기본 형식 검증
            if not self._validate_format(email):
                result.score = 0
                return None

            # 도메인 검증
            if not self._validate_domain(email):
                result.score = 0.5  # 유효하지만 일반적이지 않은 도메인
                return result

            # email-validator를 통한 상세 검증
            if not self._validate_with_library(email):
                result.score = 0
                return None

            # 한국 주요 도메인인 경우 점수 상향
            if self._is_korean_domain(email):
                result.score = 0.95
            else:
                result.score = 0.85

            return result

        except Exception:
            return None

    def _extract_email(self, text: str) -> Optional[str]:
        """텍스트에서 이메일 추출"""
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(pattern, text)
        return match.group() if match else None

    def _validate_format(self, email: str) -> bool:
        """기본 형식 검증"""
        pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._%+-]{0,63}@(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False
            
        # 추가 형식 검증
        if '..' in email:  # 연속된 점 불가
            return False
        if email.count('@') != 1:  # @ 기호는 정확히 하나
            return False
        if email.startswith('.') or email.endswith('.'):  # 시작이나 끝에 점 불가
            return False
            
        return True

    def _validate_domain(self, email: str) -> bool:
        """도메인 검증"""
        try:
            domain = email.split('@')[1]
            return bool(get_tld(f"http://{domain}", fail_silently=True))
        except Exception:
            return False

    def _validate_with_library(self, email: str) -> bool:
        """email-validator 라이브러리를 통한 검증"""
        try:
            validate_email(email, check_deliverability=False)
            return True
        except EmailNotValidError:
            return False

    def _is_korean_domain(self, email: str) -> bool:
        """한국 주요 도메인 확인"""
        try:
            domain = email.split('@')[1].lower()
            return domain in self.common_kr_domains
        except Exception:
            return False

    def validate(self, pattern: str, start: int, end: int) -> bool:
        """추가 검증 메서드"""
        try:
            email = self._extract_email(pattern)
            if not email:
                return False
                
            return (
                self._validate_format(email) and
                self._validate_domain(email) and
                self._validate_with_library(email)
            )
        except Exception:
            return False