import random
import string
import json
import os
from email_validator import validate_email, EmailNotValidError
from tld import get_tld
from tld.utils import get_tld_names
from typing import List, Dict

class KoreanEmailGenerator:
    """한국 이메일 생성기"""
    
    def __init__(self):
        # 자주 사용되는 한국 이메일 도메인
        self.common_kr_domains = [
            "naver.com", "daum.net", "kakao.com", "hanmail.net",
            "gmail.com", "nate.com", "outlook.kr"
        ]
        # TLD 라이브러리에서 유효한 TLD 목록 가져오기
        self.valid_tlds = list(get_tld_names())
        
        # 이메일 관련 템플릿
        self.templates = [
            "이메일 주소는 {email}입니다.",
            "이메일: {email}",
            "E-mail: {email}",
            "메일 주소: {email}",
            "연락처(이메일): {email}",
            "이메일로 보내주세요: {email}",
            "회신은 {email}로 부탁드립니다.",
            "답장 받을 이메일: {email}",
            "Send email to {email}",
            "Email address: {email}",
            "Contact via {email}",
            "{email}로 연락주시기 바랍니다.",
            "문의사항은 {email}로 보내주세요.",
            "메일주소는 {email}입니다.",
            "이메일 {email}로 보내주세요.",
            "Contact email: {email}",
            "Reply to: {email}",
            "Mail to: {email}",
            "Email me at {email}",
            "Send your response to {email}"
        ]
        
    def generate_email(self, invalid_ratio=0.2):
        """이메일 주소 생성"""
        if random.random() < invalid_ratio:
            return self._generate_invalid_email()
        return self._generate_valid_email()
    
    def _generate_valid_email(self):
        """유효한 이메일 주소 생성"""
        username = self._generate_username()
        domain = random.choice(self.common_kr_domains)
        
        email = f"{username}@{domain}"
        try:
            validate_email(email, check_deliverability=False)
            return email
        except EmailNotValidError:
            return self._generate_valid_email()
    
    def _generate_invalid_email(self):
        """유효하지 않은 이메일 주소 생성"""
        invalid_types = [
            self._generate_invalid_username_email,
            self._generate_invalid_domain_email,
            self._generate_invalid_format_email
        ]
        return random.choice(invalid_types)()
    
    def _generate_username(self):
        """사용자명 생성"""
        types = [
            lambda: ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(5, 10))),
            lambda: f"user{random.randint(100, 999)}",
            lambda: f"test.{random.randint(100, 999)}",
            lambda: f"korean{random.randint(100, 999)}"
        ]
        return random.choice(types)()
    
    def _generate_invalid_username_email(self):
        """잘못된 사용자명을 가진 이메일 생성"""
        invalid_usernames = [
            "!invalid@",
            "user name",
            ".username",
            "username.",
            "user..name"
        ]
        return f"{random.choice(invalid_usernames)}@{random.choice(self.common_kr_domains)}"
    
    def _generate_invalid_domain_email(self):
        """잘못된 도메인을 가진 이메일 생성"""
        invalid_domains = [
            "invalid.domain",
            f"fake-{random.choice(self.common_kr_domains)}",
            f"not-real-{random.randint(100, 999)}.com",
            "domain.invalidtld",
            f"test.{random.choice(self.valid_tlds)}x"
        ]
        return f"{self._generate_username()}@{random.choice(invalid_domains)}"
    
    def _generate_invalid_format_email(self):
        """잘못된 형식의 이메일 생성"""
        invalid_formats = [
            lambda: f"{self._generate_username()}",
            lambda: f"{self._generate_username()}@",
            lambda: f"@{random.choice(self.common_kr_domains)}",
            lambda: f"{self._generate_username()}@@{random.choice(self.common_kr_domains)}",
            lambda: f"{self._generate_username()}.@{random.choice(self.common_kr_domains)}"
        ]
        return random.choice(invalid_formats)()

    def generate_test_cases(self, num_cases: int = 1000) -> List[Dict]:
        """테스트 케이스 생성 (80% valid, 20% invalid)"""
        test_cases = []
        print(f"총 {num_cases}개의 테스트 케이스 생성 시작...")
        
        # 유효한 케이스 (80%)
        valid_cases_count = int(num_cases * 0.8)
        print(f"유효한 이메일 생성 중... (목표: {valid_cases_count}개)")
        
        for i in range(valid_cases_count):
            if i % 100 == 0:
                print(f"- {i}/{valid_cases_count} 완료")
                
            email = self._generate_valid_email()
            template = random.choice(self.templates)
            text = template.format(email=email)
            
            test_cases.append({
                "text": text,
                "email": email,
                "is_valid": True,
                "template": template
            })
        
        # 유효하지 않은 케이스 (20%)
        invalid_cases_count = num_cases - valid_cases_count
        print(f"\n유효하지 않은 이메일 생성 중... (목표: {invalid_cases_count}개)")
        
        invalid_types = ["invalid_username", "invalid_domain", "invalid_format"]
        cases_per_type = invalid_cases_count // len(invalid_types)
        
        for invalid_type in invalid_types:
            for _ in range(cases_per_type):
                if invalid_type == "invalid_username":
                    email = self._generate_invalid_username_email()
                elif invalid_type == "invalid_domain":
                    email = self._generate_invalid_domain_email()
                else:  # invalid_format
                    email = self._generate_invalid_format_email()
                    
                template = random.choice(self.templates)
                text = template.format(email=email)
                
                test_cases.append({
                    "text": text,
                    "email": email,
                    "is_valid": False,
                    "template": template,
                    "invalid_type": invalid_type
                })
        
        print("\n테스트 케이스 섞는 중...")
        random.shuffle(test_cases)
        
        # 실제 생성된 비율 확인
        actual_valid = sum(1 for case in test_cases if case["is_valid"])
        actual_invalid = len(test_cases) - actual_valid
        print(f"\n실제 생성된 비율:")
        print(f"- 유효한 케이스: {actual_valid} ({actual_valid/len(test_cases)*100:.1f}%)")
        print(f"- 유효하지 않은 케이스: {actual_invalid} ({actual_invalid/len(test_cases)*100:.1f}%)")
        
        return test_cases

    def save_to_json(self, test_cases: List[Dict], filename: str):
        """테스트 케이스를 JSON 파일로 저장"""
        test_dir = "tests"
        os.makedirs(test_dir, exist_ok=True)
        
        filepath = os.path.join(test_dir, filename)
        print(f"\n{filepath}에 저장 중...")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_cases": len(test_cases),
                    "valid_cases": sum(1 for case in test_cases if case["is_valid"]),
                    "invalid_cases": sum(1 for case in test_cases if not case["is_valid"]),
                    "template_count": len(self.templates)
                },
                "templates": self.templates,
                "test_cases": test_cases
            }, f, ensure_ascii=False, indent=2)


def main():
    # 테스트 케이스 생성기 초기화
    generator = KoreanEmailGenerator()
    
    # 1000개의 테스트 케이스 생성
    test_cases = generator.generate_test_cases(1000)
    
    # JSON 파일로 저장
    generator.save_to_json(test_cases, 'email_test_cases_1000.json')
    
    # 통계 출력
    valid_count = sum(1 for case in test_cases if case["is_valid"])
    invalid_count = len(test_cases) - valid_count
    
    print("\n=== 생성 완료 ===")
    print(f"총 테스트 케이스: {len(test_cases)}")
    print(f"유효한 케이스: {valid_count}")
    print(f"유효하지 않은 케이스: {invalid_count}")
    
    # 샘플 케이스 출력
    print("\n=== 생성된 테스트 케이스 샘플 (처음 5개) ===")
    for case in test_cases[:5]:
        print(f"\n텍스트: {case['text']}")
        print(f"이메일: {case['email']}")
        print(f"유효성: {'유효함' if case['is_valid'] else '유효하지 않음'}")
        if not case['is_valid']:
            print(f"오류 유형: {case.get('invalid_type', 'unknown')}")
        print("-" * 50)


if __name__ == "__main__":
    main()