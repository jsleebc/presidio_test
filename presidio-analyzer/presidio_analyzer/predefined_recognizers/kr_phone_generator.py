import random
import string
import json
from typing import List, Dict
import os

class KoreanPhoneNumberGenerator:
    """한국 전화번호 생성기"""
    
    def __init__(self):
        self.mobile_prefixes = ['010', '011', '016', '017', '018', '019']
        self.landline_prefixes = {
            '02': '서울', '031': '경기', '032': '인천', '033': '강원',
            '041': '충남', '042': '대전', '043': '충북', '044': '세종',
            '051': '부산', '052': '울산', '053': '대구', '054': '경북',
            '055': '경남', '061': '전남', '062': '광주', '063': '전북',
            '064': '제주'
        }

    def _generate_valid_number(self):
        """유효한 전화번호 생성"""
        if random.choice([True, False]):  # 휴대폰
            prefix = random.choice(self.mobile_prefixes)
            middle = f"{random.randint(0, 9999):04d}"
            last = f"{random.randint(0, 9999):04d}"
            return f"{prefix}-{middle}-{last}"
        else:  # 유선전화
            prefix = random.choice(list(self.landline_prefixes.keys()))
            middle = f"{random.randint(0, 999):03d}" if prefix != '02' else f"{random.randint(0, 9999):04d}"
            last = f"{random.randint(0, 9999):04d}"
            return f"{prefix}-{middle}-{last}"

    def _generate_invalid_prefix_number(self):
        """잘못된 접두사를 가진 전화번호 생성"""
        invalid_prefix = f"0{random.randint(20, 99)}"
        middle = f"{random.randint(0, 9999):04d}"
        last = f"{random.randint(0, 9999):04d}"
        return f"{invalid_prefix}-{middle}-{last}"

    def _generate_invalid_length_number(self):
        """잘못된 길이의 전화번호 생성"""
        prefix = random.choice(self.mobile_prefixes + list(self.landline_prefixes.keys()))
        middle = f"{random.randint(0, 99999):05d}"
        last = f"{random.randint(0, 999):03d}"
        return f"{prefix}-{middle}-{last}"

    def _generate_invalid_format_number(self):
        """잘못된 형식의 전화번호 생성"""
        return self._generate_valid_number().replace('-', '')

    def _generate_invalid_chars_number(self):
        """잘못된 문자가 포함된 전화번호 생성"""
        number = self._generate_valid_number()
        position = random.randint(0, len(number) - 1)
        invalid_char = random.choice(string.ascii_letters)
        return number[:position] + invalid_char + number[position+1:]

class PhoneTestGenerator:
    """전화번호 테스트 케이스 생성기"""
    
    def __init__(self):
        self.phone_generator = KoreanPhoneNumberGenerator()
        self.templates = [
            "전화번호는 {number}입니다.",
            "연락처: {number}",
            "PHONE NUMBER: {number}",
            "전화: {number}",
            "연락처는 {number}입니다.",
            "해당 고객의 전화번호는 {number}로 확인됩니다.",
            "전화정보: {number}",
            "Phone: {number}",
            "전화번호 {number}로 연락주세요.",
            "연락처({number})로 문의해주세요.",
            "고객님의 전화번호는 {number}입니다.",
            "연락받으실 번호: {number}",
            "Contact Number: {number}",
            "전화번호 확인: {number}",
            "등록된 연락처: {number}",
            "통화가능 번호: {number}",
            "전화번호 입력: {number}",
            "연락 가능 번호: {number}",
            "Tel: {number}",
            "연락정보: {number}"
        ]
        
        self.test_dir = "tests"
        os.makedirs(self.test_dir, exist_ok=True)

    def generate_test_cases(self, num_cases: int = 1000) -> List[Dict]:
        """테스트 케이스 생성 (80% valid, 20% invalid)"""
        test_cases = []
        print(f"총 {num_cases}개의 테스트 케이스 생성 시작...")

        # 유효한 케이스 (80%)
        valid_cases_count = int(num_cases * 0.8)
        print(f"유효한 전화번호 생성 중... (목표: {valid_cases_count}개)")
        
        for i in range(valid_cases_count):
            if i % 100 == 0:
                print(f"- {i}/{valid_cases_count} 완료")
            
            phone_number = self.phone_generator._generate_valid_number()
            template = random.choice(self.templates)
            text = template.format(number=phone_number)
            
            test_cases.append({
                "text": text,
                "phone_number": phone_number,
                "is_valid": True,
                "template": template
            })

        # 유효하지 않은 케이스 (20%)
        invalid_generators = [
            (self.phone_generator._generate_invalid_prefix_number, "invalid_prefix"),
            (self.phone_generator._generate_invalid_length_number, "invalid_length"),
            (self.phone_generator._generate_invalid_format_number, "invalid_format"),
            (self.phone_generator._generate_invalid_chars_number, "invalid_chars")
        ]

        invalid_cases_count = num_cases - valid_cases_count
        cases_per_type = invalid_cases_count // len(invalid_generators)
        
        for generator, invalid_type in invalid_generators:
            for _ in range(cases_per_type):
                phone_number = generator()
                template = random.choice(self.templates)
                text = template.format(number=phone_number)
                
                test_cases.append({
                    "text": text,
                    "phone_number": phone_number,
                    "is_valid": False,
                    "template": template,
                    "invalid_type": invalid_type
                })

        random.shuffle(test_cases)
        return test_cases

    def save_to_json(self, test_cases: List[Dict], filename: str):
        """테스트 케이스를 JSON 파일로 저장"""
        filepath = os.path.join(self.test_dir, filename)
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
    test_generator = PhoneTestGenerator()
    test_cases = test_generator.generate_test_cases(1000)
    test_generator.save_to_json(test_cases, 'phone_test_cases_1000.json')

if __name__ == "__main__":
    main()