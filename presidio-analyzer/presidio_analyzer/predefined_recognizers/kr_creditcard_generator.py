import random
import string
import json
import os
from typing import List, Dict

class KoreanCreditCardNumberGenerator:
    """한국 신용카드 번호 생성기"""
    
    def __init__(self):
        self.card_types = {
            'Visa': {'prefix': ['4'], 'length': [16]},
            'MasterCard': {'prefix': ['51', '52', '53', '54', '55'], 'length': [16]},
            'BC카드': {'prefix': ['94', '95'], 'length': [16]},
            '삼성카드': {'prefix': ['94'], 'length': [16]},
            '현대카드': {'prefix': ['95'], 'length': [16]},
            '롯데카드': {'prefix': ['92'], 'length': [16]},
            '신한카드': {'prefix': ['91'], 'length': [16]},
            'KB국민': {'prefix': ['93'], 'length': [16]}
        }

    def _generate_valid_card_number(self):
        """유효한 카드번호 생성"""
        card_type = random.choice(list(self.card_types.keys()))
        prefix = random.choice(self.card_types[card_type]['prefix'])
        length = self.card_types[card_type]['length'][0]
        
        number = prefix
        number += ''.join([str(random.randint(0, 9)) for _ in range(length - len(prefix) - 1)])
        checksum = self._calculate_luhn_checksum(number)
        return number + str(checksum)

    def _generate_invalid_checksum_number(self):
        """잘못된 체크섬을 가진 카드번호 생성"""
        valid_number = self._generate_valid_card_number()
        invalid_checksum = (int(valid_number[-1]) + 1) % 10
        return valid_number[:-1] + str(invalid_checksum)

    def _generate_invalid_length_number(self):
        """잘못된 길이의 카드번호 생성"""
        valid_number = self._generate_valid_card_number()
        if random.choice([True, False]):
            return valid_number[:-1]  # 한 자리 부족
        else:
            return valid_number + str(random.randint(0, 9))  # 한 자리 추가

    def _generate_invalid_prefix_number(self):
        """잘못된 접두사를 가진 카드번호 생성"""
        valid_number = self._generate_valid_card_number()
        invalid_prefix = str(random.randint(10, 90))
        return invalid_prefix + valid_number[2:]

    def _generate_invalid_chars_number(self):
        """잘못된 문자를 포함한 카드번호 생성"""
        valid_number = self._generate_valid_card_number()
        position = random.randint(0, len(valid_number) - 1)
        invalid_char = random.choice(string.ascii_letters)
        return valid_number[:position] + invalid_char + valid_number[position+1:]

    def _calculate_luhn_checksum(self, number: str) -> int:
        """Luhn 알고리즘을 사용한 체크섬 계산"""
        digits = [int(d) for d in number]
        for i in range(len(digits) - 1, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        total = sum(digits)
        return (10 - (total % 10)) % 10

class CreditCardTestGenerator:
    """신용카드 번호 테스트 케이스 생성기"""
    
    def __init__(self):
        self.card_generator = KoreanCreditCardNumberGenerator()
        self.templates = [
            "카드번호는 {number}입니다.",
            "신용카드: {number}",
            "CARD NUMBER: {number}",
            "카드 번호: {number}",
            "결제카드 번호는 {number}입니다.",
            "해당 고객의 카드번호는 {number}로 확인됩니다.",
            "카드정보: {number}",
            "Credit Card: {number}",
            "카드번호 {number}로 결제되었습니다.",
            "카드번호({number})로 조회해주세요.",
            "고객님의 카드번호는 {number}입니다.",
            "결제수단: {number}",
            "Payment method: {number}",
            "카드번호 확인: {number}",
            "등록된 카드: {number}",
            "결제카드: {number}",
            "카드번호 입력: {number}",
            "카드 일련번호: {number}",
            "Card No: {number}",
            "결제정보: {number}"
        ]
        
        self.test_dir = "tests"
        os.makedirs(self.test_dir, exist_ok=True)

    def generate_test_cases(self, num_cases: int = 1000) -> List[Dict]:
        """테스트 케이스 생성 (80% valid, 20% invalid)"""
        test_cases = []
        print(f"총 {num_cases}개의 테스트 케이스 생성 시작...")

        # 유효한 케이스 (80%)
        valid_cases_count = int(num_cases * 0.8)
        print(f"유효한 카드번호 생성 중... (목표: {valid_cases_count}개)")
        
        for i in range(valid_cases_count):
            if i % 100 == 0:
                print(f"- {i}/{valid_cases_count} 완료")
            
            card_number = self.card_generator._generate_valid_card_number()
            template = random.choice(self.templates)
            text = template.format(number=card_number)
            
            test_cases.append({
                "text": text,
                "card_number": card_number,
                "is_valid": True,
                "template": template
            })

        # 유효하지 않은 케이스 (20%)
        invalid_generators = [
            (self.card_generator._generate_invalid_checksum_number, "invalid_checksum"),
            (self.card_generator._generate_invalid_length_number, "invalid_length"),
            (self.card_generator._generate_invalid_prefix_number, "invalid_prefix"),
            (self.card_generator._generate_invalid_chars_number, "invalid_chars")
        ]

        invalid_cases_count = num_cases - valid_cases_count
        cases_per_type = invalid_cases_count // len(invalid_generators)
        
        for generator, invalid_type in invalid_generators:
            for _ in range(cases_per_type):
                card_number = generator()
                template = random.choice(self.templates)
                text = template.format(number=card_number)
                
                test_cases.append({
                    "text": text,
                    "card_number": card_number,
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
    test_generator = CreditCardTestGenerator()
    test_cases = test_generator.generate_test_cases(1000)
    test_generator.save_to_json(test_cases, 'creditcard_test_cases_1000.json')

if __name__ == "__main__":
    main()