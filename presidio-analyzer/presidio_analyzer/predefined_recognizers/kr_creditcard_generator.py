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

    def _generate_valid_looking_number(self):
        """유효해 보이지만 실제로는 유효하지 않은 번호 생성"""
        # 실제 카드 prefix와 비슷하지만 다른 번호 생성
        similar_prefixes = ['90', '96', '97', '98', '99', '50', '56', '57', '58', '59']
        prefix = random.choice(similar_prefixes)
        rest = ''.join([str(random.randint(0, 9)) for _ in range(14)])
        return prefix + rest

    def _generate_pattern_number(self):
        """패턴이 있는 번호 생성 (연속된 숫자나 반복된 숫자)"""
        patterns = [
            ''.join(str(i) for i in range(4)) * 4,  # 0123012301230123
            ''.join(str(i) * 4 for i in range(4)),  # 0000111122223333
            str(random.randint(0, 9)) * 16,         # 같은 숫자 반복
        ]
        return random.choice(patterns)

    def _generate_similar_format_number(self):
        """비슷한 형식의 다른 종류의 번호 생성"""
        # 전화번호나 계좌번호 형식
        formats = [
            f"010{random.randint(10000000, 99999999)}",  # 휴대폰 번호
            f"02{random.randint(10000000, 99999999)}",   # 서울 전화번호
            f"1{random.randint(1000000000000000, 9999999999999999)}"  # 계좌번호 형식
        ]
        return random.choice(formats)

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
        """테스트 케이스 생성"""
        test_cases = []
        print(f"총 {num_cases}개의 테스트 케이스 생성 시작...")

        # 유효한 케이스 (50%)
        valid_cases_count = int(num_cases * 0.50)
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

        # invalid_prefix 케이스 (25%)
        invalid_prefix_count = int(num_cases * 0.25)
        print("유효하지 않은 prefix 케이스 생성 중...")
        for _ in range(invalid_prefix_count):
            card_number = self.card_generator._generate_invalid_prefix_number()
            template = random.choice(self.templates)
            text = template.format(number=card_number)
            test_cases.append({
                "text": text,
                "card_number": card_number,
                "is_valid": False,
                "template": template,
                "invalid_type": "invalid_prefix"
            })

        # tricky cases (25%)
        tricky_generators = [
            (self.card_generator._generate_valid_looking_number, "valid_looking"),
            (self.card_generator._generate_pattern_number, "pattern_number"),
            (self.card_generator._generate_similar_format_number, "similar_format")
        ]
        
        remaining_cases = num_cases - valid_cases_count - invalid_prefix_count
        cases_per_tricky_type = remaining_cases // len(tricky_generators)
        print(f"Tricky 케이스 생성 중... (유형별 {cases_per_tricky_type}개)")
        
        for generator, invalid_type in tricky_generators:
            for _ in range(cases_per_tricky_type):
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