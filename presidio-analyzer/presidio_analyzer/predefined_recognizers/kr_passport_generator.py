import random
import string
import json
from typing import List, Dict
import os

class KoreanPassportNumberGenerator:
    """한국 여권번호 생성기"""
    
    def __init__(self):
        self.prefix_types = ['M', 'R']
        self.second_char = string.digits + string.ascii_uppercase

    def _generate_valid_passport_number(self):
        """유효한 여권번호 생성"""
        prefix = random.choice(self.prefix_types)
        second_char = random.choice(self.second_char)
        base_number = ''.join(random.choices(string.digits, k=6))
        
        full_number = prefix + second_char + base_number
        checksum = self._calculate_checksum(full_number)
        
        return full_number + str(checksum)

    def _generate_invalid_prefix_number(self):
        """잘못된 접두사를 가진 여권번호 생성"""
        invalid_prefix = random.choice(string.ascii_uppercase.replace('M', '').replace('R', ''))
        return invalid_prefix + self._generate_valid_passport_number()[1:]

    def _generate_invalid_second_char_number(self):
        """잘못된 두 번째 문자를 가진 여권번호 생성"""
        valid_number = self._generate_valid_passport_number()
        invalid_second_char = random.choice(string.ascii_lowercase)
        return valid_number[0] + invalid_second_char + valid_number[2:]

    def _generate_invalid_length_number(self):
        """잘못된 길이의 여권번호 생성"""
        valid_number = self._generate_valid_passport_number()
        if random.choice([True, False]):
            return valid_number[:-1]  # 한 자리 부족
        else:
            return valid_number + random.choice(string.digits)  # 한 자리 추가

    def _generate_invalid_checksum_number(self):
        """잘못된 체크섬을 가진 여권번호 생성"""
        valid_number = self._generate_valid_passport_number()
        invalid_checksum = (int(valid_number[-1]) + 1) % 10
        return valid_number[:-1] + str(invalid_checksum)

    def _generate_invalid_chars_number(self):
        """잘못된 문자를 포함한 여권번호 생성"""
        valid_number = self._generate_valid_passport_number()
        position = random.randint(2, len(valid_number) - 1)
        invalid_char = random.choice(string.ascii_lowercase)
        return valid_number[:position] + invalid_char + valid_number[position+1:]

    def _calculate_checksum(self, number: str) -> int:
        """
        표준 한국 여권 체크섬 계산
        """
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
            # A=10, B=11, ..., Z=35
            second_value = ord(second_char) - ord('A') + 10
        total += second_value * weights[1]
        
        # 나머지 숫자 처리
        for i, char in enumerate(number[2:8], 2):
            total += int(char) * weights[i]
        
        # 체크섬 계산: (10 - (total % 10)) % 10
        return (10 - (total % 10)) % 10


class PassportTestGenerator:
    """여권번호 테스트 케이스 생성기"""
    
    def __init__(self):
        self.passport_generator = KoreanPassportNumberGenerator()
        self.templates = [
            "여권번호는 {number}입니다.",
            "내 여권번호는 {number}예요.",
            "여권: {number}",
            "PASSPORT NUMBER: {number}",
            "여권 번호: {number}",
            "본인의 여권번호는 {number}입니다.",
            "해당 고객의 여권번호는 {number}로 확인됩니다.",
            "여권정보: {number}",
            "여권발급번호: {number}",
            "여권 발급 번호: {number}",
            "PASSPORT: {number}",
            "Korean Passport: {number}",
            "여권번호 {number}로 발급되었습니다.",
            "여권번호가 {number}인 여권을 신청합니다.",
            "여권번호 {number}에 대한 재발급을 신청합니다.",
            "{number} 번호로 발급된 여권입니다.",
            "여권번호({number})로 조회해주세요.",
            "고객님의 여권번호는 {number}입니다.",
            "여권번호는 {number}이며,",
            "새로 발급받은 여권번호는 {number}입니다."
        ]
        
        # tests 폴더 생성
        self.test_dir = "tests"
        os.makedirs(self.test_dir, exist_ok=True)

    def generate_test_cases(self, num_cases: int = 1000) -> List[Dict]:
        """테스트 케이스 생성 (80% valid, 20% invalid)"""
        test_cases = []
        print(f"총 {num_cases}개의 테스트 케이스 생성 시작...")
        
        # 유효한 케이스 (80%)
        valid_cases_count = int(num_cases * 0.8)
        print(f"유효한 여권번호 생성 중... (목표: {valid_cases_count}개)")
        for i in range(valid_cases_count):
            if i % 100 == 0:
                print(f"- {i}/{valid_cases_count} 완료")
            passport_number = self.passport_generator._generate_valid_passport_number()
            template = random.choice(self.templates)
            text = template.format(number=passport_number)
            test_cases.append({
                "text": text,
                "passport_number": passport_number,
                "is_valid": True,
                "template": template
            })
        
        # 유효하지 않은 케이스 (20%)
        invalid_cases_count = num_cases - valid_cases_count
        print(f"\n유효하지 않은 여권번호 생성 중... (목표: {invalid_cases_count}개)")
        
        invalid_generators = [
            (self.passport_generator._generate_invalid_prefix_number, "invalid_prefix_number"),
            (self.passport_generator._generate_invalid_second_char_number, "invalid_second_char_number"),
            (self.passport_generator._generate_invalid_length_number, "invalid_length_number"),
            (self.passport_generator._generate_invalid_checksum_number, "invalid_checksum_number"),
            (self.passport_generator._generate_invalid_chars_number, "invalid_chars_number")
        ]
        
        cases_per_type = invalid_cases_count // len(invalid_generators)
        remaining_cases = invalid_cases_count % len(invalid_generators)
        
        for generator, invalid_type in invalid_generators:
            num_type_cases = cases_per_type + (1 if remaining_cases > 0 else 0)
            remaining_cases -= 1 if remaining_cases > 0 else 0
            
            for i in range(num_type_cases):
                passport_number = generator()
                template = random.choice(self.templates)
                text = template.format(number=passport_number)
                test_cases.append({
                    "text": text,
                    "passport_number": passport_number,
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
        
        # Invalid 유형별 통계
        invalid_type_stats = {}
        for case in test_cases:
            if not case["is_valid"]:
                invalid_type = case["invalid_type"]
                invalid_type_stats[invalid_type] = invalid_type_stats.get(invalid_type, 0) + 1
        
        print("\nInvalid 유형별 분포:")
        for invalid_type, count in invalid_type_stats.items():
            print(f"- {invalid_type}: {count}개 ({count/actual_invalid*100:.1f}%)")
        
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
    # 테스트 케이스 생성기 초기화
    test_generator = PassportTestGenerator()
    
    # 1000개의 테스트 케이스 생성
    test_cases = test_generator.generate_test_cases(1000)
    
    # JSON 파일로 저장
    test_generator.save_to_json(test_cases, 'passport_test_cases_1000.json')
    
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
        print(f"여권번호: {case['passport_number']}")
        print(f"유효성: {'유효함' if case['is_valid'] else '유효하지 않음'}")
        if not case['is_valid']:
            print(f"오류 유형: {case.get('invalid_type', 'unknown')}")
        print("-" * 50)


if __name__ == "__main__":
    main()