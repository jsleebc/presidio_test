import random
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict

class KoreanRRNGenerator:
    """한국 주민등록번호 생성기"""
    
    def __init__(self):
        self.current_year = datetime.now().year

    def _generate_valid_rrn(self):
        """유효한 주민등록번호 생성"""
        birth_date = self._generate_random_date()
        gender_digit = self._get_gender_digit(birth_date.year)
        region_code = random.randint(0, 95)
        serial = random.randint(0, 999)
        
        rrn = f"{birth_date.strftime('%y%m%d')}{gender_digit}{region_code:02d}{serial:03d}"
        checksum = self._calculate_checksum(rrn)
        
        return f"{rrn}{checksum}"

    def _generate_invalid_date_rrn(self):
        """잘못된 날짜를 가진 주민등록번호 생성"""
        year = random.randint(0, 99)
        month = random.randint(1, 12)
        day = 31 if month in [4, 6, 9, 11] else (29 if month == 2 and year % 4 != 0 else random.randint(29, 31))
        
        gender_digit = self._get_gender_digit(2000 + year if year < 22 else 1900 + year)
        region_code = random.randint(0, 95)
        serial = random.randint(0, 999)
        
        rrn = f"{year:02d}{month:02d}{day:02d}{gender_digit}{region_code:02d}{serial:03d}"
        checksum = self._calculate_checksum(rrn)
        
        return f"{rrn}{checksum}"

    def _generate_invalid_checksum_rrn(self):
        """잘못된 체크섬을 가진 주민등록번호 생성"""
        valid_rrn = self._generate_valid_rrn()
        invalid_checksum = (int(valid_rrn[-1]) + 1) % 10
        return f"{valid_rrn[:-1]}{invalid_checksum}"

    def _generate_invalid_region_code_rrn(self):
        """잘못된 지역코드를 가진 주민등록번호 생성"""
        birth_date = self._generate_random_date()
        gender_digit = self._get_gender_digit(birth_date.year)
        region_code = random.randint(96, 99)  # Invalid region code
        serial = random.randint(0, 999)
        
        rrn = f"{birth_date.strftime('%y%m%d')}{gender_digit}{region_code:02d}{serial:03d}"
        checksum = self._calculate_checksum(rrn)
        
        return f"{rrn}{checksum}"

    def _generate_invalid_gender_year_rrn(self):
        """잘못된 성별/연도 조합을 가진 주민등록번호 생성"""
        birth_date = self._generate_random_date()
        year = birth_date.year
        if year < 2000:
            gender_digit = random.choice([3, 4, 7, 8])  # Invalid for pre-2000
        else:
            gender_digit = random.choice([1, 2, 5, 6])  # Invalid for post-2000
        
        region_code = random.randint(0, 95)
        serial = random.randint(0, 999)
        
        rrn = f"{birth_date.strftime('%y%m%d')}{gender_digit}{region_code:02d}{serial:03d}"
        checksum = self._calculate_checksum(rrn)
        
        return f"{rrn}{checksum}"

    def _generate_future_date_rrn(self):
        """미래 날짜를 가진 주민등록번호 생성"""
        future_date = datetime.now() + timedelta(days=random.randint(1, 3650))
        gender_digit = self._get_gender_digit(future_date.year)
        region_code = random.randint(0, 95)
        serial = random.randint(0, 999)
        
        rrn = f"{future_date.strftime('%y%m%d')}{gender_digit}{region_code:02d}{serial:03d}"
        checksum = self._calculate_checksum(rrn)
        
        return f"{rrn}{checksum}"

    def _generate_random_date(self):
        start_date = datetime(1900, 1, 1)
        end_date = datetime.now()
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        return start_date + timedelta(days=random_number_of_days)

    def _get_gender_digit(self, year):
        if 1800 <= year <= 1899:
            return random.choice([9, 0])
        elif 1900 <= year <= 1999:
            return random.choice([1, 2, 5, 6])
        else:  # 2000년대
            return random.choice([3, 4, 7, 8])

    def _calculate_checksum(self, rrn):
        multipliers = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
        checksum = sum(int(rrn[i]) * multipliers[i] for i in range(12))
        return (11 - (checksum % 11)) % 10


class RRNTestGenerator:
    """주민등록번호 테스트 케이스 생성기"""
    
    def __init__(self):
        self.rrn_generator = KoreanRRNGenerator()
        self.templates = [
            "주민등록번호는 {number}입니다.",
            "내 주민등록번호는 {number}예요.",
            "주민번호: {number}",
            "RRN: {number}",
            "주민등록번호: {number}",
            "본인의 주민등록번호는 {number}입니다.",
            "해당 고객의 주민등록번호는 {number}로 확인됩니다.",
            "주민정보: {number}",
            "주민등록정보: {number}",
            "주민등록 번호: {number}",
            "RRN NUMBER: {number}",
            "Korean RRN: {number}",
            "주민번호 {number}로 등록되어 있습니다.",
            "주민등록번호가 {number}인 회원입니다.",
            "주민등록번호 {number}에 대한 정보입니다.",
            "{number} 번호로 등록된 정보입니다.",
            "주민등록번호({number})로 조회해주세요.",
            "고객님의 주민등록번호는 {number}입니다.",
            "주민등록번호는 {number}이며,",
            "확인된 주민등록번호는 {number}입니다."
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
        print(f"유효한 주민등록번호 생성 중... (목표: {valid_cases_count}개)")
        for i in range(valid_cases_count):
            if i % 100 == 0:
                print(f"- {i}/{valid_cases_count} 완료")
            rrn = self.rrn_generator._generate_valid_rrn()
            template = random.choice(self.templates)
            text = template.format(number=rrn)
            test_cases.append({
                "text": text,
                "rrn": rrn,
                "is_valid": True,
                "template": template
            })
        
        # 유효하지 않은 케이스 (20%)
        invalid_cases_count = num_cases - valid_cases_count
        print(f"\n유효하지 않은 주민등록번호 생성 중... (목표: {invalid_cases_count}개)")
        
        invalid_generators = [
            (self.rrn_generator._generate_invalid_date_rrn, "invalid_date"),
            (self.rrn_generator._generate_invalid_checksum_rrn, "invalid_checksum"),
            (self.rrn_generator._generate_invalid_region_code_rrn, "invalid_region_code"),
            (self.rrn_generator._generate_invalid_gender_year_rrn, "invalid_gender_year"),
            (self.rrn_generator._generate_future_date_rrn, "future_date")
        ]
        
        # 각 invalid 유형별 생성할 케이스 수 계산
        cases_per_type = invalid_cases_count // len(invalid_generators)
        remaining_cases = invalid_cases_count % len(invalid_generators)
        
        # 각 유형별로 균등하게 생성
        for generator, invalid_type in invalid_generators:
            num_type_cases = cases_per_type + (1 if remaining_cases > 0 else 0)
            remaining_cases -= 1 if remaining_cases > 0 else 0
            
            for i in range(num_type_cases):
                rrn = generator()
                template = random.choice(self.templates)
                text = template.format(number=rrn)
                test_cases.append({
                    "text": text,
                    "rrn": rrn,
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
        # tests 폴더 내에 파일 저장
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
    test_generator = RRNTestGenerator()
    
    # 1000개의 테스트 케이스 생성
    test_cases = test_generator.generate_test_cases(1000)
    
    # JSON 파일로 저장
    test_generator.save_to_json(test_cases, 'rrn_test_cases_1000.json')
    
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
        print(f"주민등록번호: {case['rrn']}")
        print(f"유효성: {'유효함' if case['is_valid'] else '유효하지 않음'}")
        if not case['is_valid']:
            print(f"오류 유형: {case.get('invalid_type', 'unknown')}")
        print("-" * 50)


if __name__ == "__main__":
    main()