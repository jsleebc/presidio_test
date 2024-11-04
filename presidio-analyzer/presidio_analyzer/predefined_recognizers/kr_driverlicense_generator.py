import random
import string
import json
from typing import List, Dict
import os

class KoreanDriverLicenseGenerator:
    """한국 운전면허번호 생성기"""
    
    def __init__(self):
        self.region_codes = [str(i).zfill(2) for i in range(11, 29)]  # 11-28
        self.issue_years = [str(i).zfill(2) for i in range(0, 100)]  # 00-99

    def _generate_valid_license(self):
        """유효한 운전면허번호 생성"""
        region = random.choice(self.region_codes)
        issue_year = random.choice(self.issue_years)
        serial = f"{random.randint(0, 999999):06d}"
        check_digit = f"{random.randint(0, 99):02d}"
        return f"{region}-{issue_year}-{serial}-{check_digit}"

    def _generate_invalid_region_license(self):
        """잘못된 지역코드를 가진 운전면허번호 생성"""
        invalid_region = str(random.choice([*range(1, 11), *range(29, 100)])).zfill(2)
        issue_year = random.choice(self.issue_years)
        serial = f"{random.randint(0, 999999):06d}"
        check_digit = f"{random.randint(0, 99):02d}"
        return f"{invalid_region}-{issue_year}-{serial}-{check_digit}"

    def _generate_invalid_issue_year_license(self):
        """잘못된 발급년도를 가진 운전면허번호 생성"""
        region = random.choice(self.region_codes)
        invalid_year = "XX"
        serial = f"{random.randint(0, 999999):06d}"
        check_digit = f"{random.randint(0, 99):02d}"
        return f"{region}-{invalid_year}-{serial}-{check_digit}"

    def _generate_invalid_serial_license(self):
        """잘못된 일련번호를 가진 운전면허번호 생성"""
        region = random.choice(self.region_codes)
        issue_year = random.choice(self.issue_years)
        invalid_serial = f"{random.randint(1000000, 9999999):07d}"
        check_digit = f"{random.randint(0, 99):02d}"
        return f"{region}-{issue_year}-{invalid_serial}-{check_digit}"

    def _generate_invalid_format_license(self):
        """잘못된 형식의 운전면허번호 생성"""
        region = random.choice(self.region_codes)
        issue_year = random.choice(self.issue_years)
        serial = f"{random.randint(0, 999999):06d}"
        check_digit = f"{random.randint(0, 99):02d}"
        return f"{region}{issue_year}{serial}{check_digit}"

    def _generate_invalid_chars_license(self):
        """잘못된 문자를 포함한 운전면허번호 생성"""
        valid_license = self._generate_valid_license()
        position = random.randint(0, len(valid_license) - 1)
        invalid_chars = ['A', 'B', 'C', 'X', 'Y', 'Z']
        return valid_license[:position] + random.choice(invalid_chars) + valid_license[position+1:]

class DriverLicenseTestGenerator:
    """운전면허번호 테스트 케이스 생성기"""
    
    def __init__(self):
        self.license_generator = KoreanDriverLicenseGenerator()
        self.templates = [
            "운전면허번호는 {number}입니다.",
            "면허번호: {number}",
            "DRIVER LICENSE: {number}",
            "운전면허 번호: {number}",
            "본인의 운전면허번호는 {number}입니다.",
            "해당 고객의 면허번호는 {number}로 확인됩니다.",
            "면허정보: {number}",
            "면허증번호: {number}",
            "운전면허증 번호: {number}",
            "LICENSE: {number}",
            "Korean Driver License: {number}",
            "면허번호 {number}로 발급되었습니다.",
            "운전면허번호가 {number}인 면허증을 신청합니다.",
            "면허번호 {number}에 대한 재발급을 신청합니다.",
            "{number} 번호로 발급된 면허증입니다.",
            "면허번호({number})로 조회해주세요.",
            "고객님의 면허번호는 {number}입니다.",
            "운전면허번호는 {number}이며,",
            "새로 발급받은 면허번호는 {number}입니다.",
            "Driver's License Number: {number}"
        ]
        
        self.test_dir = "tests"
        os.makedirs(self.test_dir, exist_ok=True)

    def generate_test_cases(self, num_cases: int = 1000) -> List[Dict]:
        """테스트 케이스 생성 (80% valid, 20% invalid)"""
        test_cases = []
        print(f"총 {num_cases}개의 테스트 케이스 생성 시작...")

        # 유효한 케이스 (80%)
        valid_cases_count = int(num_cases * 0.8)
        print(f"유효한 면허번호 생성 중... (목표: {valid_cases_count}개)")
        
        for i in range(valid_cases_count):
            if i % 100 == 0:
                print(f"- {i}/{valid_cases_count} 완료")
            
            license_number = self.license_generator._generate_valid_license()
            template = random.choice(self.templates)
            text = template.format(number=license_number)
            
            test_cases.append({
                "text": text,
                "license_number": license_number,
                "is_valid": True,
                "template": template
            })

        # 유효하지 않은 케이스 (20%)
        invalid_generators = [
            (self.license_generator._generate_invalid_region_license, "invalid_region"),
            (self.license_generator._generate_invalid_issue_year_license, "invalid_year"),
            (self.license_generator._generate_invalid_serial_license, "invalid_serial"),
            (self.license_generator._generate_invalid_format_license, "invalid_format"),
            (self.license_generator._generate_invalid_chars_license, "invalid_chars")
        ]

        invalid_cases_count = num_cases - valid_cases_count
        cases_per_type = invalid_cases_count // len(invalid_generators)
        
        for generator, invalid_type in invalid_generators:
            for _ in range(cases_per_type):
                license_number = generator()
                template = random.choice(self.templates)
                text = template.format(number=license_number)
                
                test_cases.append({
                    "text": text,
                    "license_number": license_number,
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
    test_generator = DriverLicenseTestGenerator()
    test_cases = test_generator.generate_test_cases(1000)
    test_generator.save_to_json(test_cases, 'driverlicense_test_cases_1000.json')

if __name__ == "__main__":
    main()