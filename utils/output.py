"""Output Writer Utility"""

import json
import csv
from typing import Dict, List, Any, Optional
from datetime import datetime


class OutputWriter:
    """출력 유틸리티"""

    @staticmethod
    def to_json(data: Dict[str, Any], file_path: Optional[str] = None, indent: int = 2) -> str:
        """JSON 형식으로 출력

        Args:
            data: 출력할 데이터
            file_path: 파일 저장 경로 (없으면 문자열 반환)
            indent: 들여쓰기

        Returns:
            JSON 문자열
        """
        json_str = json.dumps(data, ensure_ascii=False, indent=indent, default=str)

        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)

        return json_str

    @staticmethod
    def to_csv(
        data: List[Dict[str, Any]],
        file_path: str,
        fieldnames: Optional[List[str]] = None
    ) -> None:
        """CSV 형식으로 저장

        Args:
            data: 출력할 데이터 (리스트 of 딕셔너리)
            file_path: 파일 저장 경로
            fieldnames: 컬럼명 (없으면 첫 번째 딕셔너리에서 추출)
        """
        if not data:
            raise ValueError("No data to write")

        if fieldnames is None:
            fieldnames = list(data[0].keys())

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    @staticmethod
    def format_table(
        data: Dict[str, Any],
        title: str = "Results",
        col_width: int = 20
    ) -> str:
        """테이블 형식 문자열 생성

        Args:
            data: 출력할 데이터
            title: 테이블 제목
            col_width: 컬럼 너비

        Returns:
            테이블 문자열
        """
        lines = []
        separator = "=" * (col_width * 2 + 3)

        lines.append(separator)
        lines.append(f"  {title}")
        lines.append(separator)

        for key, value in data.items():
            if isinstance(value, float):
                value = f"{value:,.2f}"
            lines.append(f"  {key:<{col_width}} | {str(value):>{col_width}}")

        lines.append(separator)

        return "\n".join(lines)

    @staticmethod
    def generate_report(
        results: Dict[str, Any],
        input_value: float,
        file_path: Optional[str] = None
    ) -> str:
        """예측 리포트 생성

        Args:
            results: 예측 결과
            input_value: 입력값
            file_path: 저장 경로

        Returns:
            리포트 문자열
        """
        report = []
        report.append("# ML-Toolkit 예측 리포트")
        report.append(f"\n생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"입력값 (총물량): {input_value:,.0f}")
        report.append("\n## 예측 결과\n")

        report.append("| 모델 | 예측 대상 | 예측값 |")
        report.append("|------|----------|--------|")

        for name, data in results.items():
            if "error" in data:
                report.append(f"| {name} | ERROR | {data['error']} |")
            else:
                report.append(f"| {name} | {data['target']} | {data['value']:,.2f} |")

        report_str = "\n".join(report)

        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_str)

        return report_str
