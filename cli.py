#!/usr/bin/env python3
"""ML-Toolkit CLI - 다목적 머신러닝 예측 도구"""

import sys
import os

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
import numpy as np
from typing import Optional

from models import ModelRegistry


def format_number(value: float, decimals: int = 2) -> str:
    """숫자 포맷팅"""
    return f"{value:,.{decimals}f}"


@click.group()
@click.version_option(version="1.0.0", prog_name="ml-toolkit")
def cli():
    """ML-Toolkit: 다목적 머신러닝 예측 도구

    물류센터 인적/물적 자원 예측을 위한 통합 ML 도구입니다.
    """
    pass


@cli.command()
@click.option('--model', '-m', required=True, help='모델 이름 (예: workers, pallets)')
@click.option('--input', '-i', 'input_value', required=True, type=float, help='입력값 (총물량)')
@click.option('--format', '-f', 'output_format', type=click.Choice(['text', 'json']), default='text', help='출력 형식')
def predict(model: str, input_value: float, output_format: str):
    """단일 모델로 예측 수행

    예시:
        ml-toolkit predict --model workers --input 200000
    """
    try:
        registry = ModelRegistry()
        m = registry.load_model(model)
        result = m.predict(np.array([[input_value]]))[0]

        if output_format == 'json':
            import json
            output = {
                "model": model,
                "input": input_value,
                "prediction": result,
                "target": m.metadata.get("target", ""),
                "formula": m.get_formula() if hasattr(m, 'get_formula') else ""
            }
            click.echo(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            target = m.metadata.get("target", "예측값")
            click.echo(f"\n{'='*50}")
            click.echo(f"  모델: {model}")
            click.echo(f"  입력 (총물량): {format_number(input_value, 0)}")
            click.echo(f"  예측 ({target}): {format_number(result)}")
            if hasattr(m, 'get_formula'):
                click.echo(f"  공식: {m.get_formula()}")
            click.echo(f"{'='*50}\n")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("predict-all")
@click.option('--input', '-i', 'input_value', required=True, type=float, help='입력값 (총물량)')
@click.option('--format', '-f', 'output_format', type=click.Choice(['table', 'json']), default='table', help='출력 형식')
def predict_all(input_value: float, output_format: str):
    """모든 모델로 예측 수행

    예시:
        ml-toolkit predict-all --input 200000
    """
    try:
        registry = ModelRegistry()
        results = registry.predict_all(input_value)

        if output_format == 'json':
            import json
            click.echo(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            click.echo(f"\n{'='*60}")
            click.echo(f"  총물량: {format_number(input_value, 0)} 기준 전체 예측")
            click.echo(f"{'='*60}")
            click.echo(f"  {'모델':<12} | {'예측 대상':<20} | {'예측값':>12}")
            click.echo(f"  {'-'*12}-+-{'-'*20}-+-{'-'*12}")

            total_workers = 0
            for name, data in results.items():
                if "error" in data:
                    click.echo(f"  {name:<12} | ERROR: {data['error']}")
                else:
                    value = data["value"]
                    target = data["target"]
                    click.echo(f"  {name:<12} | {target:<20} | {format_number(value):>12}")
                    if name != "pallets":
                        total_workers += value

            click.echo(f"  {'-'*12}-+-{'-'*20}-+-{'-'*12}")
            click.echo(f"  {'총 인원':<12} | {'':<20} | {format_number(total_workers):>12}")
            click.echo(f"{'='*60}\n")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("list")
@click.option('--verbose', '-v', is_flag=True, help='상세 정보 표시')
def list_models(verbose: bool):
    """등록된 모델 목록 조회

    예시:
        ml-toolkit list
        ml-toolkit list --verbose
    """
    registry = ModelRegistry()
    models = registry.list_models()

    click.echo(f"\n등록된 모델 ({len(models)}개):")
    click.echo("-" * 50)

    for name in models:
        config = registry.get_model_config(name)
        if verbose:
            click.echo(f"\n  [{name}]")
            click.echo(f"    타입: {config['type']}")
            click.echo(f"    대상: {config['target']}")
            click.echo(f"    입력: {config['input']}")
            click.echo(f"    설명: {config['description']}")
        else:
            click.echo(f"  - {name}: {config['description']}")

    click.echo()


@cli.command()
@click.option('--model', '-m', required=True, help='모델 이름')
def info(model: str):
    """모델 상세 정보 조회

    예시:
        ml-toolkit info --model workers
    """
    try:
        registry = ModelRegistry()
        m = registry.load_model(model)
        info_data = m.get_info()

        click.echo(f"\n{'='*50}")
        click.echo(f"  모델 정보: {model}")
        click.echo(f"{'='*50}")

        for key, value in info_data.items():
            if isinstance(value, list):
                value = f"[{', '.join(f'{v:.6f}' for v in value)}]"
            elif isinstance(value, float):
                value = f"{value:.6f}"
            click.echo(f"  {key}: {value}")

        if hasattr(m, 'get_formula'):
            click.echo(f"\n  예측 공식: {m.get_formula()}")

        click.echo(f"{'='*50}\n")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', 'input_value', required=True, type=float, help='입력값 (총물량)')
def compare(input_value: float):
    """모델 간 예측 비교

    예시:
        ml-toolkit compare --input 200000
    """
    registry = ModelRegistry()
    results = registry.predict_all(input_value)

    click.echo(f"\n{'='*60}")
    click.echo(f"  모델 비교 (총물량: {format_number(input_value, 0)})")
    click.echo(f"{'='*60}")

    # 인적자원 모델만 추출하여 민감도 순 정렬
    worker_models = []
    for name, data in results.items():
        if name != "pallets" and "value" in data:
            model = registry.load_model(name)
            coef = model.metadata.get("coef", [0])[0]
            worker_models.append({
                "name": name,
                "value": data["value"],
                "target": data["target"],
                "coef": coef
            })

    # 계수(민감도) 기준 정렬
    worker_models.sort(key=lambda x: x["coef"], reverse=True)

    click.echo(f"\n  [민감도 순위 - 물량 변화에 따른 인원 변화]")
    click.echo(f"  {'-'*55}")
    for i, m in enumerate(worker_models, 1):
        sensitivity = "높음" if m["coef"] > 0.0005 else "중간" if m["coef"] > 0.0002 else "낮음"
        click.echo(f"  {i}. {m['target']:<18} | {format_number(m['value']):>8}명 | 민감도: {sensitivity}")

    click.echo(f"{'='*60}\n")


if __name__ == "__main__":
    cli()
