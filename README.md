# ML-Toolkit

다목적 머신러닝 예측 도구 - 물류센터 인적/물적 자원 예측을 위한 통합 CLI 도구

## 설치

```bash
cd ml-toolkit
pip install -r requirements.txt
```

## 사용법

### 모델 목록 조회

```bash
python cli.py list
python cli.py list --verbose
```

### 단일 모델 예측

```bash
python cli.py predict --model workers --input 200000
python cli.py predict -m pallets -i 150000 --format json
```

### 전체 모델 예측

```bash
python cli.py predict-all --input 200000
python cli.py predict-all -i 200000 --format json
```

### 모델 정보 조회

```bash
python cli.py info --model workers
```

### 모델 비교 (민감도 순위)

```bash
python cli.py compare --input 200000
```

## 등록된 모델

| 모델 | 예측 대상 | 설명 |
|------|----------|------|
| pallets | 총파렛트수 | 물적자원 예측 |
| workers | 중근조인원수 | 인적자원 - 중근조 |
| workers2 | 석근조인원수 | 인적자원 - 석근조 |
| workers3 | 야근도착총인원수 | 인적자원 - 야근 |
| workers4 | 시간외중근인원수 | 인적자원 - 시간외중근 |
| workers5 | 시간외석근인원수 | 인적자원 - 시간외석근 |
| workers6 | 시간외야근인원수 | 인적자원 - 시간외야근 |

## 프로젝트 구조

```
ml-toolkit/
├── cli.py              # CLI 진입점
├── models/
│   ├── __init__.py
│   ├── base.py         # BaseModel 추상 클래스
│   ├── elasticnet.py   # ElasticNet 모델 래퍼
│   └── registry.py     # ModelRegistry (Singleton)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py  # 데이터 로딩 유틸리티
│   └── output.py       # 출력 유틸리티
├── tests/
│   └── test_models.py  # 테스트
├── docs/               # PDCA 문서
└── requirements.txt
```

## 예측 공식

모든 모델은 ElasticNet 기반으로 다음 공식을 사용합니다:

```
예측값 = 계수 × 총물량 + 절편
```

## 라이선스

MIT License
