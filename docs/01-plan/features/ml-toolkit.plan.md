# ML-Toolkit Plan Document

## 1. Overview

### 1.1 Project Name
**ML-Toolkit** - 다목적 머신러닝 예측 도구

### 1.2 Project Level
**Starter** - CLI 기반 빠른 구현

### 1.3 Objective
기존 proj의 ElasticNet 모델들을 확장하여 다양한 ML 모델을 통합 지원하는 다목적 도구 개발

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | 다중 모델 지원 (ElasticNet, Prophet, ARIMA, XGBoost, LSTM) | High |
| FR-02 | 기존 pkl 모델 로드 및 예측 | High |
| FR-03 | 새 모델 학습 및 저장 | Medium |
| FR-04 | CSV 데이터 입력 지원 | High |
| FR-05 | 예측 결과 CSV/JSON 출력 | Medium |
| FR-06 | 모델 성능 비교 | Low |

### 2.2 Non-Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-01 | Python 3.10+ 호환 | High |
| NFR-02 | CLI 인터페이스 | High |
| NFR-03 | 응답 시간 < 5초 | Medium |

---

## 3. Scope

### 3.1 In Scope
- 기존 7개 ElasticNet 모델 통합
- 새로운 모델 타입 추가 (Prophet, ARIMA, XGBoost)
- CLI 명령어 인터페이스
- 예측 결과 출력

### 3.2 Out of Scope
- 웹 UI (Phase 2에서 고려)
- 실시간 스트리밍 예측
- GPU 가속

---

## 4. Architecture Overview

```
ml-toolkit/
├── cli.py              # CLI 진입점
├── models/
│   ├── __init__.py
│   ├── base.py         # 베이스 모델 클래스
│   ├── elasticnet.py   # ElasticNet 래퍼
│   ├── prophet.py      # Prophet 래퍼
│   ├── arima.py        # ARIMA 래퍼
│   └── xgboost.py      # XGBoost 래퍼
├── utils/
│   ├── __init__.py
│   ├── data_loader.py  # 데이터 로딩
│   └── output.py       # 결과 출력
├── config/
│   └── models.yaml     # 모델 설정
└── tests/
    └── test_models.py  # 테스트
```

---

## 5. Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| ML Libraries | scikit-learn, prophet, statsmodels, xgboost |
| CLI | Click / Typer |
| Data | Pandas, NumPy |
| Config | PyYAML |
| Testing | pytest |

---

## 6. Milestones

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | 기존 ElasticNet 모델 통합 | Planned |
| Phase 2 | Prophet/ARIMA 추가 | Planned |
| Phase 3 | XGBoost 추가 | Planned |
| Phase 4 | CLI 인터페이스 완성 | Planned |

---

## 7. Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 모델 버전 호환성 | Medium | High | sklearn 버전 고정 |
| 데이터 형식 불일치 | Low | Medium | 데이터 검증 로직 추가 |

---

## 8. Success Criteria

- [ ] 기존 7개 pkl 모델 모두 로드 가능
- [ ] 최소 3개 모델 타입 지원
- [ ] CLI로 예측 실행 가능
- [ ] 예측 결과 정확도 기존과 동일

---

*Created: 2025-02-11*
*Phase: Plan*
