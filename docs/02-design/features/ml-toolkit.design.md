# ML-Toolkit Design Document

## 1. System Architecture

### 1.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐│
│  │ predict │  │  train  │  │  list   │  │     compare     ││
│  └────┬────┘  └────┬────┘  └────┬────┘  └────────┬────────┘│
└───────┼────────────┼───────────┼─────────────────┼─────────┘
        │            │           │                 │
┌───────▼────────────▼───────────▼─────────────────▼─────────┐
│                      Model Manager                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              ModelRegistry (Singleton)                │  │
│  │  - load_model(name) -> BaseModel                     │  │
│  │  - list_models() -> List[str]                        │  │
│  │  - get_model_info(name) -> ModelInfo                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│                      Model Layer                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ BaseModel  │◄─┤ElasticNet  │  │  Prophet   │            │
│  │ (Abstract) │  │   Model    │  │   Model    │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│        ▲              ▲               ▲                     │
│        │         ┌────┴────┐    ┌─────┴─────┐              │
│        │         │  ARIMA  │    │  XGBoost  │              │
│        │         │  Model  │    │   Model   │              │
│        │         └─────────┘    └───────────┘              │
└─────────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ DataLoader │  │ DataValidator│ │OutputWriter│            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Class Design

### 2.1 BaseModel (Abstract)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

class BaseModel(ABC):
    """모든 ML 모델의 베이스 클래스"""

    def __init__(self, name: str, model_path: str = None):
        self.name = name
        self.model_path = model_path
        self.model = None
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def load(self) -> None:
        """모델 로드"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 수행"""
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """모델 학습"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """모델 저장"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "path": self.model_path,
            **self.metadata
        }
```

### 2.2 ElasticNetModel

```python
class ElasticNetModel(BaseModel):
    """ElasticNet 모델 래퍼"""

    def load(self) -> None:
        import joblib
        self.model = joblib.load(self.model_path)
        self.metadata = {
            "coef": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
            "alpha": self.model.alpha,
            "l1_ratio": self.model.l1_ratio
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import ElasticNet
        self.model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=0)
        self.model.fit(X, y)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.model, path)
```

### 2.3 ModelRegistry

```python
class ModelRegistry:
    """모델 레지스트리 (Singleton)"""

    _instance = None

    MODEL_CONFIGS = {
        "pallets": {"path": "ElasticNet_pallets.pkl", "type": "elasticnet", "target": "총파렛트수"},
        "workers": {"path": "ElasticNet_workers.pkl", "type": "elasticnet", "target": "중근조인원수"},
        "workers2": {"path": "ElasticNet_workers2.pkl", "type": "elasticnet", "target": "석근조인원수"},
        "workers3": {"path": "ElasticNet_workers3.pkl", "type": "elasticnet", "target": "야근도착총인원수"},
        "workers4": {"path": "ElasticNet_workers4.pkl", "type": "elasticnet", "target": "시간외중근인원수"},
        "workers5": {"path": "ElasticNet_workers5.pkl", "type": "elasticnet", "target": "시간외석근인원수"},
        "workers6": {"path": "ElasticNet_workers6.pkl", "type": "elasticnet", "target": "시간외야근인원수"},
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, name: str) -> BaseModel:
        config = self.MODEL_CONFIGS.get(name)
        if not config:
            raise ValueError(f"Unknown model: {name}")

        model_class = self._get_model_class(config["type"])
        model = model_class(name=name, model_path=config["path"])
        model.load()
        return model

    def list_models(self) -> List[str]:
        return list(self.MODEL_CONFIGS.keys())
```

---

## 3. CLI Commands

### 3.1 Command Structure

```bash
# 예측 실행
ml-toolkit predict --model workers --input 200000

# 모든 모델로 예측
ml-toolkit predict-all --input 200000

# 모델 목록 조회
ml-toolkit list

# 모델 상세 정보
ml-toolkit info --model workers

# 새 모델 학습
ml-toolkit train --model custom --data data.csv --target column_name

# 모델 비교
ml-toolkit compare --input 200000
```

### 3.2 CLI Implementation

```python
import click

@click.group()
def cli():
    """ML-Toolkit: 다목적 머신러닝 예측 도구"""
    pass

@cli.command()
@click.option('--model', '-m', required=True, help='모델 이름')
@click.option('--input', '-i', required=True, type=float, help='입력값 (총물량)')
def predict(model: str, input: float):
    """단일 모델로 예측"""
    registry = ModelRegistry()
    m = registry.load_model(model)
    result = m.predict([[input]])[0]
    click.echo(f"예측 결과: {result:.2f}")

@cli.command()
def list():
    """등록된 모델 목록"""
    registry = ModelRegistry()
    for name in registry.list_models():
        click.echo(f"  - {name}")

if __name__ == "__main__":
    cli()
```

---

## 4. Data Flow

### 4.1 Prediction Flow

```
Input (총물량: 200000)
        │
        ▼
┌───────────────┐
│   CLI Layer   │  ml-toolkit predict --model workers --input 200000
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ModelRegistry  │  registry.load_model("workers")
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ElasticNetModel│  model.predict([[200000]])
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Output: 64.71 │
└───────────────┘
```

---

## 5. Configuration

### 5.1 models.yaml

```yaml
models:
  elasticnet:
    - name: pallets
      path: ElasticNet_pallets.pkl
      target: 총파렛트수
      input: 총물량
    - name: workers
      path: ElasticNet_workers.pkl
      target: 중근조인원수
      input: 총물량
    # ... more models

  prophet:
    - name: parcel_forecast
      path: prophet_parcel.pkl
      target: 소포물량
      input: date

settings:
  base_path: /Users/shindongsik/_Book_claude-code/proj/proj
  output_format: table  # table, json, csv
```

---

## 6. Implementation Order

| Order | Component | File | Priority |
|-------|-----------|------|----------|
| 1 | BaseModel | models/base.py | High |
| 2 | ElasticNetModel | models/elasticnet.py | High |
| 3 | ModelRegistry | models/registry.py | High |
| 4 | CLI (predict, list) | cli.py | High |
| 5 | DataLoader | utils/data_loader.py | Medium |
| 6 | OutputWriter | utils/output.py | Medium |
| 7 | Tests | tests/test_models.py | Medium |

---

## 7. Dependencies

```
# requirements.txt
click>=8.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.1.0
pyyaml>=6.0
```

---

*Created: 2025-02-11*
*Phase: Design*
*References: ml-toolkit.plan.md*
