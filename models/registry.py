"""Model Registry - Singleton Pattern"""

import os
from typing import Dict, List, Optional, Type
from .base import BaseModel
from .elasticnet import ElasticNetModel


class ModelRegistry:
    """모델 레지스트리 (Singleton)

    모든 등록된 모델을 관리하고 로드하는 중앙 레지스트리
    """

    _instance = None
    _base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 기본 모델 설정
    MODEL_CONFIGS: Dict[str, Dict] = {
        "pallets": {
            "path": "ElasticNet_pallets.pkl",
            "type": "elasticnet",
            "target": "총파렛트수",
            "input": "총물량",
            "description": "물적자원 - 총파렛트수 예측"
        },
        "workers": {
            "path": "ElasticNet_workers.pkl",
            "type": "elasticnet",
            "target": "중근조인원수",
            "input": "총물량",
            "description": "인적자원 - 중근조 인원수 예측"
        },
        "workers2": {
            "path": "ElasticNet_workers2.pkl",
            "type": "elasticnet",
            "target": "석근조인원수",
            "input": "총물량",
            "description": "인적자원 - 석근조 인원수 예측"
        },
        "workers3": {
            "path": "ElasticNet_workers3.pkl",
            "type": "elasticnet",
            "target": "야근도착총인원수",
            "input": "총물량",
            "description": "인적자원 - 야근도착 인원수 예측"
        },
        "workers4": {
            "path": "ElasticNet_workers4.pkl",
            "type": "elasticnet",
            "target": "시간외중근인원수",
            "input": "총물량",
            "description": "인적자원 - 시간외중근 인원수 예측"
        },
        "workers5": {
            "path": "ElasticNet_workers5.pkl",
            "type": "elasticnet",
            "target": "시간외석근인원수",
            "input": "총물량",
            "description": "인적자원 - 시간외석근 인원수 예측"
        },
        "workers6": {
            "path": "ElasticNet_workers6.pkl",
            "type": "elasticnet",
            "target": "시간외야근인원수",
            "input": "총물량",
            "description": "인적자원 - 시간외야근 인원수 예측"
        },
    }

    # 모델 타입 매핑
    MODEL_CLASSES: Dict[str, Type[BaseModel]] = {
        "elasticnet": ElasticNetModel,
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache: Dict[str, BaseModel] = {}
        return cls._instance

    def load_model(self, name: str, use_cache: bool = True) -> BaseModel:
        """모델 로드

        Args:
            name: 모델 이름
            use_cache: 캐시 사용 여부

        Returns:
            로드된 모델 인스턴스
        """
        # 캐시 확인
        if use_cache and name in self._cache:
            return self._cache[name]

        # 설정 확인
        config = self.MODEL_CONFIGS.get(name)
        if not config:
            raise ValueError(f"Unknown model: {name}. Available: {self.list_models()}")

        # 모델 클래스 가져오기
        model_class = self.MODEL_CLASSES.get(config["type"])
        if not model_class:
            raise ValueError(f"Unknown model type: {config['type']}")

        # 모델 경로 구성
        model_path = os.path.join(self._base_path, config["path"])

        # 모델 생성 및 로드
        model = model_class(name=name, model_path=model_path)
        model.load()

        # 추가 메타데이터
        model.metadata["target"] = config["target"]
        model.metadata["input"] = config["input"]
        model.metadata["description"] = config["description"]

        # 캐시에 저장
        if use_cache:
            self._cache[name] = model

        return model

    def list_models(self) -> List[str]:
        """등록된 모델 이름 목록 반환"""
        return list(self.MODEL_CONFIGS.keys())

    def get_model_config(self, name: str) -> Optional[Dict]:
        """모델 설정 반환"""
        return self.MODEL_CONFIGS.get(name)

    def register_model(
        self,
        name: str,
        path: str,
        model_type: str,
        target: str,
        input_col: str,
        description: str = ""
    ) -> None:
        """새 모델 등록

        Args:
            name: 모델 이름
            path: 모델 파일 경로
            model_type: 모델 타입 (elasticnet, prophet, etc.)
            target: 예측 대상 컬럼
            input_col: 입력 컬럼
            description: 모델 설명
        """
        self.MODEL_CONFIGS[name] = {
            "path": path,
            "type": model_type,
            "target": target,
            "input": input_col,
            "description": description
        }

    def clear_cache(self) -> None:
        """캐시 초기화"""
        self._cache.clear()

    def predict_all(self, input_value: float) -> Dict[str, float]:
        """모든 모델로 예측 수행

        Args:
            input_value: 입력값 (총물량)

        Returns:
            모델별 예측 결과
        """
        import numpy as np
        results = {}

        for name in self.list_models():
            try:
                model = self.load_model(name)
                prediction = model.predict(np.array([[input_value]]))[0]
                results[name] = {
                    "value": prediction,
                    "target": model.metadata.get("target", ""),
                }
            except Exception as e:
                results[name] = {"error": str(e)}

        return results
