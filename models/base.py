"""Base Model Abstract Class"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseModel(ABC):
    """모든 ML 모델의 베이스 클래스"""

    def __init__(self, name: str, model_path: Optional[str] = None):
        """
        Args:
            name: 모델 이름
            model_path: 모델 파일 경로
        """
        self.name = name
        self.model_path = model_path
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """모델 로드"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 수행

        Args:
            X: 입력 데이터 (2D array)

        Returns:
            예측 결과
        """
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """모델 학습

        Args:
            X: 학습 입력 데이터
            y: 학습 타겟 데이터
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """모델 저장

        Args:
            path: 저장 경로
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "path": self.model_path,
            "is_loaded": self._is_loaded,
            **self.metadata
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', loaded={self._is_loaded})"
