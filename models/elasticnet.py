"""ElasticNet Model Wrapper"""

import os
from typing import Optional
import numpy as np
import joblib
from sklearn.linear_model import ElasticNet

from .base import BaseModel


class ElasticNetModel(BaseModel):
    """ElasticNet 모델 래퍼"""

    def __init__(
        self,
        name: str,
        model_path: Optional[str] = None,
        alpha: float = 0.5,
        l1_ratio: float = 0.5
    ):
        super().__init__(name, model_path)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def load(self) -> None:
        """pkl 파일에서 모델 로드"""
        if not self.model_path:
            raise ValueError("model_path is required for loading")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        self._is_loaded = True

        # 메타데이터 추출
        self.metadata = {
            "coef": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
            "alpha": self.model.alpha,
            "l1_ratio": self.model.l1_ratio,
            "n_features": self.model.n_features_in_
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 수행

        Args:
            X: 입력 데이터 (2D array 또는 1D array)

        Returns:
            예측 결과
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # 1D 배열을 2D로 변환
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self.model.predict(X)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: Optional[float] = None,
        l1_ratio: Optional[float] = None,
        random_state: int = 0
    ) -> None:
        """모델 학습

        Args:
            X: 학습 입력 데이터
            y: 학습 타겟 데이터
            alpha: 정규화 강도
            l1_ratio: L1 비율
            random_state: 랜덤 시드
        """
        alpha = alpha or self.alpha
        l1_ratio = l1_ratio or self.l1_ratio

        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state
        )
        self.model.fit(X, y)
        self._is_loaded = True

        # 메타데이터 업데이트
        self.metadata = {
            "coef": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "n_features": self.model.n_features_in_
        }

    def save(self, path: str) -> None:
        """모델 저장

        Args:
            path: 저장 경로
        """
        if not self._is_loaded:
            raise RuntimeError("No model to save. Train or load a model first.")

        joblib.dump(self.model, path)
        self.model_path = path

    def get_formula(self) -> str:
        """예측 공식 문자열 반환"""
        if not self._is_loaded:
            return "Model not loaded"

        coef = self.metadata["coef"][0]
        intercept = self.metadata["intercept"]
        sign = "+" if intercept >= 0 else "-"
        return f"y = {coef:.8f} × x {sign} {abs(intercept):.4f}"
