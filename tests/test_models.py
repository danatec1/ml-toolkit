"""ML-Toolkit Model Tests"""

import sys
import os
import pytest
import numpy as np

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ModelRegistry, ElasticNetModel


class TestModelRegistry:
    """ModelRegistry 테스트"""

    def test_singleton(self):
        """싱글톤 패턴 확인"""
        r1 = ModelRegistry()
        r2 = ModelRegistry()
        assert r1 is r2

    def test_list_models(self):
        """모델 목록 조회"""
        registry = ModelRegistry()
        models = registry.list_models()
        assert len(models) == 7
        assert "workers" in models
        assert "pallets" in models

    def test_load_model(self):
        """모델 로드"""
        registry = ModelRegistry()
        model = registry.load_model("workers")
        assert model is not None
        assert model._is_loaded

    def test_predict_all(self):
        """전체 예측"""
        registry = ModelRegistry()
        results = registry.predict_all(200000)
        assert len(results) == 7
        assert "workers" in results
        assert "value" in results["workers"]


class TestElasticNetModel:
    """ElasticNetModel 테스트"""

    def test_load_and_predict(self):
        """로드 및 예측"""
        registry = ModelRegistry()
        model = registry.load_model("workers")

        result = model.predict(np.array([[200000]]))
        assert len(result) == 1
        assert result[0] > 0

    def test_get_info(self):
        """모델 정보 조회"""
        registry = ModelRegistry()
        model = registry.load_model("workers")
        info = model.get_info()

        assert "coef" in info
        assert "intercept" in info
        assert info["is_loaded"] is True

    def test_get_formula(self):
        """공식 조회"""
        registry = ModelRegistry()
        model = registry.load_model("workers")
        formula = model.get_formula()

        assert "y =" in formula
        assert "x" in formula


class TestPredictions:
    """예측 정확도 테스트"""

    def test_workers_prediction(self):
        """workers 모델 예측값 검증"""
        registry = ModelRegistry()
        model = registry.load_model("workers")

        # 총물량 200000 -> 약 64.71명
        result = model.predict(np.array([[200000]]))[0]
        assert 60 < result < 70

    def test_pallets_prediction(self):
        """pallets 모델 예측값 검증"""
        registry = ModelRegistry()
        model = registry.load_model("pallets")

        # 총물량 200000 -> 약 1093개
        result = model.predict(np.array([[200000]]))[0]
        assert 1000 < result < 1200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
