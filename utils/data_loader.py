"""Data Loader Utility"""

import os
from typing import Optional, Tuple
import pandas as pd
import numpy as np


class DataLoader:
    """데이터 로딩 유틸리티"""

    @staticmethod
    def load_csv(
        file_path: str,
        input_col: str = "총물량",
        target_col: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
        """CSV 파일 로드

        Args:
            file_path: CSV 파일 경로
            input_col: 입력 컬럼명
            target_col: 타겟 컬럼명 (없으면 None)

        Returns:
            (X, y, df) 튜플
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        if input_col not in df.columns:
            raise ValueError(f"Column '{input_col}' not found in CSV. Available: {df.columns.tolist()}")

        X = df[[input_col]].to_numpy()

        y = None
        if target_col:
            if target_col not in df.columns:
                raise ValueError(f"Column '{target_col}' not found in CSV")
            y = df[target_col].to_numpy()

        return X, y, df

    @staticmethod
    def validate_data(X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """데이터 검증 및 전처리

        Args:
            X: 입력 데이터
            y: 타겟 데이터

        Returns:
            검증된 (X, y) 튜플
        """
        # NaN 제거
        mask = ~np.isnan(X).any(axis=1)

        if y is not None:
            mask &= ~np.isnan(y)
            y = y[mask]

        X = X[mask]

        # 음수 값 제거 (물량은 0 이상이어야 함)
        mask = (X > 0).all(axis=1)
        X = X[mask]

        if y is not None:
            y = y[mask]

        return X, y

    @staticmethod
    def get_summary(df: pd.DataFrame) -> dict:
        """데이터프레임 요약 정보"""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing": df.isnull().sum().to_dict()
        }
