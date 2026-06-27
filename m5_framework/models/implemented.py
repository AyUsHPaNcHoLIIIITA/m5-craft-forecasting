import numpy as np
import pandas as pd
from .base import BaseM5Model

class PrecomputedModel(BaseM5Model):
    """
    Helper class for models whose predictions have already been computed 
    during the expensive training pipeline. It assumes the test dataframe 
    contains a specific column with the predictions.
    
    A real production model would implement actual inference here, using 
    PyTorch/LightGBM APIs to score the test_df features.
    """
    def __init__(self, name: str, pred_column: str):
        self._name = name
        self.pred_column = pred_column
        
    @property
    def name(self) -> str:
        return self._name
        
    def load(self, model_path: str = None) -> None:
        # In this helper, loading is a no-op since predictions are in the df.
        pass
        
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.pred_column not in test_df.columns:
            raise ValueError(f"Expected column '{self.pred_column}' not found in test_df.")
        return test_df[self.pred_column].values

class TFTModel(PrecomputedModel):
    def __init__(self):
        super().__init__("TFT Baseline", "tft_pred")

class LGBMModel(PrecomputedModel):
    def __init__(self):
        super().__init__("LightGBM Baseline", "lgb_pred")

class FixedBlendModel(BaseM5Model):
    """
    A simple fixed blend baseline: 60% TFT, 40% LGBM.
    """
    @property
    def name(self) -> str:
        return "Fixed 60/40 Blend"
        
    def load(self, model_path: str = None) -> None:
        pass
        
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        tft_pred = test_df["tft_pred"].values
        lgb_pred = test_df["lgb_pred"].values
        return np.clip(0.6 * tft_pred + 0.4 * lgb_pred, 0, None)

class CraftZIGateModel(PrecomputedModel):
    def __init__(self):
        # By default this will use the ZI-Gate output
        super().__init__("CRAFT ZI-Gate", "craft_pred")

class OracleModel(BaseM5Model):
    """
    Theoretical upper bound (Oracle) that chooses the best prediction perfectly.
    """
    @property
    def name(self) -> str:
        return "Oracle (Upper Bound)"
        
    def load(self, model_path: str = None) -> None:
        pass
        
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        tft_pred = test_df["tft_pred"].values
        lgb_pred = test_df["lgb_pred"].values
        actual = test_df["actual_sales"].values
        
        tft_ae = np.abs(actual - tft_pred)
        lgb_ae = np.abs(actual - lgb_pred)
        
        return np.where(tft_ae <= lgb_ae, tft_pred, lgb_pred)
