import abc
import pandas as pd
import numpy as np

class BaseM5Model(abc.ABC):
    """
    Abstract base class for all forecasting models in the benchmarking framework.
    """
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the model variant."""
        pass
        
    @abc.abstractmethod
    def load(self, model_path: str = None) -> None:
        """
        Load the pre-trained model weights or configurations.
        """
        pass
        
    @abc.abstractmethod
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for the test set.
        
        Args:
            test_df: The aligned DataFrame containing features and true actual_sales.
            
        Returns:
            np.ndarray of predictions aligned with test_df.
        """
        pass
