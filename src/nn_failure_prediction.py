from src.config import Config
from src.result import Result
from src.data import Data
from src.model import PredictiveModel
from src.visualization import Visualization


class NNFailurePrediction:
    def __init__(self, **kwargs):
        self.config = Config(**kwargs)
        self.result = Result()
        self.data = Data(self)
        self.predictive_model = PredictiveModel(self)
        self.visualization = Visualization(self)
