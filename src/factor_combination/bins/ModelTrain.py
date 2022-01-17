"""
main function for ML model combinations 
"""

# load packages
import warnings
warnings.filterwarnings('ignore')

# load file
from src.factor_combination.configuration import config as cfg
from src.factor_combination.tools.ModelCollection import (
    LinearModel,
    XgbModel,
    LgbModel,
    RandomForest
)


def run():
    """ main """
    # select model 
    model_selection = cfg.model_selection 

    # train 
    if "xgb" in model_selection:
        xgb = XgbModel()
        if cfg.nested:
            xgb.NestedXgbTrain()
        else:
            xgb.XgbTrain()

    elif "lgb" in model_selection:
        lgb = LgbModel()
        lgb.LgbTrain()

    elif "rf" in model_selection:
        rf = RandomForest()
        rf.RFTrain()

    elif "linear" in model_selection:
        lt = LinearModel(model_type = 'linear')
        lt.LinearTrain()

    else:
        raise NotImplementedError(f'{model_selection} DNE')
