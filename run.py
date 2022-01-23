"""
run modules

major modules:

Factor Generation:

- 'gen': factor geneartor 
    - config in 'factor_generation/raw_factor/config.py'

Backtest: 

- 'backtest_factor': factor backtest (continuous values on each cross section)
    - config in 'backtest/configuration/config.py'

- 'backtest_signal': signal backtest (0, 1, -1 on each cross section)
    - config in 'backtest/configuration/config.py'

Factor Combination 

- 'comb': factor combination (ML)
    - config in 'factor_combination/configuration/config.py'

Portfolio Optimization

- 'opt_fac_ret': generate risk factor returns 
    - config in 'portfolio_optimizer/config.py'

- 'opt_cov_est': covariance estimation 
    - config in 'portfolio_optimizer/config.py'

- 'opt_weight': optimize weight
    - config in 'portfolio_optimizer/config.py'


Graph Clustering 

- 'cluster': graph clustering 

other modules: 
- 'pairs': run pairs factor generation
    - config in 'factor_generation/raw_factor/pairs_modified.py'

- 'gen_risk': risk factor generator 
    - config in 'factor_generation/raw_factor/style_factor_config.py'
"""

# load packages 
import sys

# =========================
# ------- module ----------
# =========================

# ----- factor gen -------
def generator_factor():
    """ run factor geneartion """
    from src.factor_generation.raw_factor.FactorGenerator import FactorGenerator

    fg = FactorGenerator()
    fg.run()

# ------ backtest --------
def backtest_factor():
    """ run factor backtest """
    from src.backtest.bin.batch_factor_test import run

    run()

def backtest_signal():
    """ run signal backtest """
    from src.backtest.bin.batch_signal_test import run

    run()


# ------ factor combination ------
def factor_combination():
    """ run ml on factors """
    from src.factor_combination.bins.ModelTrain import run

    run()

# ----- portfolio optimization ------
def portfolio_optimization_fac_ret():
    """ run factor return generation """
    from src.portfolio_optimization.FactorReturnGenerator import FactorReturnGenerator

    loading_process = FactorReturnGenerator()
    loading_process.start_cal_return_process()

def portfolio_optimization_cov_est():
    """ run covariance estimation """
    from src.portfolio_optimization.CovMatrixEstimator import CovMatrixEstimator

    calculating_process = CovMatrixEstimator()
    calculating_process.start_cal_cov_process()

def portfolio_optimization_weight():
    """ adjust weight """
    from src.portfolio_optimization.WeightOptimizer import WeightOptimizer

    calculating_process = WeightOptimizer()
    calculating_process.start_weight_optimize_process()


# =======================
# ------ others ---------
# =======================

def run_pairs():
    """
    run pairs factor generation
    """
    from src.factor_generation.raw_factor.pairs_modified import run

    run()

def run_risk_factor_gen():
    """
    generate risk factors
    """
    from src.factor_generation.raw_factor.StyleFactorGenerator import StyleFactorGenerator

    loading_process = StyleFactorGenerator()
    loading_process.start_loading_data_process()

# =======================
# ------ main -----------
# =======================

def main(targets):
    """ 
    run modules 
    """
    # --------- main modules ----------
    # factor generator 
    if 'gen' in targets:
        generator_factor()
    
    # backtest
    elif 'backtest_factor' in targets:
        backtest_factor()

    elif 'backtest_signal' in targets:
        backtest_signal()
    
    # ml factor combination 
    elif 'comb' in targets:
        factor_combination()
    
    # Markowitz portfolio optimization
    elif 'opt_cov_est' in targets:
        portfolio_optimization_cov_est()
    
    elif 'opt_fac_ret' in targets:
        portfolio_optimization_fac_ret()

    elif 'opt_weight' in targets:
        portfolio_optimization_weight()


    # ---------- side modules ------------
    # pairs 
    elif 'pairs' in targets:
        run_pairs()

    elif 'gen_risk' in targets:
        run_risk_factor_gen()
    
    else:
        raise NotImplementedError(
            'Target not Found / Module not Defined. Please pick from the following modes: \n' + 
            '\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n'.format(
                'gen',
                'backtest_factor',
                'backtest_signal',
                'comb',
                'opt_cov_est',
                'opt_fac_ret',
                'opt_weight',
                'pairs',
                'gen_risk'
            )
        )

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
