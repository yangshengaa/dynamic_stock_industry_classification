"""
run modules

major modules:
- 'gen': factor geneartor 
    - config in 'factor_generation/raw_factor/config.py'

- 'backtest_factor': factor backtest (continuous values on each cross section)
    - config in 'backtest/configuration/config.py'

- 'backtest_signal': signal backtest (0, 1, -1 on each cross section)
    - config in 'backtest/configuration/config.py'

- 'comb': factor combination (ML)
    - config in 'factor_combination/configuration/config.py'

- 'opt': portfolio optimization 
    - config in 'portfolio_optimizer

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

def generator_factor():
    """ run factor geneartion """
    from src.factor_generation.raw_factor.FactorGenerator import FactorGenerator

    fg = FactorGenerator()
    fg.run()

def backtest_factor():
    """ run factor backtest """
    from src.backtest.bin.batch_factor_test import run

    run()

def backtest_signal():
    """ run signal backtest """
    from src.backtest.bin.batch_signal_test import run

    run()

def factor_combination():
    """ run ml on factors """
    pass 

def portfolio_optimization():
    """ run portfolio optimization """
    pass 

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
    elif 'opt' in targets:
        portfolio_optimization()


    # ---------- side modules ------------
    # pairs 
    elif 'pairs' in targets:
        run_pairs()

    elif 'gen_risk' in targets:
        run_risk_factor_gen()
    
    else:
        raise NotImplementedError('Target not Found / Module not Defined')

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
