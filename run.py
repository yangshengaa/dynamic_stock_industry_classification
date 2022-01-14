"""
run modules

- 'gen': factor geneartor 
    - config in 'factor_generation/raw_factor/config.py'

- 'backtest': factor backtest 
    - config in 'backtest/configuration/config.py'

- 'comb': factor combination (ML)
    - config in 'factor_combination/configuration/config.py'

- 'opt': portfolio optimization 
    - config in 'portfolio_optimizer

- 'cluster': graph clustering 

# other tiny modules: 
- 'pairs': run pairs factor generation
    - config in 'factor_generation/raw_factor/pairs_modified.py'
"""

# load packages 
import sys

# =========================
# ------- module ----------
# =========================

def generator_factor():
    """ 
    run factor geneartion
    """
    from src.factor_generation.raw_factor.FactorGenerator import FactorGenerator

    fg = FactorGenerator()
    fg.run()

def backtest():
    """ 
    run backtest
    """
    pass 

def factor_combination():
    """
    run ml on factors 
    """
    pass 

def portfolio_optimization():
    """ 
    run portfolio optimization
    """
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
    if 'backtest' in targets:
        backtest()
    
    # ml factor combination 
    if 'comb' in targets:
        factor_combination()
    
    # Markowitz portfolio optimization
    if 'opt' in targets:
        portfolio_optimization()


    # ---------- side modules ------------
    # pairs 
    if 'pairs' in targets:
        run_pairs()

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
