"""
run framework
- 'gen': factor geneartor 
- 'backtest': factor backtest 
- 'comb': factor combination (ML)
- 'opt': portfolio optimization 
- 'cluster': graph clustering 
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
    from src.factor_generator.raw_factor.FactorGenerator import FactorGenerator

    fg = FactorGenerator()
    fg.run()

def backtest():
    """ 
    run backtest
    """

# =======================
# ------ main -----------
# =======================

def main(targets):
    """ 
    run modules 
    """
    if 'gen' in targets:
        generator_factor()


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
