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

- 'cluster_train': train graph clustering
    - config in 'graph_cluster/config.py'

other modules: 
- 'pairs': run pairs factor generation
    - config in 'factor_generation/raw_factor/pairs_modified.py'

- 'gen_risk': risk factor generator 
    - config in 'factor_generation/raw_factor/style_factor_config.py'
"""

# load packages 
import sys
import argparse

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
    
    # accept arguments for meta control 
    parser = argparse.ArgumentParser(description='portfolio optimization factor return estimation config')
    parser.add_argument('--use_dynamic_ind', type=bool, default=None, help='whether to use dynamic industry')
    parser.add_argument('--dynamic_ind_name', default=None, help='name of the dynamic industry')
    args, _ = parser.parse_known_args()

    # init 
    loading_process = FactorReturnGenerator()
    # change config 
    if not args.use_dynamic_ind is None: 
        loading_process.use_dynamic_ind = args.use_dynamic_ind
    if not args.dynamic_ind_name is None:
        loading_process.dynamic_ind_name = args.dynamic_ind_name
    # run 
    loading_process.start_cal_return_process()

def portfolio_optimization_cov_est():
    """ run covariance estimation """
    from src.portfolio_optimization.CovMatrixEstimator import CovMatrixEstimator

    # accept arguments for meta control 
    parser = argparse.ArgumentParser(description='portfolio optimization factor return estimation config')
    parser.add_argument('--use_dynamic_ind', type=bool, default=None, help='whether to use dynamic industry')
    parser.add_argument('--dynamic_ind_name', default=None, help='name of the dynamic industry')
    args, _ = parser.parse_known_args()

    # init 
    calculating_process = CovMatrixEstimator()
    # change config 
    if not args.use_dynamic_ind is None:
        calculating_process.use_dynamic_ind = args.use_dynamic_ind
    if not args.dynamic_ind_name is None: 
        calculating_process.dynamic_ind_name = args.dynamic_ind_name
    # run
    calculating_process.start_cal_cov_process()

def portfolio_optimization_weight():
    """ adjust weight """
    from src.portfolio_optimization.WeightOptimizer import WeightOptimizer

    calculating_process = WeightOptimizer()
    calculating_process.start_weight_optimize_process()

# ----- graph clustering ------
def graph_clustering_train():
    """ train graph clusters """
    from src.graph_cluster.IndustryTrainer import IndustryTrainer, MultiIndustryTrainer

    parser = argparse.ArgumentParser(description="graph clustering config")
    # single
    parser.add_argument('--graph_type', default=None, choices=['AG', 'MST', 'PMFG'], help='type of graph')
    parser.add_argument('--num_clusters', type=int, default=None, help='number of clusters')
    parser.add_argument('--clustering_type', default=None, choices=['single_linkage', 'spectral', 'node2vec', 'sub2vec'], help='type of clustering')
    parser.add_argument('--filter_mode', default=None, type=int, choices=[0, 1, 2], help='filter noise mode')
    # multi 
    parser.add_argument('--use_multi', type=bool, default=False, choices=[True, False], help='whether to train multiple labels at a time')
    parser.add_argument('--multi_num_clusters', 
        type=int,
        nargs='+', 
        default=[5, 10, 20, 30, 40, 60], 
        help='multiple num clusters'
    )
    parser.add_argument('--multi_clustering_type', 
        type=str,
        nargs='+',
        default=['spectral', 'node2vec', 'sub2vec'],
        help='multiple clustering type'
    )
    args, _ = parser.parse_known_args()

    if not args.use_multi: 
        industry_trainer = IndustryTrainer(
            graph_type=args.graph_type,
            num_clusters=args.num_clusters,
            clustering_type=args.clustering_type,
            filter_mode=args.filter_mode
        )
        industry_trainer.run()
    else:
        industry_trainer = MultiIndustryTrainer(
            graph_type=args.graph_type,
            num_clusters=args.multi_num_clusters,
            clustering_type=args.multi_clustering_type,
            filter_mode=args.filter_mode
        )
        industry_trainer.run()

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

    # graph clustering 
    elif 'cluster_train' in targets:
        graph_clustering_train()
    
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
                'cluster_train',
                'pairs',
                'gen_risk'
            )
        )

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
