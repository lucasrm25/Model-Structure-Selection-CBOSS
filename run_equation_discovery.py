#!/usr/bin/env python3
'''
Author: Lucas Rath

Runs various constrained combinatorial optimization methods for structure identification of dynamical systems.

See --help for more information.
'''

def run_optimizers(
    exp_name:str,
    run_id:str,
    model_cfg:dict  = dict(),
    res_folder:str  = None,
    aim_active:bool = False,
    method_name:str = None,
    flags:str       = None
):
    ''' Run experiment defined by `cfg:dict`:
        ```yaml
        Lorenz_k3:
            model_name: Lorenz_Model
            n_init: 50
            evalBudget: 500
            model:
                random_seed_init_dataset: 2023
                kwargs:
                k: 3
                coeff_L1norm_thres: 100
        ```
    '''

    import sys, pathlib
    # sys.path.append( str(pathlib.Path(__file__).parent / 'CBOSS') )
    import numpy as np
    from pathlib import Path
    from typing import Callable, List, Tuple, Any
    import yaml
    import json
    import torch
    import re
    import traceback
    from CBOSS.optimizers.CBOSS import CBOSS
    from CBOSS.optimizers.random_sampler import random_sampler
    from CBOSS.optimizers.simulated_annealing import simulatedAnnealing
    from CBOSS.models import models
    from CBOSS.utils.Tee import Tee2
    from CBOSS.utils.python_utils import rget
    from CBOSS.utils.log_utils import (
        print_opt_progress,
        aim_track_opt_metrics, aim_plot_opt_progress, aim_plot_opt_graph, 
        aim_plot_equation_discovery_improvement
    ) 
    from CBOSS.utils.file_encoders import CompactJSONEncoder
    
    if aim_active:
        try:
            import aim
            aim_repo = aim.Repo(str(Path(__file__).parent))
        except Exception as e:
            aim_active = False
            print(f'Could not initialize AIM... skiping. \nError: {e}')


    ''' Parse Inputs
    ===================== '''
    
    method_name = method_name.upper()
    flags = flags.upper()

    assert method_name in ['RS', 'SA', 'CBOSS', 'PR', 'BOCS'], f'Unknown method: {method_name}'

    ''' Pre-Processing
    ===================== '''

    if res_folder is None:
        res_folder = Path(__file__).parent/'results'/f'{exp_name}'
    res_folder.mkdir(parents=True, exist_ok=True)

    # method_identifier = f'{exp_name}_{method}_{flags}_{run_id}'
    method_identifier = '_'.join([s for s in [exp_name,method_name,flags,run_id] if s != ''])

    print(f'\n========\n\tRUNNING {method_identifier} on {res_folder}\n===================\n')


    ''' Define Optimization Problem
    ================================= '''
    
    Model = getattr(models, model_cfg['model_name'])

    randomstate = np.random.RandomState(model_cfg['model']['random_seed_init_dataset'])
    model = Model(**model_cfg['model']['kwargs'], random_generator=randomstate)

    def evaluate_fun(X):
        y, c, l = model.evaluate(X)
        return y, c, l


    ''' Pre-evaluate model
    ======================== '''

    randomstate =  np.random.RandomState(model_cfg['model']['random_seed_init_dataset'])
    X = model.sample(model_cfg['n_init'], random_generator=randomstate).astype(int)
    y, c, l = evaluate_fun(X)

    print(f'initial y_min: {np.nanmin(np.where((c <= 0).all(1,keepdims=True), y, np.nan)):.4f}')


    class Log:
        def __init__(self):
            self.nbr_evals = 0
            if aim_active:
                self.aim_run = aim.Run(experiment=exp_name, repo=aim_repo)
                self.aim_run.name = method_name
                self.aim_run['model_cfg'] = model_cfg
            else:
                self.aim_run = None
            
        def log_funs(self, *args, **kw):
            ''' function handle for logging, called by the optimizers every iteration 
            '''
            log = kw.pop('log')
            iter_new_evals = np.arange(self.nbr_evals, log['X'].shape[0])
            print_opt_progress(*args, **kw, log=log, evalBudget=model_cfg['evalBudget'])
            if aim_active:
                kw['aim_run'] = self.aim_run
                new_log = {k: v[iter_new_evals] for k,v in log.items()}
                aim_track_opt_metrics(*args, **kw, log=new_log)
                aim_plot_opt_progress(*args, **kw, log=log)
                aim_plot_opt_graph(*args, **kw, log=log)
                # aim_plot_equation_discovery_improvement(*args, **kw, log=log, iter_new_evals=iter_new_evals, model=model, nbr_init_samples=model_cfg['n_init'])
            self.nbr_evals = log['X'].shape[0]

        def __del__(self):
            if aim_active:
                print('Closing logger')
                self.aim_run.close()

    logger = Log()

    def run_method(optimizer_fun:Callable[[Any],dict]) -> None:
        ''' Interface for starting experiment tracking (aim), calling optimizer and saving results
        '''
        try:
        # if True:        
            print(f'\n\n===============Starting {method_identifier}===============\n')
            with open( res_folder / f'{method_identifier}.cfg.yaml', 'w' ) as f:
                yaml.dump(model_cfg, f)

            # run optimization
            with (open(res_folder / f'{method_identifier}.log', 'w') as fstdout, Tee2(stdout_streams=[fstdout]) as t ):
                optimizer_log = optimizer_fun()
            
            print(f'\n\n===============Finished {method_identifier}===============\n')

        # except KeyboardInterrupt as e:
        except BaseException as e:
            print(f'{method_identifier} raised a {e.__class__.__name__} error\n{e}', file=sys.stderr)
            # dump results
            with open( res_folder / f'{method_identifier}.FAILED.log', 'w' ) as f:
                f.write(f'{method_identifier} raised {e.__class__.__name__} error:\n\n{traceback.format_exc()}')
        else:
            # dump results
            # with open( res_folder / f'{method_identifier}.log.pkl', 'wb' ) as f:
            #     pickle.dump( optimizer_log, file=f)
            
            # write results as JSON
            with open( res_folder / f'{method_identifier}.res.json', 'w' ) as f:
                json.dump( optimizer_log, f, ensure_ascii=False, indent=4, cls=CompactJSONEncoder)


    ''' Run optimizers
    ===================== '''
    if method_name == 'CBOSS':
        
        pflags = flags.upper().replace(' ','').split('_')
        assert len(pflags) == 3, f'Expected 3 flags for CBOSS, got {len(pflags)}: {pflags}'
        acq_fun, kernel, batchsize = pflags
        
        kernels = [k for k in ['POLY','DIFF'] if k in kernel]
        
        # parse batchsize with regex as 2 letters BS and a number
        batchsize = int(re.findall(r'BS(\d+)', batchsize)[0])
        
        run_method(
            optimizer_fun = lambda : (
                CBOSS(
                    eval_fun = evaluate_fun,
                    X = X, y = y, c = c, l = l,
                    product_space = model.product_space,
                    evalBudget = model_cfg['evalBudget'],
                    **rget(model_cfg, *['optimizers','CBOSS','kwargs']),
                    verbose = True,
                    double_precision = True,
                    batchsize = batchsize,
                    acq_fun = acq_fun,
                    kernels = kernels,
                    log_fun = lambda *args,**kw: logger.log_funs(*args, **kw)
                )
            )
        )
    if method_name == 'RS':
        run_method(
            optimizer_fun = lambda : (
                random_sampler(
                    eval_fun = evaluate_fun,
                    X = X, y = y, c = c, l = l,
                    product_space = model.product_space,
                    evalBudget = model_cfg['evalBudget'],
                    log_fun = lambda *args,**kw: logger.log_funs(*args, **kw)
                )
            )
        )
    if method_name == 'SA':
        run_method(
            optimizer_fun = lambda : (
                simulatedAnnealing(
                    eval_fun = evaluate_fun,
                    X = X, y = y, c = c, l = l,
                    product_space = model.product_space,
                    evalBudget = model_cfg['evalBudget'],
                    log_fun = lambda *args,**kw: logger.log_funs(*args, **kw)
                )
            )
        )
    if method_name == 'PR':
        from bo_pr.discrete_mixed_bo.run_one_replication import run_one_replication
        def PR_opt():
            res = run_one_replication(
                label = 'pr__ei',
                iterations = model_cfg['evalBudget'],
                function_name = 'equation_discovery',
                problem_kwargs = dict(model=model),
                batch_size = 1,
                mc_samples = 256,
                dtype = torch.double,
                device = "cpu",
                use_trust_region = False,
                X_init = torch.as_tensor(X),
                Y_init = -torch.hstack([ torch.as_tensor(y), torch.as_tensor(c)]),
            )
            opt_log = dict(
                X = res['X_all'].numpy(), 
                y = - res['Y_all'][:,0:1].numpy(), 
                c = - res['Y_all'][:,1:].numpy(), 
                l = (~torch.isnan(res['Y_all']).any(dim=1,keepdim=True)).numpy().astype(int),
                walltime=res['wall_time'].numpy()
            )
            return opt_log
        run_method(
            optimizer_fun = PR_opt
        )
    if method_name == 'BOCS':
        sys.path.append( str(pathlib.Path(__file__).parent / 'BOCS' / 'BOCSpy' ) )
        from BOCS import BOCS
        inputs = {
            'n_vars': len(model.product_space),
            'evalBudget': model_cfg['evalBudget'], # * 0 + 51,
            'model':   lambda x: model.evaluate(X=x[None,:])[0].squeeze(),
            'penalty': lambda x: 0.0,
            'x_vals': X,
            'y_vals': y[:,0],
        }
        def BOCS_opt():
            X_new, obj_iter = BOCS(
                inputs = inputs,
                order = 2,
                acquisitionFn = 'SA'
            )
            # evaluate samples
            y_new,c_new,l_new = evaluate_fun(X_new)
            opt_log = dict(
                X= np.concatenate([X, X_new], axis=0), 
                y= np.concatenate([y, y_new], axis=0),
                c= np.concatenate([c, c_new], axis=0),
                l= np.concatenate([l, l_new], axis=0)
            )
            return opt_log
        run_method(
            optimizer_fun = BOCS_opt
        )


import sys
from joblib import Parallel, delayed
import datetime
from tqdm import tqdm
from pathlib import Path
import yaml
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Equation Discovery')
    parser.add_argument(
        # '-e', '--exp', default='NonLinearDampedOscillator_k5', type=str,
        # '-e', '--exp', default='SEIR_k3', type=str,
        '-e', '--exp', default='CylinderWake_k3', type=str,
        # '-e', '--exp', default='Lorenz_k3', type=str,
        help='name of the optimization problem'
    )
    parser.add_argument(
        '-r', '--nbr_reruns',  default=1, type=int, 
        help='number of reruns'
    )
    parser.add_argument(
        '-j', '--nbr_jobs', default=1, type=int, 
        help='number of jobs'
    )
    parser.add_argument(
        '-m', '--method', default='CBOSS', type=str, 
        # '-m', '--method', default='RS', type=str, 
        # '-m', '--method', default='SA', type=str, 
        # '-m', '--method', default='PR', type=str, 
        # '-m', '--method', default='BOCS', type=str, 
        help='name of the optimization method'
    )
    parser.add_argument(
        '-f', '--flags', default='FRCHEI_KPOLYDIFF_BS2', type=str, 
        # '-f', '--flags', default='FRCEI_KPOLYDIFF_BS2', type=str, 
        help='flags for the optimization method'
    )
    parser.add_argument(
        # '-t', '--testname', default='main', type=str, 
        # '-t', '--testname', default='main_svgp', type=str,
        '-t', '--testname', default='main_test', type=str,
        help='name of this test run'
    )
    parser.add_argument(
        '-i', '--id', default=None, type=str,
        help='name of this test run'
    )
    parser.add_argument(
        '--aim', action=argparse.BooleanOptionalAction, default=False,
        help='include this flag for experiment tracking with aim'
    )
    args = parser.parse_args()

    # parse configurations
    with open(Path(__file__).parent/'configs_equation_discovery.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    #DONOTCOMMIT
    # cfg[args.exp]['evalBudget'] = 52

    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') if args.id is None else args.id

    with Parallel(n_jobs=args.nbr_jobs, prefer=["threads","processes"][1], backend='loky') as parallel:
        parallel(
            delayed(run_optimizers)(
                exp_name = args.exp,
                run_id = f'{run_id}_{run_nbr}',
                model_cfg = cfg[args.exp],
                aim_active = args.aim,
                method_name = args.method,
                flags = args.flags,
                res_folder = Path(__file__).parent/'results'/f'{args.exp}'/args.testname,
            )
            for run_nbr in tqdm(range(args.nbr_reruns))
        )
    print('finished')
