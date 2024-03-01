''' Author: Lucas Rath

This script plots the learned equations from the BO experiments
It plots the learned equations for the best, worst, and median runs for each method and experiment.
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml, json
import glob

from CBOSS.models import models
from CBOSS.utils.file_encoders import CompactJSONEncoder

def rec_stem(path):
    stem = Path(path.stem)
    return stem.name if stem == path else rec_stem(path=stem)

# get configurations x where there has been an improvement
def get_feas_improvement_idx(y:np.ndarray, c:np.ndarray):
    y_feas = y.copy()
    y_feas[ (c > 0).any(1) ] = np.nan
    y_best = np.fmin.accumulate(y_feas)
    iter_improvement = np.where( (np.roll(y_best,1)!=y_best) & ~np.isnan(y_best) )[0]
    return iter_improvement

def plot_simulation_trajectories(X, y, c, iter_improvement, fignames):
    sns.set_style('darkgrid')
    figs = model.plot(X=X)
    for x, yi, ci, it, fig, figname in zip(X, y, c, iter_improvement, figs, fignames):
        x_str = ''.join([str(xi) for xi in x.astype(int)])
        fig.suptitle(f'x: {x_str}  y: {yi[0]:.2f}  c: {ci[0]:.2f}\n')
        fig.set_size_inches(12,11)
        plt.tight_layout()
        fig.savefig(f'{figname}.png')
        print(f'\tSaving: {figname}')
        plt.close(fig)
    plt.close()


if __name__ == "__main__":

    config_file_path = Path(__file__).parent/'..'/'configs_equation_discovery.yaml'
    results_folder = Path(__file__).parent.parent / 'results'
    n_images = 4
    delete_old_images = True
    
    exps = dict(
        NLDO    = dict(cfg='NonLinearDampedOscillator_k5', title='Nonlinear Damped Oscillator', ylim=[-0.95, -0.45], gl=['NonLinearDampedOscillator_k5/main*/']),
        SEIR    = dict(cfg='SEIR_k3',                      title='SEIR',                        ylim=[-1.6, 0.],     gl=['SEIR_k3/main*/']), 
        CylWake = dict(cfg='CylinderWake_k3',              title='Cylinder Wake',               ylim=[-1.3, 0.0],    gl=['CylinderWake_k3/main*/']), 
        Lorenz  = dict(cfg='Lorenz_k3',                    title='Lorenz Oscillator',           ylim=[-0.2, 0.0],    gl=['Lorenz_k3/main*/']), 
        # Chua    = dict(cfg='ChuaOscillator_k3',            title='Chua Oscillator',             ylim=[-0.18, -0.06], gl=['ChuaOscillator_k3/main*/']),
    )
    methods = dict(
        RS      = dict(gl=['*_RS_*.res.json'],                                                plot_kwargs=dict(label='RS',     color='tab:red')),
        SA      = dict(gl=['*_SA_*.res.json'],                                                plot_kwargs=dict(label='SA',     color='tab:green')),
        PR      = dict(gl=['*_PR_*.res.json'],                                                plot_kwargs=dict(label='PR',     color='tab:purple')),
        CHEI    = dict(gl=['*CBO_CHEI_*_BS2_*.res.json',   '*CBO_CHEI_KPOLYDIFF_2023*.res.json'],   plot_kwargs=dict(label='CHEI',   color='tab:brown', linestyle='-')),
        FRCEI   = dict(gl=['*CBO_FRCEI_*_BS2_*.res.json',  '*CBO_FRCEI_KPOLYDIFF_2023*.res.json'],  plot_kwargs=dict(label='FRCEI',  color='tab:orange')),
        FRCHEI  = dict(gl=['*CBO_FRCHEI_*_BS2_*.res.json', '*CBO_FRCHEI_KPOLYDIFF_2023*.res.json'], plot_kwargs=dict(label='FRCHEI', color='tab:blue')),
    )

    with open(config_file_path, 'r') as f:
        cfg = yaml.safe_load(f)

    data = dict()
    for exp_name, exp in exps.items():

        data[exp_name] = dict()
        
        ''' Init model
        '''
        cfg_exp = cfg[exp['cfg']]
        model_Cls    = cfg_exp['model_name']
        model_kwargs = cfg_exp['model']['kwargs']
        Model = getattr(models, model_Cls)
        randomstate = np.random.RandomState(cfg_exp['model']['random_seed_init_dataset'])
        model = Model(**model_kwargs, random_generator=randomstate)

        
        for method_name, method in methods.items():
            print(f'processing: {exp_name} - {method_name}')
            
            data[exp_name][method_name] = dict()
            
            # skip all but SEIR
            # if exp_name != 'SEIR':# or method_name != 'SA':
            #     continue

            resfiles = []
            for gl in method['gl']:
                resfiles += sorted([Path(f) for method_dir in method['gl'] for exp_dir in exp['gl'] for f in glob.glob(str(results_folder / exp_dir / method_dir)) ])

            # read json files and extract data
            results = []
            for resfile in resfiles:
                with open(resfile, 'rb') as f:
                    res = json.load(f)
                    # convert all arrays to numpy
                    res = {k: np.array(v) for k, v in res.items()}
                    results += [res]
            
            if len(results) == 0:
                continue
            
            # calculate the best feasible y for all results
            y_min_feas = [
                np.nanmin( res['y'][ (res['c'] <= 0).all(1) ] )
                for res in results
            ]
            
            # only process best, worst and average results
            idx_best  = np.argmin(y_min_feas)
            idx_worst = np.argmax(y_min_feas)
            idx_avg   = np.argsort(y_min_feas)[len(y_min_feas)//2]


            selector = [
                dict(name = 'best',  idx=idx_best,  results=results[idx_best],  resfiles=resfiles[idx_best] ),
                dict(name = 'worst', idx=idx_worst, results=results[idx_worst], resfiles=resfiles[idx_worst] ),
                dict(name = 'avg',   idx=idx_avg,   results=results[idx_avg],   resfiles=resfiles[idx_avg] )
            ]
            for s in selector:
                resfile, res, name_flag = s['resfiles'], s['results'], s['name']

                iter_improvement = get_feas_improvement_idx(y=res['y'], c=res['c'])

                # select only 3 values from array iter_improvement for plotting (first, last, and `n-2` in between)
                idx_selector = np.linspace(0, len(iter_improvement)-1, n_images, dtype=int)
                iter_improvement = iter_improvement[idx_selector]
                # remove duplicates
                iter_improvement = np.unique(iter_improvement)

                # create folder for saving figures
                img_folder = resfile.parent.parent/'images'
                img_folder.mkdir(exist_ok=True, parents=True)


                ''' Plot best learned equations for different runs and methods - Standard plot
                ==============================================================='''

                # if delete_old_images:
                #     # delete images from the same result file `resfile` that already exist
                #     old_fignames = list((img_folder).glob(f'{rec_stem(resfile)}*.*'))
                #     for old_figname in old_fignames:
                #         print(f'\tDeleting: {old_figname.name}')
                #         old_figname.unlink()

                # define path for saving figures for each configuration
                fignames = [img_folder/f"{rec_stem(resfile)}__{name_flag}__iter_{idx:03d}" for idx in iter_improvement]

                X=res['X'][iter_improvement]
                y=res['y'][iter_improvement]
                c=res['c'][iter_improvement]

                # Fit coefficients
                Xi = model.fit_coefficients(X=X)	# <N,nm,n>
                # Simulate and evaluate
                Z_sim, success = model.simulate(Xi=Xi, method='batchode')
                Z_true = model.Z_true

                # store simulation results to do summary later
                data[exp_name][method_name][name_flag] = dict(
                    X=X, y=y, c=c,
                    Xi=Xi, Z_sim=Z_sim, success=success,
                    iter_improvement=iter_improvement,
                    fignames=fignames,
                )
            
                # save default model plots
                plot_simulation_trajectories( 
                    X=X, 
                    y=y, 
                    c=c, 
                    iter_improvement=iter_improvement, 
                    fignames=fignames 
                )

                ''' Plot best learned equations for different runs and methods
                ==============================================================='''
                
                sns.set_style('darkgrid')
                plt.rcParams.update({
                    'axes.facecolor': 'none',
                    'lines.linewidth': 1.5,
                    'grid.color': 'lightgray',
                    'legend.facecolor': '#EAEAF2',
                    'figure.constrained_layout.use': False
                })
                
                # plot only the best learned equation
                x = X[-1]
                yi = y[-1]
                ci = c[-1]
                Zsim_i = Z_sim[-1]
                iter_improvement_i = iter_improvement[-1]
                
                figname = f'{img_folder}/{exp_name}__{method_name}__{name_flag}__final'
                
                fig = plt.figure(figsize=(3,2.5))
                if exp_name == 'NLDO':
                    # plot 2d
                    ax = fig.add_subplot(111)
                    ax.plot(*Zsim_i.T, label='simulation')
                    ax.plot(*Z_true.T, label='measurement', alpha=0.5)
                    ax.set_xlabel('x'), ax.set_ylabel('y')
                    ax.tick_params(axis='x', colors='grey')
                    ax.tick_params(axis='y', colors='grey')
                    plt.tight_layout(rect=[-0.0, -0.05, 1.0, 1.])
                elif exp_name == 'SEIR':
                    # plot 2d
                    ax = fig.add_subplot(111)
                    for i, color in enumerate(['tab:blue', 'tab:purple', 'tab:green']):
                        ax.plot(Z_true[:,i], label=f'{model.n_names[i]}_meas', alpha=0.5, color='tab:orange')
                        ax.plot(Zsim_i[:,i], label=f'{model.n_names[i]}_sim', linestyle='--', color=color)
                        
                    ax.set_xlabel('time'), ax.set_ylabel('population')
                    ax.tick_params(axis='x', colors='grey')
                    ax.tick_params(axis='y', colors='grey')
                    plt.tight_layout(rect=[-0.0, -0.05, 1.0, 1.])
                else:
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot3D(*Zsim_i.T, label='simulation')
                    ax.plot3D(*Z_true.T, label='measurement', alpha=0.5)
                    ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                    ax.tick_params(axis='x', colors='grey')
                    ax.tick_params(axis='y', colors='grey')
                    ax.tick_params(axis='z', colors='grey')
                    plt.tight_layout(rect=[-0.1, 0, 0.9, 1.1])
                
                fig.savefig(f'{figname}.png', dpi=200)
                fig.savefig(f'{figname}.svg')
                print(f'\tSaving: {figname}')
                plt.close(fig)
                
                # save configuration used to generate the figure
                with open(f'{figname}.json', 'w') as f:
                    json.dump(dict(
                        figname=figname,
                        resfile=str(resfile),
                        exp_name=exp_name,
                        method_name=method_name,
                        name_flag=name_flag,
                        iter = iter_improvement_i,
                        x=x.tolist(),
                        y=yi.tolist(),
                        c=ci.tolist(),
                        # Zsim_i=Zsim_i.tolist(),
                        # Z_true=Z_true.tolist(),
                    ), f, indent=4, cls=CompactJSONEncoder)
        pass
    pass
