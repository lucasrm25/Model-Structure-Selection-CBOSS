''' For a given method and run, plot the best feasible learned equations in the course of optimization
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml, json
from matplotlib import gridspec
import seaborn as sns

from CBOSS.models import models

def rec_stem(path):
    stem = Path(path.stem)
    return stem.name if stem == path else rec_stem(path=stem)

def get_feas_improvement_idx(y:np.ndarray, c:np.ndarray):
    y_feas = y.copy()
    y_feas[ (c > 0).any(1) ] = np.nan
    y_best = np.fmin.accumulate(y_feas)
    iter_improvement = np.where( (np.roll(y_best,1)!=y_best) & ~np.isnan(y_best) )[0]
    return iter_improvement

if __name__ == "__main__":

    ''' User Configurations
    ========================= '''

    n_images = 4
    
    config_file_path = Path(__file__).parent/'..'/'configs_equation_discovery.yaml'
    results_folder = Path(__file__).parent.parent / 'results'
    
    # define the result file to plot
    exp_name = 'CylinderWake_k3'
    resfile = results_folder / f'{exp_name}/main/CylinderWake_k3_CBO_FRCEI_KPOLYDIFF_BS2_20230424-223005_1.res.json'

    # define folder where the images will be saved
    img_folder = resfile.parent.parent/'images'
    img_folder.mkdir(exist_ok=True, parents=True)
    
    
    ''' Generate image
    ========================= '''
    
    with open(config_file_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # init model
    cfg_exp = cfg[exp_name]
    model_Cls    = cfg_exp['model_name']
    model_kwargs = cfg_exp['model']['kwargs']
    Model = getattr(models, model_Cls)
    randomstate = np.random.RandomState(cfg_exp['model']['random_seed_init_dataset'])
    model = Model(**model_kwargs, random_generator=randomstate)


    # load result
    with open(results_folder / resfile, 'rb') as f:
        res = json.load(f)
        # convert all arrays to numpy
        res = {k: np.array(v) for k, v in res.items()}
    
    
    # get configurations x where there has been an improvement
    iter_improvement = get_feas_improvement_idx(res['y'], res['c'])
    
    # select only 3 values from array iter_improvement for plotting (first, last, and `n-2` in between)
    idx_selector = np.linspace(0, len(iter_improvement)-1, n_images, dtype=int)
    iter_improvement = iter_improvement[idx_selector]



    X = res['X'][iter_improvement] 
    y = res['y'][iter_improvement] 
    c = res['c'][iter_improvement] 


    # Fit coefficients
    Xi = model.fit_coefficients(X=X)	# <N,nm,n>
    # Simulate and evaluate
    Z_sim, success = model.simulate(Xi=Xi, method='batchode')
    # true simulation results
    Z_true = model.Z_true


    # unset seaborn style
    # sns.reset_orig()
        
        
    ''' Plotting
    ========================= '''
    
    for i, (xi, yi, ci, Z_sim_i, iteri) in enumerate(zip(X, y, c, Z_sim, iter_improvement)):
        
        figname = img_folder/f"{rec_stem(resfile)}__iter_{iteri}"
    
        fig = plt.figure(figsize=(3,2.5))
        if exp_name == 'NLDO':
            # plot 2d
            ax = fig.add_subplot(111)
            ax.plot(*Z_sim_i.T, label='simulation')
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
                ax.plot(Z_sim_i[:,i], label=f'{model.n_names[i]}_sim', linestyle='--', color=color)
                
            ax.set_xlabel('time'), ax.set_ylabel('population')
            ax.tick_params(axis='x', colors='grey')
            ax.tick_params(axis='y', colors='grey')
            plt.tight_layout(rect=[-0.0, -0.05, 1.0, 1.])
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.plot3D(*Z_sim_i.T, label='simulation')
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
    
    
    
    
    if False:
        figname = img_folder/f"{rec_stem(resfile)}__opt_progress"
        
        sns.set_style('darkgrid')

        plt.rcParams.update({
            'axes.facecolor': 'none',
            'lines.linewidth': 1.5,
            'grid.color': 'lightgray',
            'legend.facecolor': '#EAEAF2',
        })
    
        fig = plt.figure(figsize=(14,3))
        gs = gridspec.GridSpec(1, n_images)
        for i, (xi, yi, ci, Z_sim_i) in enumerate(zip(X, y, c, Z_sim)):
            ax = fig.add_subplot(gs[0,i], projection='3d')
            ax.plot3D(*Z_sim_i.T, label='simulation')
            ax.plot3D(*Z_true.T, label='measurement', alpha=0.5)
            # ax.set_title(f'x: {xi}  y: {yi[0]:.2f}  c: {ci[0]:.2f}\n')
            # ax.legend(loc='upper left')
            ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')

            # change the grid pane color to transparent
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            # ax.xaxis.set_pane_color('#EAEAF2')
            # ax.yaxis.set_pane_color('#EAEAF2')
            # ax.zaxis.set_pane_color('#EAEAF2')

            # change ticks color to grey
            ax.tick_params(axis='x', colors='grey')
            ax.tick_params(axis='y', colors='grey')
            ax.tick_params(axis='z', colors='grey')
            
            # ax.set_facecolor('#EAEAF2')
        # increase the lateral distance between subplots
        
        lines, labels = ax.get_legend_handles_labels()
        leg = fig.legend(
            lines, labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.1),
            ncol=2,
            title=None, frameon=True,
        )
        leg.set_in_layout(True)
        # tight layout with increased padding at the bottom
        fig.tight_layout(rect=[0, 0.05, 0.95, 1.2])
        # fig.tight_layout()
        plt.subplots_adjust(wspace=0.15)
        
        fig.savefig(f'{figname}.png', dpi=300)
        fig.savefig(f'{figname}.svg')
        plt.close()

