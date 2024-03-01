''' Author: Lucas Rath

This script plots the learned equations from the BO experiments
It plots the learned equations for the best, worst, and median runs for each method and experiment.
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import yaml, json
import glob

from CBOSS.models import models



if __name__ == "__main__":

    config_file_path = Path(__file__).parent/'..'/'configs_equation_discovery.yaml'
    results_folder = Path(__file__).parent.parent / 'results'
    n_images = 4
    
    # create folder for saving figures
    img_folder = results_folder/'images'
    img_folder.mkdir(exist_ok=True, parents=True)
    
    
    exps = dict(
        NLDO    = dict(cfg='NonLinearDampedOscillator_k5', title='Nonlinear Damped Oscillator', ylim=[-0.95, -0.45], gl=['NonLinearDampedOscillator_k5/main*/']),
        SEIR    = dict(cfg='SEIR_k3',                      title='SEIR',                        ylim=[-1.6, 0.],     gl=['SEIR_k3/main*/']), 
        CylWake = dict(cfg='CylinderWake_k3',              title='Cylinder Wake',               ylim=[-1.3, 0.0],    gl=['CylinderWake_k3/main*/']), 
        Lorenz  = dict(cfg='Lorenz_k3',                    title='Lorenz Oscillator',           ylim=[-0.2, 0.0],    gl=['Lorenz_k3/main*/']), 
        # Chua    = dict(cfg='ChuaOscillator_k3',            title='Chua Oscillator',             ylim=[-0.18, -0.06], gl=['ChuaOscillator_k3/main*/']),
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

        Z_true = model.Z_true

        sns.set_style('darkgrid')
        plt.rcParams.update({
            'axes.facecolor': 'none',
            'lines.linewidth': 1.5,
            'grid.color': 'lightgray',
            'legend.facecolor': '#EAEAF2',
            'figure.constrained_layout.use': False
        })
        
        
        figname = f'{img_folder}/{exp_name}__measurement'
        
        alpha = 0.8

        fig = plt.figure(figsize=(3,2.5))
        if exp_name == 'NLDO':
            # plot 2d
            ax = fig.add_subplot(111)
            
            # ax.plot(*Z_true.T, label='measurement', alpha=alpha, color='tab:orange')
            
            from matplotlib.collections import LineCollection
            from matplotlib.colors import ListedColormap, BoundaryNorm
            norm = plt.Normalize(model.t_eval.min(), model.t_eval.max())
            points = np.array([Z_true[:,0], Z_true[:,1]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='jet', norm=norm)
            # Set the values used for colormapping
            lc.set_array(model.t_eval)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            
            limits = np.array([[Z_true[:,0].min(), Z_true[:,0].max()], [Z_true[:,1].min(), Z_true[:,1].max()]])
            ranges = limits[:,1] - limits[:,0]
            ax.set_xlim(limits[0,:] + 0.02*np.array([-1, 1])*ranges[0])
            ax.set_ylim(limits[1,:] + 0.02*np.array([-1, 1])*ranges[1])
            # ax.set_xlim([Z_true[:,0].min()*1.05, Z_true[:,0].max()*1.05] + np.array([]))
            # ax.set_ylim([Z_true[:,1].min()*1.05, Z_true[:,1].max()*1.05])
            ax.set_xlabel('x'), ax.set_ylabel('y')
            ax.tick_params(axis='x', colors='grey')
            ax.tick_params(axis='y', colors='grey')
            plt.tight_layout(rect=[-0.0, -0.05, 1.0, 1.])
            
            fig.savefig(f'{figname}.png', dpi=200)
            
        elif exp_name == 'SEIR':
            # plot 2d
            ax = fig.add_subplot(111)
            for i, (color, label, ls) in enumerate(zip(['tab:orange', 'tab:orange', 'tab:orange'], ['S', 'E', 'I'], ['-', '--', '-.'])):
                ax.plot(Z_true[:,i], label=label, alpha=alpha, color=color, linestyle=ls)
                # ax.annotate('local max', xy=(2, 1))
            ax.set_xlabel('time')
            # ax.set_ylabel('population')
            ax.tick_params(axis='x', colors='grey')
            ax.tick_params(axis='y', colors='grey')
            plt.tight_layout(rect=[-0.0, -0.05, 1.0, 1.])
            plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8)
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.plot3D(*Z_true.T, label='measurement', alpha=alpha, color='tab:orange')
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
