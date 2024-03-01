''' For a given method and run, plot the best feasible learned equations in the course of optimization
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml, json
import seaborn as sns

from CBOSS.utils.log_utils import gen_configuration_graph_2D

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
    plots = dict(
        NLDO = dict(
            resfiles = [
                results_folder / f'NonLinearDampedOscillator_k5/main/NonLinearDampedOscillator_k5_CBO_CHEI_KPOLYDIFF_BS2_20230526-145946_1_0.res.json',
                results_folder / f'NonLinearDampedOscillator_k5/main/NonLinearDampedOscillator_k5_CBO_FRCHEI_KPOLYDIFF_20230520-133840_1_0.res.json',
                results_folder / f'NonLinearDampedOscillator_k5/main/NonLinearDampedOscillator_k5_RS_20230529-210258_1_0.res.json',
                results_folder / f'NonLinearDampedOscillator_k5/main/NonLinearDampedOscillator_k5_SA_20230529-220108_1_0.res.json',
            ],
            vmax=-0.5, vmin=-1.0,
        )
    )

    
    ''' Generate image
    ========================= '''
    
    with open(config_file_path, 'r') as f:
        cfg = yaml.safe_load(f)


    ''' Plotting
    ========================= '''
    
    for plot in plots.values():
    
        for resfile in plot['resfiles']:
            
            # define folder where the images will be saved
            img_folder = resfile.parent.parent/'images'
            img_folder.mkdir(exist_ok=True, parents=True)
        
            # load result
            with open(results_folder / resfile, 'rb') as f:
                res = json.load(f)
                # convert all arrays to numpy
                res = {k: np.array(v) for k, v in res.items()}
        
        
            figname = img_folder/f"{rec_stem(resfile)}__graph"
            
            node_x, node_y, edge_x, edge_y = gen_configuration_graph_2D(X=res['X'])
            
            idx_failure = np.isnan(res['y'][:,0])
            
            fig = plt.figure(figsize=(3,3))
            plt.plot(edge_x, edge_y, c='k', alpha=0.5, linewidth=0.5,  zorder=0)
            plt.scatter(
                node_x, node_y, c=res['y'][:,0], 
                s=20, edgecolors='k', linewidth=0.2, 
                cmap='Reds_r', 
                vmin=plot['vmin'], vmax=plot['vmax'], zorder=1
            )
            plt.scatter(node_x[idx_failure], node_y[idx_failure], s=20, facecolor='k', edgecolors='k', linewidth=0.2, marker='s')
            # plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
            plt.axis('off')
            plt.tight_layout()
            fig.savefig(f'{figname}.png', dpi=200)
            fig.savefig(f'{figname}.svg')
            print(f'\tSaving: {figname}')
            plt.close(fig)
        
        
    