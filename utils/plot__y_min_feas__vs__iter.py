import numpy as np
import pandas as pd
import glob
from pathlib import Path
import yaml
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

if __name__ == "__main__":

    results_folder = Path(__file__).parent.parent / 'results'
    
    lw = 0.8

    data = dict(
        main = dict(
            name = 'yminfeas_wall_nbrfeas__x__iter__main', 
            exps = dict(
                NLDO    = dict(title='Nonlinear Damped Osc.', ylim=[-0.95, -0.45], gl=['NonLinearDampedOscillator_k5/main/']),
                SEIR    = dict(title='SEIR',                  ylim=[-1.6, 0.],     gl=['SEIR_k3/main/']), 
                CylWake = dict(title='Cylinder Wake',         ylim=[-1.3, 0.0],    gl=['CylinderWake_k3/main/']), 
                Lorenz  = dict(title='Lorenz Oscillator',     ylim=[-0.2, 0.0],    gl=['Lorenz_k3/main/']), 
            ),
            methods = dict(
                RS          = dict(gl=['*_RS_*.res.json'],                                                      plot_kwargs=dict(label='RS',         linewidth=lw, color='tab:red')),
                SA          = dict(gl=['*_SA_*.res.json'],                                                      plot_kwargs=dict(label='SA',         linewidth=lw, color='tab:green')),
                PR          = dict(gl=['*_PR_*.res.json'],                                                      plot_kwargs=dict(label='PR',         linewidth=lw, color='tab:purple')),
                FRCEI_bs2   = dict(gl=['*CBO_FRCEI_*_BS2_*.res.json'],                                          plot_kwargs=dict(label='CBOSS-FRCEI',  linewidth=lw, color='tab:orange')),
                FRCHEI_bs2  = dict(gl=['*CBO_FRCHEI_*_BS2_*.res.json', '*CBO_FRCHEI_KPOLYDIFF_2023*.res.json'], plot_kwargs=dict(label='CBOSS-FRCHEI', linewidth=lw, color='tab:blue')),
            ),
            plots = (True,True,True)
        ),
        ablation_variants = dict(
            name = 'yminfeas__x__iter__variants', 
            exps = dict(
                NLDO    = dict(title='Nonlinear Damped Oscillator', ylim=[-0.95, -0.45], gl=['NonLinearDampedOscillator_k5/main/']),
                SEIR    = dict(title='SEIR',                        ylim=[-1.6, 0.],     gl=['SEIR_k3/main/']), 
                CylWake = dict(title='Cylinder Wake',               ylim=[-1.3, 0.0],    gl=['CylinderWake_k3/main/']), 
                Lorenz  = dict(title='Lorenz Oscillator',           ylim=[-0.2, 0.0],    gl=['Lorenz_k3/main/']), 
            ),
            methods = dict(
                CHEI_bs2    = dict(gl=['*CBO_CHEI_*_BS2_*.res.json'],                                           plot_kwargs=dict(label='CBOSS-CHEI',   linewidth=lw, color='tab:orange')),
                FRCEI_bs2   = dict(gl=['*CBO_FRCEI_*_BS2_*.res.json'],                                          plot_kwargs=dict(label='CBOSS-FRCEI',  linewidth=lw, color='tab:green')),
                FRCHEI_bs2  = dict(gl=['*CBO_FRCHEI_*_BS2_*.res.json', '*CBO_FRCHEI_KPOLYDIFF_2023*.res.json'], plot_kwargs=dict(label='CBOSS-FRCHEI', linewidth=lw, color='tab:blue')),
            ),
            plots = (True,True,True)
        ),
        ablation_kernel = dict(
            name = 'yminfeas__x__iter__kernel', 
            exps = dict(
                NLDO    = dict(title='Nonlinear Damped Oscillator', ylim=[-0.95, -0.45], gl=['NonLinearDampedOscillator_k5/main/']),
                SEIR    = dict(title='SEIR',                        ylim=[-1.6, 0.],     gl=['SEIR_k3/main/']), 
                CylWake = dict(title='Cylinder Wake',               ylim=[-1.3, 0.0],    gl=['CylinderWake_k3/main/']), 
                Lorenz  = dict(title='Lorenz Oscillator',           ylim=[-0.22, 0.0],   gl=['Lorenz_k3/main/']), 
            ),
            methods = dict(
                FRCHEI_Kpolydiff  = dict(gl=['*CBO_FRCHEI_KPOLYDIFF_BS2_*.res.json', '*CBO_FRCHEI_KPOLYDIFF_2023*.res.json'], plot_kwargs=dict(label='CBOSS-FRCHEI_Kpolydiff', linewidth=lw, color='tab:blue')),
                FRCHEI_Kpoly      = dict(gl=['*CBO_FRCHEI_KDIFF_BS2_*.res.json',     '*CBO_FRCHEI_KDIFF_2023*.res.json'],     plot_kwargs=dict(label='CBOSS-FRCHEI_Kdiff',     linewidth=lw, color='tab:orange')),
                FRCHEI_Kdiff      = dict(gl=['*CBO_FRCHEI_KPOLY_BS2_*.res.json',     '*CBO_FRCHEI_KPOLY_2023*.res.json'],     plot_kwargs=dict(label='CBOSS-FRCHEI_Kpoly',     linewidth=lw, color='tab:green')),
            ),
            plots = (True,False,True)
        ),
        ablation_BS = dict(
            name = 'yminfeas__x__iter__BS',
            exps = dict(
                NLDO    = dict(title='Nonlinear Damped Oscillator', ylim=[-0.95, -0.45], gl=['NonLinearDampedOscillator_k5/main/']),
                SEIR    = dict(title='SEIR',                        ylim=[-1.6, 0.],     gl=['SEIR_k3/main/']), 
                CylWake = dict(title='Cylinder Wake',               ylim=[-1.3, 0.0],    gl=['CylinderWake_k3/main/']), 
                Lorenz  = dict(title='Lorenz Oscillator',           ylim=[-0.2, 0.0],    gl=['Lorenz_k3/main/']), 
            ),
            methods = dict(
                FRCHEI_bs1  = dict(gl=['*CBO_FRCHEI_KPOLYDIFF_BS1_*.res.json'],                                         plot_kwargs=dict(label='CBOSS-FRCHEI bs1', linewidth=lw, color='tab:orange')),
                FRCHEI_bs2  = dict(gl=['*CBO_FRCHEI_KPOLYDIFF_BS2_*.res.json', '*CBO_FRCHEI_KPOLYDIFF_2023*.res.json'], plot_kwargs=dict(label='CBOSS-FRCHEI bs2', linewidth=lw, color='tab:blue')),
                FRCHEI_bs4  = dict(gl=['*CBO_FRCHEI_KPOLYDIFF_BS4_*.res.json'],                                         plot_kwargs=dict(label='CBOSS-FRCHEI bs4', linewidth=lw, color='tab:green')),
            ),
            plots = (True,False,True)
        )
    )
    
    ''' Process data
        ==================
    '''
    
    for plot_name, plot_specs in data.items():
        exps = plot_specs['exps']
        methods = plot_specs['methods']
        print(f"Processing: {plot_specs['name']} - {plot_name}")

        y_feas_min     = dict()
        walltime       = dict()
        walltime_diff  = dict()
        nbr_feas       = dict()

        for exp_name, exp in exps.items():

            y_feas_min[exp_name] = dict()
            walltime[exp_name] = dict()
            walltime_diff[exp_name] = dict()
            nbr_feas[exp_name] = dict()

            for method_name, method in methods.items():

                resfiles = [f for method_dir in method['gl'] for exp_dir in exp['gl'] for f in glob.glob(str(results_folder / exp_dir / method_dir)) ]

                print(f"processing: {exp_name} - {method['gl']}")

                ''' Process all runs for this method
                '''
                logs = dict()
                for resfile in resfiles:
                    print(f'\tprocessing: {resfile}')

                    filename = Path(resfile).stem
                    with open(resfile, 'r') as f:
                        log = json.load(f)
                        
                    # convert all log entries to numpy arrays
                    for k, v in log.items():
                        if isinstance(v, list):
                            log[k] = np.array(v)

                    run_dir  = Path(resfile).parent
                    run_name = Path(resfile).with_suffix('').stem
                    cfgfile = (run_dir / run_name).with_suffix('.cfg.yaml')
                    with open(cfgfile, 'r') as f:
                        cfg = yaml.full_load(f)


                    def calc_y_feas_min(log):
                        y_min_feas = np.fmin.accumulate( np.where((log['c'] <= 0).all(1)[:,None], log['y'], np.nan) )
                        return y_min_feas

                    def calc_nbr_feas(log):
                        feas = np.cumsum( (log['c'] <= 0).all(1) & log['l'].all(1) )  / np.arange(1, len(log['l'])+1)
                        # feas = np.cumsum( log['l'].all(1) ) / np.arange(1, len(log['l'])+1)
                        return feas

                    def calc_walltime(log):
                        # walltime = np.ones(500) * np.nan
                        # if 'walltime' in log and log['walltime'] is not None and not (log['walltime'] == 0).all() and len(log['walltime']) > 0:
                        #     walltime[:log['walltime'].shape[0]] = log['walltime'][:500]
                        return log['walltime'] / 60.0
                    
                    def calc_walltime_diff(log):
                        # calculate walltime per iteration (difference between two wall times)
                        walltime_diff = np.diff(log['walltime']) / 60
                        # add a moving average filter of length 2
                        walltime_diff = np.convolve(walltime_diff, np.ones(2)/2, mode='valid')
                        return walltime_diff

                    logs[run_name] = dict(
                        cfg = cfg,
                        log = log,
                        stats = dict(
                            nbr_feas = calc_nbr_feas(log),
                            y_feas_min = calc_y_feas_min(log),
                            walltime = calc_walltime(log),
                            walltime_diff = calc_walltime_diff(log)
                        )
                    )
                    
                    # if (method_name == 'SA') and (exp_name == 'SEIR'):
                    #     idx_feas = (log['c'] <= 0).all(1)
                    #     log['y'][idx_feas][:50]

                ''' create dataframe for this method
                '''
                if len(resfiles) > 0:
                    y_feas_min[exp_name][method_name] = pd.DataFrame(
                        data = np.hstack([
                            np.asarray(log['stats']['y_feas_min'])
                            for modelname, log in logs.items()
                        ]),
                        columns = [
                            modelname
                            for modelname, log in logs.items()
                        ]
                    )
                    walltime[exp_name][method_name] = pd.DataFrame(
                        data = np.vstack([
                            np.asarray(log['stats']['walltime'])
                            for modelname, log in logs.items()
                        ]).T,
                        columns = [
                            modelname
                            for modelname, log in logs.items()
                        ]
                    )
                    walltime_diff[exp_name][method_name] = pd.DataFrame(
                        data = np.vstack([
                            np.asarray(log['stats']['walltime_diff'])
                            for modelname, log in logs.items()
                        ]).T,
                        columns = [
                            modelname
                            for modelname, log in logs.items()
                        ]
                    )
                    nbr_feas[exp_name][method_name] = pd.DataFrame(
                        data = np.vstack([
                            np.asarray(log['stats']['nbr_feas'])
                            for modelname, log in logs.items()
                        ]).T,
                        columns = [
                            modelname
                            for modelname, log in logs.items()
                        ]
                    )

        # if True:
        #     if True:

        ''' START PLOTTING
            ================================================
        '''
        plots = plot_specs['plots']

        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        sns.set_style('darkgrid')
        matplotlib.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.size': 8, 
            'axes.titlesize': 8,
            'legend.fontsize': 8, 
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'axes.titlepad': 5,
            'xtick.major.pad': 0,
            'ytick.major.pad': 0,
            # 'axes.labelpad': 5,
        })
        tight_layout_args = dict(
            rect=(0, 0.11, 1, 1), pad=0.0, w_pad=0.0, h_pad=0.5
        ) if sum(plots) == 2 else dict(
            rect=(0, 0.08, 1, 1), pad=0.0, w_pad=0.0, h_pad=0.0
        )
        legend_args = dict(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.),
            ncol=8,
            title=None, frameon=True,
        )

        h = 1.2
        fig, axs = plt.subplots( sum(plots), len(exps), figsize=(6.5, h*sum(plots)+0.3) )


        ''' Plot best feasible y over the number of iterations
            ==================================================
        ''' 
        # sns.set_style('darkgrid')
        # w,h = 4.0, 3.2
        # fig, axs = plt.subplots( 1, len(exps), figsize=(w*len(exps),h) )
        
        i_axs = 0

        if plots[0]:
            
            for i, ((exp_name, exp), ax) in enumerate(zip(exps.items(), axs[i_axs])): 

                # for method_name, y_feas_min_df in y_feas_min[exp_name].items():
                for method_name, method in methods.items():
                    
                    if method_name not in y_feas_min[exp_name]:
                        continue
                    y_feas_min_df = y_feas_min[exp_name][method_name]
                    
                    # for run in y_feas_min_df:
                    #     ax.plot(y_feas_min_df[run], alpha=0.2, **method['plot_kwargs'])
                    
                    ax.plot(y_feas_min_df.mean(1), **method['plot_kwargs'])
                    ax.fill_between( 
                        y_feas_min_df.index, 
                        y_feas_min_df.mean(1)-y_feas_min_df.sem(1), y_feas_min_df.mean(1)+y_feas_min_df.sem(1),
                        # alpha=0.3, color=method.get('plot_kwargs',{}).get('color',{})
                        color=method.get('plot_kwargs',{}).get('color',{}), alpha=0.3, linewidth=0.2
                    )
                ax.set_xlim([40,500])
                ax.set_xticks(np.arange(100, 501, 100))
                ax.set_ylim(exp['ylim'])
                
                # ax.set_xlabel('evaluations')
                if i == 0:
                    ax.set_ylabel(f'Best feasible\nobjective')
                ax.set_title(exp['title'])
                
            i_axs += 1

        ''' Nbr of feasible evaluations
            ==================================================
        ''' 
        if plots[1]:
        
            for i, ((exp_name, exp), ax) in enumerate(zip(exps.items(), axs[i_axs])): 

                # for method_name, walltime in walltime[exp_name].items():
                for method_name, method in methods.items():
                    pass
                    if method_name not in nbr_feas[exp_name]:
                        continue
                    nbr_feas_df = nbr_feas[exp_name][method_name]
                    
                    # for run in walltime_df:
                    #     ax.plot(walltime_df[run], alpha=0.2, **method['plot_kwargs'])
                    
                    ax.plot(nbr_feas_df.mean(1), **method['plot_kwargs'])
                    ax.fill_between( 
                        nbr_feas_df.index, 
                        nbr_feas_df.mean(1)-nbr_feas_df.sem(1), nbr_feas_df.mean(1)+nbr_feas_df.sem(1),
                        color=method.get('plot_kwargs',{}).get('color',{}), alpha=0.3, linewidth=0.2
                    )
                ax.set_xlim([40,500])
                ax.set_xticks(np.arange(100, 501, 100))
                ax.set_ylim([0,1])
                if i == 0:
                    ax.set_ylabel(f'Frequency of \nfeasible evaluations')

            i_axs += 1

        ''' Plot the wall time over the number of iterations
            ================================================
        '''
        
        if plots[2]:
            total_time = False
            for i, ((exp_name, exp), ax) in enumerate(zip(exps.items(), axs[i_axs])): 

                # for method_name, walltime in walltime[exp_name].items():
                for method_name, method in methods.items():
                    
                    if (method_name == 'RS') or (method_name == 'SA'):
                        continue
                    
                    if method_name not in walltime_diff[exp_name]:
                        continue
                    
                    walltime_df = walltime[exp_name][method_name] if total_time else walltime_diff[exp_name][method_name]
                    
                    # for run in walltime_df:
                    #     ax.plot(walltime_df[run], alpha=0.2, **method['plot_kwargs'])
                    
                    mu   = walltime_df.mean(1)
                    stde = walltime_df.sem(1)
                    idx  = mu >= 1e-8
                    
                    ax.plot( mu[idx], **method['plot_kwargs'])
                    ax.fill_between( 
                        walltime_df.index[idx], 
                        mu[idx]-stde[idx], mu[idx]+stde[idx],
                        # alpha=0.3, color=method.get('plot_kwargs',{}).get('color',{})
                        color=method.get('plot_kwargs',{}).get('color',{}), alpha=0.3, linewidth=0.2
                    )
                ax.set_xlim([40,500])
                ax.set_xticks(np.arange(100, 501, 100))
                ax.set_xlabel('Number of evaluations')
                # ax.set_ylim(exp['ylim'])
                
                # ax.set_xlabel('evaluations')
                if i == 0:
                    ax.set_ylabel('Total wall time [min]' if total_time else 'Wall time [min]')
                
                if not total_time:
                    ax.set_yscale('log')
                    ax.set_ylim([1e-1, 3e1])
                    ax.tick_params(axis='y', which='both', bottom=True)            
                    locmin = mticker.LogLocator(base=10, subs=np.arange(0.1,1,0.1), numticks=10)  
                    ax.yaxis.set_minor_locator(locmin)
                    ax.grid(True, which="minor", axis='y', ls="-")
                    
            i_axs += 1

        line_labels = {ll[1]:ll[0] for ax in fig.axes for ll in zip(*ax.get_legend_handles_labels()) if ll[1] != ''}
        # sort dictionary by the order in methods
        method_labels = {k: method['plot_kwargs']['label'] for k, method in methods.items()}
        line_labels = {k: line_labels[k] for k in method_labels.values() if k in line_labels.keys()}
        fig.legend(line_labels.values(), line_labels.keys(), **legend_args)
        
        fig.align_ylabels(axs)
        plt.tight_layout(**tight_layout_args)
        fig.savefig( results_folder / f"{plot_specs['name']}.pdf")
        fig.savefig( results_folder / f"{plot_specs['name']}.png")
        plt.close()
