#!/usr/bin/zsh

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --job-name=CBO
#SBATCH --output=logs/output.%J.log
#SBATCH --time=5-00:00:00
#SBATCH --account=rwth1409
#SBATCH --array=1-10

### beginning of executable commands
source ~/anaconda3/bin/activate opt
conda env list

# get current timestamp
timestamp=$(date +"%Y%m%d-%H%M%S")


# experiment_name="NonLinearDampedOscillator_k5"
# experiment_name="Lorenz_k3"
# experiment_name="SEIR_k3"
experiment_name="CylinderWake_k3"


# ============ Main experiments ============ #
# python run_equation_discovery.py -e ${experiment_name} -j 1 -r 1 -m CBO -f FRCHEI_KPOLYDIFF_BS2 -i "${timestamp}_${SLURM_ARRAY_TASK_ID}"
# python run_equation_discovery.py -e ${experiment_name} -j 1 -r 1 -m CBO -f CHEI_KPOLYDIFF_BS2 -i "${timestamp}_${SLURM_ARRAY_TASK_ID}"
# python run_equation_discovery.py -e ${experiment_name} -j 1 -r 1 -m CBO -f FRCEI_KPOLYDIFF_BS2 -i "${timestamp}_${SLURM_ARRAY_TASK_ID}"

# ============ Ablation study: batch size ============ #
# python run_equation_discovery.py -e ${experiment_name} -j 1 -r 1 -m CBO -f FRCHEI_KPOLYDIFF_BS1 -i "${timestamp}_${SLURM_ARRAY_TASK_ID}"
# python run_equation_discovery.py -e ${experiment_name} -j 1 -r 1 -m CBO -f FRCHEI_KPOLYDIFF_BS4 -i "${timestamp}_${SLURM_ARRAY_TASK_ID}"

# ============ Ablation study: kernel type ============ #
# python run_equation_discovery.py -e ${experiment_name} -j 1 -r 1 -m CBO -f FRCHEI_KDIFF_BS2 -i "${timestamp}_${SLURM_ARRAY_TASK_ID}"
python run_equation_discovery.py -e ${experiment_name} -j 1 -r 1 -m CBO -f FRCHEI_KPOLY_BS2 -i "${timestamp}_${SLURM_ARRAY_TASK_ID}"

# ============ Baselines ============ #
# python run_equation_discovery.py -e ${experiment_name} -j 1 -r 1 -m RS -i "${timestamp}_${SLURM_ARRAY_TASK_ID}"
# python run_equation_discovery.py -e ${experiment_name} -j 1 -r 1 -m SA -i "${timestamp}_${SLURM_ARRAY_TASK_ID}"
# python run_equation_discovery.py -e ${experiment_name} -j 1 -r 1 -m PR -i "${timestamp}_${SLURM_ARRAY_TASK_ID}"


echo "job finished - task ${SLURM_ARRAY_TASK_ID} !!!"
