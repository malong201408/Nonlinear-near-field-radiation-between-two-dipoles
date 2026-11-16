#!/bin/bash
# ---------------------------------------------------------------------------
# ver41, single job.  Runs every stage that is still missing, then writes the
# integNL12 CSVs.  If it hits the time limit, just sbatch it again: finished
# R-rows are in figures_noF1/NL_rows/ and are skipped on the rerun.
#
# One node only: Mathematica's LaunchKernels[] starts subkernels on the node
# the master runs on, so the 2nd node of the old 2-node request sat idle.
# To use two nodes, see run_ver41_array.sh.
# ---------------------------------------------------------------------------
#SBATCH --job-name=RPPP_v41
#SBATCH --partition=snsm_itn19
#SBATCH --qos=snsm19
#SBATCH --output=output.%j
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8192
#SBATCH --mail-user=longm2@usf.edu

module purge
module load apps/mathematica/12.3

export NL_MODE=all
export NL_KERNELS=${SLURM_CPUS_PER_TASK:-16}

math < Code_nolinear_ver41_RPPP_Tem_order_TriFun_eta0015om0_M1.m
