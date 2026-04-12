for script in mv_exp/eval_scripts/*.sh; do
    echo "Submitting $script"
    sbatch "$script"
done
