for script in mv_exp/batch_size_discovery_scripts/*.sh; do
    echo "Submitting $script"
    sbatch "$script"
done
