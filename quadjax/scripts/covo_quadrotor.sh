for task in tracking_zigzag; do
    for controller in mppi covo_online covo_offline; do
        for N in 8192; do
            for H in 32; do
                for lam in 0.01; do
                    echo "Running with H = $H, lam = $lam, N = $N, controller = $controller, task = $task"
                    python ../envs/quad3d_free.py --task ${task} --dynamics bodyrate --controller ${controller} --mode eval --controller_params "N${N}_H${H}_lam${lam}" --name "quad3d_${task}_${controller}_N${N}_H${H}_lam${lam}" --noDR
                done
            done
        done
    done
done