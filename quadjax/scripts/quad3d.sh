task=tracking_zigzag

# for controller in mppi mppi_zeji_mean mppi_zeji_lqr
# do
#     for N in 16 32 64 128 256 512
#     do
#         echo "Running with N = $N"
#         python ../envs/quad2d_free.py --task ${task} --dynamics bodyrate --controller ${controller} --mode eval --controller_params "N${N}_H16_lam0.01"
#     done
# done
export JAX_DEBUG_NANS=True
for task in tracking_zigzag; do
    for controller in mppi mppi_zeji_zero mppi_zeji_pid mppi_zeji_mean; do
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
export JAX_DEBUG_NANS=False