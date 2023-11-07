task=tracking_zigzag

# for controller in mppi mppi_zeji_mean mppi_zeji_lqr
# do
#     for N in 16 32 64 128 256 512
#     do
#         echo "Running with N = $N"
#         python ../envs/quad2d_free.py --task ${task} --dynamics bodyrate --controller ${controller} --mode eval --controller_params "N${N}_H16_lam0.01"
#     done
# done

N=8192
task=tracking_zigzag
for controller in mppi mppi_zeji_mean # mppi_zeji_zero mppi_zeji_mean mppi_zeji_lqr mppi_zeji_ppo
do 
    for H in 16
    do
        for lam in 0.03
        do
            echo "Running with H = $H"
            python ../envs/quad2d_free.py --task ${task} --dynamics bodyrate --controller ${controller} --mode eval --controller_params "N${N}_H${H}_lam${lam}"
        done
    done
done
