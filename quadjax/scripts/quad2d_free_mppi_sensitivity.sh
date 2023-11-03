task=tracking_zigzag

# for controller in mppi mppi_zeji_mean mppi_zeji_lqr
# do
#     for N in 16 32 64 128 256 512
#     do
#         echo "Running with N = $N"
#         python ../envs/quad2d_free.py --task ${task} --dynamics bodyrate --controller ${controller} --mode eval --controller_params "N${N}_H16_lam0.01"
#     done
# done

N=512
for controller in mppi mppi_zeji_mean mppi_zeji_lqr
do
    for H in 2 4 8 16 32 64
    do
        echo "Running with H = $H"
        python ../envs/quad2d_free.py --task ${task} --dynamics bodyrate --controller ${controller} --mode eval --controller_params "N${N}_H${H}_lam0.01"
    done
done