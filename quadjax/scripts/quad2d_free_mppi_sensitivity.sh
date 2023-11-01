task=tracking_zigzag

for controller in mppi
do
    for N in 2 4 8 16 32 64 128 256 512 1024
    do
        echo "Running with N = $N"
        python ../envs/quad2d_free.py --task ${task} --dynamics bodyrate --controller ${controller} --mode eval --controller_params "N${N}_H32_lam0.01"
    done
done

N=1024
for controller in mppi mppi_zeji_mean
do
    for H in 2 4 8 16 32 64 128
    do
        echo "Running with H = $H"
        python ../envs/quad2d_free.py --task ${task} --dynamics bodyrate --controller ${controller} --mode eval --controller_params "N${N}_H${H}_lam0.01"
    done
done