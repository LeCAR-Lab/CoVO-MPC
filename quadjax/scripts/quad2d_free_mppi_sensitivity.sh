for N in 2 4 8 16 32 64 128 256 512 1024 2048 4096
do
    echo "Running with N = $N"
    python ../envs/quad2d_free.py --mode eval --controller mppi --controller_params "N${N}_H40"
done

for H in 2 4 8 16 32 64
do
    echo "Running with N = $N"
    python ../envs/quad2d_free.py --mode eval --controller mppi --controller_params "N64_H${H}"
done