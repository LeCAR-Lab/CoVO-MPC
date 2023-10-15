# experinment with noDR (no domain randomization)

# DATT
python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller l1_esitimate_only --obs_type quad_l1 --name DATT 

# addictive DATT
python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller l1 --obs_type quad_l1 --name A-DATT 

# robust PPO
python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller base --obs_type quad --name robust_PPO

# expert PPO
python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller base --obs_type quad_params --name expert_PPO

# RMA
python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller base --obs_type quad --RMA --name RMA


# experinment with DR

# DATT
python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller l1_esitimate_only --obs_type quad_l1 --name DATT-DR

# addictive DATT
python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller l1 --obs_type quad_l1 --name A-DATT-DR

# robust PPO
python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller base --obs_type quad --name robust_PPO-DR

# expert PPO
python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller base --obs_type quad_params --name expert_PPO-DR

# RMA
python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller base --obs_type quad --RMA --name RMA-DR

# RMA-L1
python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller l1_esitimate_only --obs_type quad_l1 --RMA --name RMA-L1-DR

# addictive RMA-L1
python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller l1 --obs_type quad_l1 --RMA --name A-RMA-L1-DR