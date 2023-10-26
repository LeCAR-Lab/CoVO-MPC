# experinment with noDR (no domain randomization)
for disturb in "none"
do
    # DATT
    python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller l1_esitimate_only --obs_type quad_l1 --name DATT --disturb_type $disturb

    # addictive DATT
    python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller l1 --obs_type quad_l1 --name A-DATT  --disturb_type $disturb

    # robust PPO
    python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller base --obs_type quad --name robust_PPO --disturb_type $disturb

    # expert PPO
    python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller base --obs_type quad_params --name expert_PPO --disturb_type $disturb

    # RMA
    python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller base --obs_type quad --RMA --name RMA --disturb_type $disturb


    # experinment with DR

    # DATT
    python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller l1_esitimate_only --obs_type quad_l1 --name DATT-DR --disturb_type $disturb

    # addictive DATT
    python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller l1 --obs_type quad_l1 --name A-DATT-DR --disturb_type $disturb

    # robust PPO
    python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller base --obs_type quad --name robust_PPO-DR --disturb_type $disturb

    # expert PPO
    python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller base --obs_type quad_params --name expert_PPO-DR --disturb_type $disturb

    # RMA
    python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller base --obs_type quad --RMA --name RMA-DR --disturb_type $disturb

    # RMA-L1
    python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller l1_esitimate_only --obs_type quad_l1 --RMA --name RMA-L1-DR --disturb_type $disturb

    # addictive RMA-L1
    python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller l1 --obs_type quad_l1 --RMA --name A-RMA-L1-DR --disturb_type $disturb
done