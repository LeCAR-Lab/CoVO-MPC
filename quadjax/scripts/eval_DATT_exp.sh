cd ../envs/

# disturb="drag" # periodic sin drag

for disturb in "drag" "periodic" "sin"
do
    # DATT
    python quad3d_free.py --dynamics bodyrate --noDR --lower_controller l1_esitimate_only --obs_type quad_l1 --controller nn --task tracking_zigzag --controller_params rma/ppo_params_DATT --mode eval --disturb_type $disturb --name "DATT_$disturb"

    # addictive DATT
    python quad3d_free.py --dynamics bodyrate --noDR --lower_controller l1 --obs_type quad_l1 --controller nn --task tracking_zigzag --controller_params rma/ppo_params_A-DATT --mode eval --disturb_type $disturb --name "A-DATT_$disturb"

    # robust PPO
    python quad3d_free.py --dynamics bodyrate --noDR --lower_controller base --obs_type quad --controller nn --task tracking_zigzag --controller_params rma/ppo_params_robust_PPO --mode eval --disturb_type $disturb --name "robust_PPO_$disturb"

    # expert PPO
    python quad3d_free.py --dynamics bodyrate --noDR --lower_controller base --obs_type quad_params --controller nn --task tracking_zigzag --controller_params rma/ppo_params_expert_PPO --mode eval --disturb_type $disturb --name "expert_PPO_$disturb"

    # RMA
    python quad3d_free.py --dynamics bodyrate --noDR --lower_controller base --obs_type quad --controller RMA --task tracking_zigzag --controller_params rma/ppo_params_RMA --mode eval --disturb_type $disturb --name "RMA_$disturb"


    # DATT
    python quad3d_free.py --dynamics bodyrate --lower_controller l1_esitimate_only --obs_type quad_l1 --controller nn --task tracking_zigzag --controller_params rma/ppo_params_DATT-DR --mode eval --disturb_type $disturb --name "DATT-DR_$disturb"

    # addictive DATT
    python quad3d_free.py --dynamics bodyrate --lower_controller l1 --obs_type quad_l1 --controller nn --task tracking_zigzag --controller_params rma/ppo_params_A-DATT-DR --mode eval --disturb_type $disturb --name "A-DATT-DR_$disturb"

    # robust PPO
    python quad3d_free.py --dynamics bodyrate --lower_controller base --obs_type quad --controller nn --task tracking_zigzag --controller_params rma/ppo_params_robust_PPO-DR --mode eval --disturb_type $disturb --name "robust_PPO-DR_$disturb"

    # expert PPO
    python quad3d_free.py --dynamics bodyrate --lower_controller base --obs_type quad_params --controller nn --task tracking_zigzag --controller_params rma/ppo_params_expert_PPO-DR --mode eval --disturb_type $disturb --name "expert_PPO-DR_$disturb"

    # RMA
    python quad3d_free.py --dynamics bodyrate --lower_controller base --obs_type quad --controller RMA --task tracking_zigzag --controller_params rma/ppo_params_RMA-DR --mode eval --disturb_type $disturb --name "RMA-DR_$disturb"

    # RMA-L1
    python quad3d_free.py --dynamics bodyrate --lower_controller l1_esitimate_only --obs_type quad_l1 --controller RMA --task tracking_zigzag --controller_params rma/ppo_params_RMA-L1-DR --mode eval --disturb_type $disturb --name "RMA-L1-DR_$disturb"

    # addictive RMA-L1
    python quad3d_free.py --dynamics bodyrate --lower_controller l1 --obs_type quad_l1 --controller RMA --task tracking_zigzag --controller_params rma/ppo_params_A-RMA-L1-DR --mode eval --disturb_type $disturb --name "A-RMA-L1-DR_$disturb"
done

cd ../scripts/