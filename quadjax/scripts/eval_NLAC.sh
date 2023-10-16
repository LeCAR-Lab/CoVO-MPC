cd ../envs/

# disturb="drag" # periodic sin drag

for disturb in "drag" "periodic" "sin"
do    
    # NLAC
    echo "eval $disturb NLAC"
    python quad3d_free.py --dynamics bodyrate --noDR --lower_controller nlac_esitimate_only --obs_type quad_nlac --controller nn --task tracking_zigzag --controller_params ppo_params_NLAC --mode eval --disturb_type $disturb --name "NLAC_$disturb"

    # addictive NLAC
    python quad3d_free.py --dynamics bodyrate --noDR --lower_controller nlac --obs_type quad_nlac --controller nn --task tracking_zigzag --controller_params ppo_params_A-NLAC --mode eval --disturb_type $disturb --name "A-NLAC_$disturb"

    # NLAC
    python quad3d_free.py --dynamics bodyrate --lower_controller nlac_esitimate_only --obs_type quad_nlac --controller nn --task tracking_zigzag --controller_params ppo_params_NLAC-DR --mode eval --disturb_type $disturb --name "NLAC-DR_$disturb"

    # addictive NLAC
    python quad3d_free.py --dynamics bodyrate --lower_controller nlac --obs_type quad_nlac --controller nn --task tracking_zigzag --controller_params ppo_params_A-NLAC-DR --mode eval --disturb_type $disturb --name "A-NLAC-DR_$disturb"
done

cd ../scripts/