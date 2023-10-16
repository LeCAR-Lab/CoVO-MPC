# for disturb in "drag" "periodic" "sin"
# do
#     python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller nlac_esitimate_only --obs_type quad_nlac --disturb_type $disturb --name "NLAC_$disturb"

#     python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller nlac --obs_type quad_nlac --disturb_type $disturb --name "A-NLAC_$disturb"

#     python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller nlac_esitimate_only --obs_type quad_nlac --disturb_type $disturb --name "NLAC-DR_$disturb"

#     python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller nlac --obs_type quad_nlac --disturb_type $disturb --name "A-NLAC-DR_$disturb"
# done
python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller nlac_esitimate_only --obs_type quad_nlac --name "NLAC"

python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller nlac --obs_type quad_nlac --name "A-NLAC"

python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller nlac_esitimate_only --obs_type quad_nlac --name "NLAC-DR"

python ../train.py --env quad3d_free --dynamics bodyrate --lower_controller nlac --obs_type quad_nlac --name "A-NLAC-DR"
