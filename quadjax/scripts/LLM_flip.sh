## Experimental Code for drone tack (instructed by LLM)

TASK_NAME="flip"

# python ../train.py --env quad3d_free --task horizon_flip --dynamics bodyrate --noDR --lower_controller base --obs_type quad --name $TASK_NAME
python ../train.py --env quad3d_free --task horizon_flip --dynamics free --noDR --lower_controller base --obs_type quad --name $TASK_NAME

# python ../test/vis.py --name $TASK_NAME