## Experimental Code for drone tack (instructed by LLM)

$TASK_NAME = robust_PPO

python ../train.py --env quad3d_free --dynamics bodyrate --noDR --lower_controller base --obs_type quad --name $TASK_NAME
python ../test/vis.py --name $TASK_NAME