# Define local and remote directories
local_dir="/Users/pcy/Desktop/Research/quadjax/results/"
remote_dir="pcy@lecar-legion-t7-01.wifi.local.cmu.edu:/home/pcy/Research/quadjax/results/*"

# Use scp to copy the local directory to the remote server
scp $remote_dir $local_dir 