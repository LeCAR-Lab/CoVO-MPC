#!/bin/bash

# Define local and remote directories
local_dir="/Users/pcy/Desktop/Research/quadjax/quadjax/"
remote_dir="pcy@lecar-legion-t7-01.wifi.local.cmu.edu:/home/pcy/Research/quadjax/quadjax"

# Use scp to copy the local directory to the remote server
scp -r $local_dir $remote_dir