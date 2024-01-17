import imageio
import numpy as np
import cv2

# load gif files
gif1 = imageio.mimread('/home/pcy/Research/code/quadjax/results/covo_vis.gif')
gif2 = imageio.mimread('/home/pcy/Research/code/quadjax/results/mppi_vis.gif')

title1 = 'CoVo-MPC'
title2 = 'MPPI'
# convert text to image
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255, 255, 255)
lineType = 2
for i in range(len(gif1)):
    cv2.putText(gif1[i], title1, (10, 20), font, fontScale, fontColor, lineType)
    cv2.putText(gif2[i], title2, (10, 20), font, fontScale, fontColor, lineType)
# add title to each frame


# concate each frame side by side
gif = []
for i in range(len(gif1)):
    gif.append(np.concatenate((gif1[i], gif2[i]), axis=1))

# save the concated gif
imageio.mimsave('/home/pcy/Research/code/quadjax/results/concat.gif', gif, fps=50)