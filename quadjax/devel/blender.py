import bpy
from bpy_extras.object_utils import object_data_add
import os
from mathutils import Quaternion, Vector

class AnimateRigidbody:

    def __init__(self, rigidbody_data, output_path):
        # Cleanup
        for item in bpy.data.objects:
            bpy.data.objects.remove(item)
        
        # Import crazyflie object
        bpy.ops.wm.obj_import(filepath="/home/pcy/Research/code/crazyswarm2-adaptive/src/crazyswarm2/crazyflie_sim/crazyflie_sim/visualization/data/model/cf.obj", directory="/home/pcy/Research/code/crazyswarm2-adaptive/src/crazyswarm2/crazyflie_sim/crazyflie_sim/visualization/data/model/", files=[{"name":"cf.obj", "name":"cf.obj"}])
        self.crazyflie = bpy.data.objects["cf"]
        self.crazyflie.rotation_mode = "QUATERNION"
        
        # Add camera and set attributes for a top down view
        camera_data = bpy.data.cameras.new(name='Camera')
        camera_object = object_data_add(bpy.context, camera_data)
        # Adjust these camera attributes according to your needs
        camera_object.location = Vector((0, 0, 10))
        camera_object.rotation_euler = Vector((0, 0, 1))
        camera_object.data.type = 'ORTHO'
        camera_object.data.ortho_scale = 3

        # Make this camera the active object
        bpy.context.scene.camera = camera_object
        
        # Add a light source for simple lighting
        light_data = bpy.data.lights.new(name='light', type='POINT')
        light_object = object_data_add(bpy.context, light_data)
        light_object.location = Vector((0, 0, 10))
        
        # Prepare the scene
        self.scene = bpy.context.scene
        self.fps = 30          # Set framerate (Modify as needed)
        self.scene.render.fps = self.fps
        self.scene.render.image_settings.file_format = "PNG"      
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.scene.render.filepath = self.output_path + "/"
        
        # Process and store the animation data
        self.animation_data = []
        for time, position, orientation in rigidbody_data:
            frame = time * self.fps      # Convert time to corresponding frame number
            self.animation_data.append((frame, position, orientation))

    def create_animation(self):
        for frame, position, orientation in self.animation_data:
            # Set the object's transform for every frame
            self.crazyflie.location = position
            self.crazyflie.rotation_quaternion = Quaternion(orientation)
            # Insert keyframes
            self.crazyflie.keyframe_insert(data_path="location", frame=frame)
            self.crazyflie.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        # Finally, render the animation
        # bpy.ops.render.render(animation=True)
        

# Data format: [(time, position, orientation),... ]
# replace the following lines with your own data. 
# The position is a tuple (x, y, z) and orientation is also a tuple (w, x, y, z) representing a quaternion.
rigidbody_data = [
    (0, (1, 1, 1), (1, 0, 0, 0)),
    (1, (2, 2, 2), (0, 1, 0, 0)),
    # Add more data points as needed
]

output_path = "/home/pcy/Research/code/quadjax/results/render"

animator = AnimateRigidbody(rigidbody_data, output_path)
animator.create_animation()
