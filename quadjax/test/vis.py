import meshcat
from  meshcat import geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation, convert_frames_to_video
import numpy as np
import time
import pickle

# Create a Meshcat visualizer
vis = meshcat.Visualizer()
anim = Animation(default_framerate=50)

# def vis_vector(obj, origin, vec, scale=1.0):
#     # visualize the force with arrow
#     vec_norm = np.linalg.norm(vec)
#     if vec_norm == 0:
#         return
#     vec = vec / vec_norm
#     # gernerate two unit vectors perpendicular to the force vector
#     if vec[0] == 0 and vec[1] == 0:
#         vec_1 = np.array([1, 0, 0])
#         vec_2 = np.array([0, 1, 0])
#     else:
#         vec_1 = np.array([vec[1], -vec[0], 0])
#         vec_1 /= np.linalg.norm(vec_1)
#         vec_2 = np.cross(vec, vec_1)
#     rot_mat = np.eye(4)
#     rot_mat[:3, 2] = vec
#     rot_mat[:3, 0] = vec_1
#     rot_mat[:3, 1] = vec_2
#     rot_mat[:3, :3] *= vec_norm*scale
#     obj.set_transform(tf.translation_matrix(origin) @ rot_mat)

# def vis_traj(traj_x, traj_v):
#     for i in range(0, 300):
#         vis_vector(vis[f'traj{i}'], traj_x[i], traj_v[i], scale=0.5)

def set_frame(i, name, pos, quat):
    # convert quat from [x,y,z, w] to [w, x,y,z]
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    transform = tf.translation_matrix(pos) @ tf.quaternion_matrix(quat)
    with anim.at_frame(vis, i) as frame:
        frame[name].set_transform(transform)

# Add a box to the scene
box = g.Box([1, 1, 1])
vis["drone"].set_object(g.StlMeshGeometry.from_file('../assets/crazyflie2.stl'))
vis["drone_frame"].set_object(g.StlMeshGeometry.from_file('../assets/axes.stl'))
vis["obj"].set_object(g.Sphere(0.02), material=g.MeshLambertMaterial(color=0x0000ff))
vis["obj_tar"].set_object(g.Sphere(0.02), material=g.MeshLambertMaterial(color=0xff0000))
for i in range(0, 300, 2):
    # vis[f"traj{i}"].set_object(g.StlMeshGeometry.from_file(
    #     '../assets/arrow.stl'), material=g.MeshLambertMaterial(color=0xf000ff))
    vis[f"traj{i}"].set_object(g.Sphere(0.01), material=g.MeshLambertMaterial(color=0x00ff00))
    # vis[f"traj{i}"].set_object(g.StlMeshGeometry.from_file(
    #     '../assets/arrow.stl'), material=g.MeshLambertMaterial(color=0xf000ff))

# load state sequence from pickle and check if load is successful
file_path = "../../results/state_seq.pkl"
with open(file_path, "rb") as f:
    state_seq = pickle.load(f)

# visualize the trajectory
# with anim.at_frame(vis, 0) as frame:
#     vis_traj(state_seq[0].pos_traj,state_seq[0].vel_traj)

# Apply the transformations according to the time sequence
for i, state in enumerate(state_seq):
    if i % 20 == 0:
        # plot the trajectory by connecting the points in traj_x with lines
        for j in range(0, 300, 2):
            set_frame(i, f'traj{j}', state_seq[i].pos_traj[j], np.array([0,0,0,1]))
    set_frame(i, 'drone', state.pos, state.quat)
    set_frame(i, 'drone_frame', state.pos, state.quat)
    set_frame(i, 'obj', state.pos_obj, np.array([0,0,0,1]))
    set_frame(i, 'obj_tar', state.pos_tar, np.array([0,0,0,1]))
    # update camera position by moving it linearly along a-axis
    set_frame(i, 'camera', np.array([0, 0, 0.5]) + 0.1*i*np.array([1, 0, 0]), np.array([0,0,0,1]))
    
vis.set_animation(anim)
time.sleep(5)