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


# def vis_traj(traj_x, traj_v):
#     for i in range(0, 300):
#         vis_vector(vis[f'traj{i}'], traj_x[i], traj_v[i], scale=0.5)
def origin_vec_to_transform(origin, vec, scale=1.0):
    # visualize the force with arrow
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return
    vec = vec / vec_norm
    # gernerate two unit vectors perpendicular to the force vector
    if vec[0] == 0 and vec[1] == 0:
        vec_1 = np.array([1, 0, 0])
        vec_2 = np.array([0, 1, 0])
    else:
        vec_1 = np.array([vec[1], -vec[0], 0])
        vec_1 /= np.linalg.norm(vec_1)
        vec_2 = np.cross(vec, vec_1)
    rot_mat = np.eye(4)
    rot_mat[:3, 2] = vec
    rot_mat[:3, 0] = vec_1
    rot_mat[:3, 1] = vec_2
    rot_mat[:3, :3] *= vec_norm * scale
    return tf.translation_matrix(origin)


def pos_quat_to_transform(pos, quat):
    # convert quat from [x,y,z, w] to [w, x,y,z]
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    return tf.translation_matrix(pos) @ tf.quaternion_matrix(quat)

def set_frame(i, name, transform):
    # convert quat from [x,y,z, w] to [w, x,y,z]
    with anim.at_frame(vis, i) as frame:
        frame[name].set_transform(transform)

# Add a box to the scene
box = g.Box([1, 1, 1])
vis["drone"].set_object(g.StlMeshGeometry.from_file('../assets/crazyflie2.stl'))
vis["drone_frame"].set_object(g.StlMeshGeometry.from_file('../assets/axes.stl'))
vis["obj"].set_object(g.Sphere(0.02), material=g.MeshLambertMaterial(color=0x0000ff))
vis["obj_tar"].set_object(g.Sphere(0.02), material=g.MeshLambertMaterial(color=0xff0000))
vis["disturb"].set_object(g.StlMeshGeometry.from_file('../assets/arrow.stl'))

# create a circle with small capacity
vis["circle"].set_object(g.Cylinder(0.02, 0.11), material=g.MeshLambertMaterial(color=0x0000ff))
vis["circle"].set_transform(tf.rotation_matrix(np.pi/2, [0, 0, 1]))

for i in range(0, 300, 2):
    vis[f"traj{i}"].set_object(g.Sphere(0.01), material=g.MeshLambertMaterial(color=0x00ff00))

# load state sequence from pickle and check if load is successful
# file_path = "../../results/state_seq_.pkl"
file_path = "../../results/state_seq_quad3d_free_jumping_l1_bodyrate.pkl"
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
            set_frame(i, f'traj{j}', pos_quat_to_transform(state_seq[i].pos_traj[j], np.array([0,0,0,1])))
    set_frame(i, 'drone', pos_quat_to_transform(state.pos, state.quat))
    set_frame(i, 'drone_frame', pos_quat_to_transform(state.pos, state.quat))
    set_frame(i, 'obj', pos_quat_to_transform(state.pos_obj, np.array([0,0,0,1])))
    set_frame(i, 'obj_tar', pos_quat_to_transform(state.pos_tar, np.array([0,0,0,1])))
    set_frame(i, 'disturb', origin_vec_to_transform(state.pos, state.f_disturb, 2.0))
    
vis.set_animation(anim)
time.sleep(5)