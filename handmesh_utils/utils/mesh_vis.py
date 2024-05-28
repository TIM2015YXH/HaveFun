import sys
sys.path.insert(0, '/Users/chenxingyu/Documents/hand_mesh')
from utils.read import read_mesh
import numpy as np


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)


def display_plt(verts, faces=None, alpha=0.2, j=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    ax.scatter([0, ], [0, ], [0, ], color='g')
    if j is not None:
        for i in range(j.shape[0]):
            ax.scatter(j[i, 0], j[i, 1], j[i, 2], color='r')
    plt.xlabel('x')
    plt.ylabel('y')
    cam_equal_aspect_3d(ax, np.concatenate([verts, j], 0))
    plt.show()


def display_open3d(
        verts,
        faces=None,
):
    import open3d as o3d
    geometry = o3d.geometry.TriangleMesh()
    geometry.triangles = o3d.utility.Vector3iVector(faces)

    geometry.vertices = o3d.utility.Vector3dVector(verts)
    geometry.compute_vertex_normals()
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window(
        window_name="display",
        width=1024,
        height=768,
    )
    vis.add_geometry(geometry)

    def kill(vis):
        exit(0)

    vis.register_key_callback(ord("Q"), kill)

    while True:
        geometry.compute_vertex_normals()
        vis.update_geometry(geometry)
        vis.update_renderer()
        vis.poll_events()


def display_pyrender(verts, faces=None):
    import trimesh
    import pyrender

    # region Viewer Options >>>>>>>>>
    scene = pyrender.Scene()
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    node_cam = pyrender.Node(camera=cam, matrix=np.eye(4))
    scene.add_node(node_cam)
    scene.set_pose(node_cam, pose=np.eye(4))
    vertex_colors = np.array([200, 200, 200, 255])
    # endregion <<<<<<<<<<<<

    dt = np.array([0, 0, -500.0])
    dt = dt[np.newaxis, :]
    verts = verts * 1000.0 + dt
    tri_mesh = trimesh.Trimesh(verts, faces, vertex_colors=vertex_colors)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)# viewport_size=(1280, 768)


if __name__ == '__main__':
    path = '/Users/chenxingyu/Documents/manocpp/cmake-build-debug/test.obj'
    mesh = read_mesh(path)

    verts = mesh.x
    faces = mesh.face.T

    joints = np.array([  0.0898886,   0.0058807,  0.00577788,
                0.100447,    0.090164,   0.0103542,
                0.109751,     0.11962,  0.0147513,
                0.117005,    0.139195,   0.0142552,
                0.123579,   0.0880594,  0.0116582,
                0.13865 ,   0.113703,   0.0135001,
                0.149684,    0.132457,   0.0121719,
                0.148831,   0.0542472, -0.00223584,
                0.165137,    0.064505, 0.000863671,
                0.179668,   0.0737114,  0.00400771,
                0.139154,   0.0700289,  0.00571762,
                0.151953,   0.0933143,  0.00908394,
                0.165533,    0.111641,   0.0123459,
                0.0742713,   0.0368861, -0.00313077,
                0.0609259,   0.0616049,  0.00331954,
                0.0551911,   0.0865335,  0.00212716,
                0.126373 ,    0.16077,  0.0163005,
                0.163869,    0.152593,   0.0146586,
                0.194361,   0.0841209,  0.00947901,
                0.179447,    0.129882,   0.0190132,
                0.0407685,    0.115436,  0.00980879]).reshape(21, 3)
    verts_file = '/Users/chenxingyu/Downloads/mano_yar2.txt'
    with open(verts_file, "r") as f:
        data = f.read()
    data = np.array([float(d) for d in data.split(' ')[:-1]])
    verts = data[:778*3].reshape(778, 3)
    verts[:, :3] *= -1
    joints = data[778*3:].reshape(21, 3)
    # verts[:, 0] += 0.38308
    # verts[:, 1] += 0.84272
    # verts[:, 2] += 0.309926

    display_plt(verts, faces, j=joints)
