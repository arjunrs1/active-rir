import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

EPS_MAPPER = 1e-8

EXPLORED_COLOR = (220, 183, 226)
GT_OBSTACLE_COLOR = (204, 204, 204)
CORRECT_OBSTACLE_COLOR = (51, 102, 0)
FALSE_OBSTACLE_COLOR = (102, 204, 0)
TRAJECTORY_COLOR = (0, 0, 0)


def load_points(points_file: str, transform=True, scene_dataset="replica"):
    r"""
    Helper method to load points data from files stored on disk and transform if necessary
    :param points_file: path to files containing points data
    :param transform: transform coordinate systems of loaded points for use in Habitat or not
    :param scene_dataset: name of scenes dataset ("replica", "mp3d", etc.)
    :return: points in transformed coordinate system for use with Habitat
    """
    points_data = np.loadtxt(points_file, delimiter="\t")
    if transform:
        if scene_dataset == "replica":
            points = list(zip(
                points_data[:, 1],
                points_data[:, 3] - 1.5528907,
                -points_data[:, 2])
            )
        elif scene_dataset == "mp3d":
            points = list(zip(
                points_data[:, 1],
                points_data[:, 3] - 1.5,
                -points_data[:, 2])
            )
        else:
            raise NotImplementedError
    else:
        points = list(zip(
            points_data[:, 1],
            points_data[:, 2],
            points_data[:, 3])
        )
    points_index = points_data[:, 0].astype(int)
    points_dict = dict(zip(points_index, points))
    assert list(points_index) == list(range(len(points)))
    return points_dict, points


def load_points_data(parent_folder, graph_file, transform=True, scene_dataset="replica"):
    r"""
    Main method to load points data from files stored on disk and transform if necessary
    :param parent_folder: parent folder containing files with points data
    :param graph_file: files containing connectivity of points per scene
    :param transform: transform coordinate systems of loaded points for use in Habitat or not
    :param scene_dataset: name of scenes dataset ("replica", "mp3d", etc.)
    :return: 1. points in transformed coordinate system for use with Habitat
             2. graph object containing information about the connectivity of points in a scene
    """
    points_file = os.path.join(parent_folder, 'points.txt')
    graph_file = os.path.join(parent_folder, graph_file)

    _, points = load_points(points_file, transform=transform, scene_dataset=scene_dataset)
    if not os.path.exists(graph_file):
        raise FileExistsError(graph_file + ' does not exist!')
    else:
        with open(graph_file, 'rb') as fo:
            graph = pickle.load(fo)

    return points, graph


def _to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)
    

class Mapper():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.map_config = {"size": config.map_size, "scale": config.map_scale}
        V = self.map_config["size"]
        s = self.map_config["scale"]
        self.img_mean_t = rearrange(
            torch.Tensor(self.config.NORMALIZATION.img_mean), "c -> () c () ()"
        )
        self.img_std_t = rearrange(
            torch.Tensor(self.config.NORMALIZATION.img_std), "c -> () c () ()"
        )
        # Cache to store pre-computed information
        self._cache = {}

    def _spatial_transform(self, p, dx, invert=False):
        """
        Applies the transformation dx to image p.
        Inputs:
            p - (bs, 2, H, W) map
            dx - (bs, 3) egocentric transformation --- (dx, dy, dtheta)

        Conventions:
            The origin is at the center of the map.
            X is upward with agent's forward direction
            Y is rightward with agent's rightward direction

        Note: These denote transforms in an agent's position. Not the image directly.
        For example, if an agent is moving upward, then the map will be moving downward.
        To disable this behavior, set invert=False.
        """
        s = self.map_config["scale"]
        # Convert dx to map image coordinate system with X as rightward and Y as downward
        dx_map = torch.stack(
            [(dx[:, 1] / s), -(dx[:, 0] / s), dx[:, 2]], dim=1
        )  # anti-clockwise rotation
        p_trans = spatial_transform_map(p, dx_map, invert=invert)

        return p_trans

    def _register_map(self, m, p, x, x_prev=None):
        """
        Given the locally computed map, register it to the global map based
        on the current position.

        Inputs:
            m - (bs, F, M, M) global map
            p - (bs, F, V, V) local map
            x - (bs, 3) in global coordinates
            x_prev - (bs, 3) in global coordinates of previous position
        """
        V = self.map_config["size"]
        s = self.map_config["scale"]
        M = m.shape[2]
        Vby2 = (V - 1) // 2 if V % 2 == 1 else V // 2
        Mby2 = (M - 1) // 2 if M % 2 == 1 else M // 2
        # The agent stands at the bottom-center of the egomap and looks upward
        left_h_pad = Mby2 - V + 1
        right_h_pad = M - V - left_h_pad
        left_w_pad = Mby2 - Vby2
        right_w_pad = M - V - left_w_pad

        # Add zero padding to p so that it matches size of global map
        p_pad = F.pad(
            p, (left_w_pad, right_w_pad, left_h_pad, right_h_pad), "constant", 0
        )

        #convert poses to conventional format:
        #x_prev[:, [0, 1]] = x_prev[:, [1, 0]]
        #x[:, [0, 1]] = x[:, [1, 0]]
        x[:,4] *= -1
        #x_prev[:,4] *= -1

        #compute relative change in pose:
        rel_x = subtract_pose(x_prev[:, [0, 1, 4]], x[:, [0, 1, 4]]) #w.r.t prev pose
        #rel_x = x[:, [0, 1, 4]] #OR w.r.t first pose

        # Register the local map
        p_reg = self._spatial_transform(p_pad, rel_x)
        # Aggregate
        m_updated = self._aggregate(m, p_reg)
        
        #transform updated map into current pose
        m_updated_transformed = self._spatial_transform(m_updated, rel_x, invert=True)
        #m_updated_transformed = m_updated

        return m_updated_transformed

    def _aggregate(self, m, p_reg):
        """
        Inputs:
            m - (bs, 2, M, M) - global map
            p_reg - (bs, 2, M, M) - registered egomap
        """
        reg_type = self.config.registration_type
        beta = self.config.map_registration_momentum
        if reg_type == "max":
            m_updated = torch.max(m, p_reg)
        elif reg_type == "overwrite":
            # Overwrite only the currently explored regions
            mask = (p_reg[:, 1] > self.config.thresh_explored).float()
            mask = mask.unsqueeze(1)
            m_updated = m * (1 - mask) + p_reg * mask
        elif reg_type == "moving_average":
            mask_unexplored = (
                (p_reg[:, 1] <= self.config.thresh_explored).float().unsqueeze(1)
            )
            mask_unfilled = (m[:, 1] == 0).float().unsqueeze(1)
            m_ma = p_reg * (1 - beta) + m * beta
            m_updated = (
                m * mask_unexplored
                + m_ma * (1.0 - mask_unexplored) * (1.0 - mask_unfilled)
                + p_reg * (1.0 - mask_unexplored) * mask_unfilled
            )
        elif reg_type == "entropy_moving_average":
            explored_mask = (p_reg[:, 1] > self.config.thresh_explored).float()
            log_p_reg = torch.log(p_reg + EPS_MAPPER)
            log_1_p_reg = torch.log(1 - p_reg + EPS_MAPPER)
            entropy = -p_reg * log_p_reg - (1 - p_reg) * log_1_p_reg
            entropy_mask = (entropy.mean(dim=1) < self.config.thresh_entropy).float()
            explored_mask = explored_mask * entropy_mask
            unfilled_mask = (m[:, 1] == 0).float()
            m_updated = m
            # For regions that are unfilled, write as it is
            mask = unfilled_mask * explored_mask
            mask = mask.unsqueeze(1)
            m_updated = m_updated * (1 - mask) + p_reg * mask
            # For regions that are filled, do a moving average
            mask = (1 - unfilled_mask) * explored_mask
            mask = mask.unsqueeze(1)
            p_reg_ma = (p_reg * (1 - beta) + m_updated * beta) * mask
            m_updated = m_updated * (1 - mask) + p_reg_ma * mask
        else:
            raise ValueError(
                f"Mapper: registration_type: {self.config.registration_type} not defined!"
            )

        return m_updated

    def ext_register_map(self, m, p, x, x_prev=None):
        return self._register_map(m, p, x, x_prev)

def spatial_transform_map(p, x, invert=True, mode="bilinear"):
    """
    Inputs:
        p     - (bs, f, H, W) Tensor
        x     - (bs, 3) Tensor (x, y, theta) transforms to perform
    Outputs:
        p_trans - (bs, f, H, W) Tensor
    Conventions:
        Shift in X is rightward, and shift in Y is downward. Rotation is clockwise.

    Note: These denote transforms in an agent's position. Not the image directly.
    For example, if an agent is moving upward, then the map will be moving downward.
    To disable this behavior, set invert=False.
    """
    device = p.device
    H, W = p.shape[2:]

    trans_x = x[:, 0]
    trans_y = x[:, 1]
    # Convert translations to -1.0 to 1.0 range
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H / 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W / 2

    trans_x = trans_x / Wby2
    trans_y = trans_y / Hby2
    rot_t = x[:, 2]

    sin_t = torch.sin(rot_t)
    cos_t = torch.cos(rot_t)

    # This R convention means Y axis is downwards.
    A = torch.zeros(p.size(0), 3, 3).to(device)
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = -sin_t
    A[:, 1, 0] = sin_t
    A[:, 1, 1] = cos_t
    A[:, 0, 2] = trans_x
    A[:, 1, 2] = trans_y
    A[:, 2, 2] = 1

    # Since this is a source to target mapping, and F.affine_grid expects
    # target to source mapping, we have to invert this for normal behavior.
    Ainv = torch.inverse(A)

    # If target to source mapping is required, invert is enabled and we invert
    # it again.
    if invert:
        Ainv = torch.inverse(Ainv)

    Ainv = Ainv[:, :2]
    grid = F.affine_grid(Ainv, p.size())
    p_trans = F.grid_sample(p, grid, mode=mode)

    return p_trans

def generate_topdown_allocentric_map(
    global_map,
    pred_coverage_map,
    thresh_explored=0.6,
    thresh_obstacle=0.6,
    zoom=True,
):
    """
    Inputs:
        global_map        - (2, H, W) numpy array
        pred_coverage_map - (2, H, W) numpy array
    """
    H, W = global_map.shape[1:]
    colored_map = np.ones((H, W, 3), np.uint8) * 255
    global_obstacle_map = (global_map[0] == 1) & (global_map[1] == 1)

    # First show explored regions
    explored_map = pred_coverage_map[1] >= thresh_explored
    colored_map[explored_map, :] = np.array(EXPLORED_COLOR)

    # Show GT obstacles in explored regions
    gt_obstacles_in_explored_map = global_obstacle_map & explored_map
    colored_map[gt_obstacles_in_explored_map, :] = np.array(GT_OBSTACLE_COLOR)

    # Show correctly predicted obstacles in dark green
    pred_obstacles = (pred_coverage_map[0] >= thresh_obstacle) & explored_map
    correct_pred_obstacles = pred_obstacles & gt_obstacles_in_explored_map
    colored_map[correct_pred_obstacles, :] = np.array(CORRECT_OBSTACLE_COLOR)

    # Show in-correctly predicted obstacles in light green
    false_pred_obstacles = pred_obstacles & ~gt_obstacles_in_explored_map
    colored_map[false_pred_obstacles, :] = np.array(FALSE_OBSTACLE_COLOR)

    if zoom:
        # Add an initial padding to ensure a non-zero boundary.
        global_occ_map = np.pad(global_map[0], 5, mode="constant", constant_values=1.0)
        # Zoom into the map based on extents in global_map
        global_map_ysum = (1 - global_occ_map).sum(axis=0)  # (W, )
        global_map_xsum = (1 - global_occ_map).sum(axis=1)  # (H, )
        x_start = W
        x_end = 0
        y_start = H
        y_end = 0
        for i in range(W - 1):
            if global_map_ysum[i] == 0 and global_map_ysum[i + 1] > 0:
                x_start = min(x_start, i)
            if global_map_ysum[i] > 0 and global_map_ysum[i + 1] == 0:
                x_end = max(x_end, i)

        for i in range(H - 1):
            if global_map_xsum[i] == 0 and global_map_xsum[i + 1] > 0:
                y_start = min(y_start, i)
            if global_map_xsum[i] > 0 and global_map_xsum[i + 1] == 0:
                y_end = max(y_end, i)

        # Remove the initial padding
        x_start = max(x_start - 5, 0)
        y_start = max(y_start - 5, 0)
        x_end = max(x_end - 5, 0)
        y_end = max(y_end - 5, 0)

        # Some padding
        x_start = max(x_start - 5, 0)
        x_end = min(x_end + 5, W - 1)
        x_width = x_end - x_start + 1
        y_start = max(y_start - 5, 0)
        y_end = min(y_end + 5, H - 1)
        y_width = y_end - y_start + 1
        max_width = max(x_width, y_width)

        colored_map = colored_map[
            y_start : (y_start + max_width), x_start : (x_start + max_width)
        ]

    return colored_map

def convert_world2map(world_coors, map_shape, map_scale):
    """
    World coordinate system:
        Agent starts at (0, 0) facing upward along X. Y is rightward.
    Map coordinate system:
        Agent starts at (W/2, H/2) with X rightward and Y downward.

    Inputs:
        world_coors: (bs, 2) --- (x, y) in world coordinates
        map_shape: tuple with (H, W)
        map_scale: scalar indicating the cell size in the map
    """
    H, W = map_shape
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2

    x_world = world_coors[:, 0]
    y_world = world_coors[:, 1]

    x_map = torch.clamp((Wby2 + y_world / map_scale), 0, W - 1).round()
    y_map = torch.clamp((Hby2 - x_world / map_scale), 0, H - 1).round()

    map_coors = torch.stack([x_map, y_map], dim=1)  # (bs, 2)

    return map_coors

def add_pose(pose_a, pose_ab):
    """
    Add pose_ab (in ego-coordinates of pose_a) to pose_a
    Inputs:
        pose_a - (bs, 3) --- (x, y, theta)
        pose_b - (bs, 3) --- (x, y, theta)

    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction
    """

    x_a, y_a, theta_a = torch.unbind(pose_a, dim=1)
    x_ab, y_ab, theta_ab = torch.unbind(pose_ab, dim=1)

    r_ab = torch.sqrt(x_ab ** 2 + y_ab ** 2)
    phi_ab = torch.atan2(y_ab, x_ab)

    x_b = x_a + r_ab * torch.cos(phi_ab + theta_a)
    y_b = y_a + r_ab * torch.sin(phi_ab + theta_a)
    theta_b = theta_a + theta_ab
    theta_b = torch.atan2(torch.sin(theta_b), torch.cos(theta_b))

    pose_b = torch.stack([x_b, y_b, theta_b], dim=1)  # (bs, 3)

    return pose_b

def convert_gt2channel_to_gtrgb(gts):
    """
    Inputs:
        gts   - (H, W, 2) numpy array with values between 0.0 to 1.0
              - channel 0 is 1 if occupied space
              - channel 1 is 1 if explored space
    """
    H, W, _ = gts.shape

    exp_mask = (gts[..., 1] >= 0.5).astype(np.float32)
    occ_mask = (gts[..., 0] >= 0.5).astype(np.float32) * exp_mask
    free_mask = (gts[..., 0] < 0.5).astype(np.float32) * exp_mask
    unk_mask = 1 - exp_mask

    gt_imgs = np.stack(
        [
            0.0 * occ_mask + 0.0 * free_mask + 255.0 * unk_mask,
            0.0 * occ_mask + 255.0 * free_mask + 255.0 * unk_mask,
            255.0 * occ_mask + 0.0 * free_mask + 255.0 * unk_mask,
        ],
        axis=2,
    ).astype(
        np.uint8
    )  # (H, W, 3)

    return gt_imgs

def subtract_pose(pose_a, pose_b):
    """
    Compute pose of pose_b in the egocentric coordinate frame of pose_a.
    Inputs:
        pose_a - (bs, 3) --- (x, y, theta)
        pose_b - (bs, 3) --- (x, y, theta)

    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction
    """

    x_a, y_a, theta_a = torch.unbind(pose_a, dim=1)
    x_b, y_b, theta_b = torch.unbind(pose_b, dim=1)

    r_ab = torch.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)  # (bs, )
    phi_ab = torch.atan2(y_b - y_a, x_b - x_a) - theta_a  # (bs, )
    theta_ab = theta_b - theta_a  # (bs, )
    theta_ab = torch.atan2(torch.sin(theta_ab), torch.cos(theta_ab))

    x_ab = torch.stack(
        [r_ab * torch.cos(phi_ab), r_ab * torch.sin(phi_ab), theta_ab,], dim=1
    )  # (bs, 3)

    return x_ab

def measure_area_seen_performance(map_states, map_scale=1.0, reduction="mean"):
    """
    Inputs:
        map_states - (bs, 2, M, M) world map with channel 0 representing occupied
                     regions (1s) and channel 1 representing explored regions (1s)
    """

    bs = map_states.shape[0]
    explored_map = (map_states[:, 1] > 0.5).float()  # (bs, M, M)
    occ_space_map = (map_states[:, 0] > 0.5).float() * explored_map  # (bs, M, M)
    free_space_map = (map_states[:, 0] <= 0.5).float() * explored_map  # (bs, M, M)

    all_cells_seen = explored_map.view(bs, -1).sum(dim=1)  # (bs, )
    occ_cells_seen = occ_space_map.view(bs, -1).sum(dim=1)  # (bs, )
    free_cells_seen = free_space_map.view(bs, -1).sum(dim=1)  # (bs, )

    area_seen = all_cells_seen * (map_scale ** 2)
    free_space_seen = free_cells_seen * (map_scale ** 2)
    occupied_space_seen = occ_cells_seen * (map_scale ** 2)

    if reduction == "mean":
        area_seen = area_seen.mean().item()
        free_space_seen = free_space_seen.mean().item()
        occupied_space_seen = occupied_space_seen.mean().item()
    elif reduction == "sum":
        area_seen = area_seen.sum().item()
        free_space_seen = free_space_seen.sum().item()
        occupied_space_seen = occupied_space_seen.sum().item()

    return {
        "area_seen": area_seen,
        "free_space_seen": free_space_seen,
        "occupied_space_seen": occupied_space_seen,
    }
