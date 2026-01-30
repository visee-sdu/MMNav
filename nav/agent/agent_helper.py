import math
import os
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
import math
from agent.utils.fmm_planner import FMMPlanner
from agent.utils.segmentation import SemanticPredMaskRCNN
from constants import color_palette
import agent.utils.pose as pu
import agent.utils.visualization as vu


# The untrap helper for the bruteforce untrap mode (from Stubborn)
class UnTrapHelper:
    def __init__(self):
        self.total_id = 0
        self.epi_id = 0

    def reset(self, full=False):
        self.total_id += 1
        if full:
            self.total_id = 0
        self.epi_id = 0

    def get_action(self):
        self.epi_id += 1
        if self.epi_id > 30:
            return np.random.randint(2, 4)
        if self.epi_id > 18:
            if self.total_id % 2 == 0:
                return 2
            else:
                return 3
        if self.epi_id  < 3:
            if self.total_id % 2 == 0:
                return 2
            else:
                return 3
        else:
            if self.total_id % 2 == 0:
                return 3
            else:
                return 2


class Agent_Helper:
    """
    Class containing functions for motion planning and visualization.
    """

    def __init__(self, args,agent_states, agent_id):

        self.args = args
        self.agent_id = agent_id

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])  # 120， 160

        # initialize semantic segmentation prediction model
        self.seg_model = SemanticPredMaskRCNN(args)  # 输出 h*w*10的语义图，代表9个目标+1个背景的概率
        
        # initializations for planning:
        self.selem = skimage.morphology.disk(args.col_rad)  # 一个圆盘形状的结构元素，用于形态学操作，半径为 args.col_rad=4；

        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.last_start = None
        self.rank = 0
        self.episode_no = 0
        self.stg = None
        self.goal_cat = -1
        self.untrap = UnTrapHelper()
        self.agent_states = agent_states

        # We move forward 1 extra step after approaching goal to make the agent closer to goal
        self.forward_after_stop_preset = self.args.move_forward_after_stop
        self.forward_after_stop = self.forward_after_stop_preset  # 停止后向前移动一步 1

        self.map_size = args.map_size_cm // args.map_resolution  # 4800//5
        self.full_w, self.full_h = self.map_size, self.map_size    # 960, 960
        self.local_w = int(self.full_w / args.global_downscaling)  # 480
        self.local_h = int(self.full_h / args.global_downscaling)  # 480
        self.found_goal = None

        self.edge_buffer = 10 if args.num_sem_categories <= 16 else 40

        if args.visualize:
            self.legend = cv2.imread('nav/new_hm3d_legend.png')[:118]
            self.vis_image = None
            self.rgb_vis = None

    def reset(self):
        args = self.args

        self.obs_shape = None

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)  # 960, 960
        self.collision_map = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        self.episode_no += 1
        self.timestep = 0
        self.prev_blocked = 0
        self._previous_action = -1
        self.block_threshold = 4
        self.untrap.reset(full=True)
        self.forward_after_stop = self.forward_after_stop_preset


    def plan_act(self, planner_inputs, agent_id):
        """
        Function responsible for motion planning and visualization.

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'obstacle'  (ndarray): (M, M) map prediction
                    'exp_pred'  (ndarray): (M, M) exploration mask 
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found
                    'goal_name' (str): name of target category

        Returns:
            action (dict): {'action': action}
        """

        self.timestep += 1
        self.goal_name = planner_inputs['goal_name']

        action = self._plan(planner_inputs)  # 调用私有函数 _plan() 执行规划逻辑，输入就是当前的局部地图状态

        if self.args.visualize:
            self._visualize(planner_inputs, agent_id)

        action = {'action': action}  # 封装成标准的 {'action': ...} 字典格式，方便与 Habitat 或 RL 环境对接；
        self.last_action = action['action']
        return action

    
    def set_goal_cat(self, goal_cat):
        self.goal_cat = goal_cat
        

    def preprocess_inputs(self, rgb, depth, info):
        obs = self._preprocess_obs(rgb, depth, info)  # N*H*W的obs，N=3+1+C，其中C为语义分割类别数

        self.obs = obs
        self.info = info

        return obs, info

    
    def _preprocess_obs(self, rgb, depth, info):
        args = self.args
        
        if args.use_gt_seg:
            sem_seg_pred[:, :, self.goal_cat] = info['goalseg']  # 直接使用gt
        else:
            sem_seg_pred = self._get_sem_pred(rgb.astype(np.uint8), depth=depth)  # 根据 RGB 和 depth 预测语义分割结果， h*w*10

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)  # 预处理深度图，h*w，单位为厘米（cm）

        ds = args.env_frame_width // args.frame_width  # Downscaling factor, 640//160
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))  # 将 RGB 缩小至目标尺寸
            depth = depth[ds // 2::ds, ds // 2::ds]  # 对 depth 和 sem_seg_pred 用 slicing 的方式降采样（跳步采样 + 中心对齐），效率更高。
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)  # 将 [H, W] 的深度图变为 [H, W, 1]，以便与 RGB 和语义图拼接。
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)  # (3 + 1 + 10, H, W)

        return state  # N*H*W的obs，N=3+1+C，其中C为语义分割类别数

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1  # 创建一个副本，避免原地修改

        for i in range(depth.shape[1]):  # 处理无效像素（逐列处理）
            # Invalid pixels have value = 0
            invalid = depth[:, i] == 0.
            if np.mean(invalid) > 0.9:  # 某一列的深度值如果大于 90% 都是 0，表示这一列几乎完全无效，直接将无效像素填为该列的最大值
                depth[:, i][invalid] = depth[:, i].max()
            else:
                depth[:, i][invalid] = 100.0   # 否则将无效像素设为 100 米（10000cm），表示非常远，相当于 ignore。


        # Also ignore too-far pixels
        mask2 = depth > 0.99  # 将过远（大于 0.99）的像素当作无效，重置为 0，统一处理。
        depth[mask2] = 0.

        mask1 = depth == 0  # 再次统一将无效值设为 100 米
        depth[mask1] = 100.0 

        # Convert to cm 
        depth = min_d * 100.0 + depth * (max_d - min_d) * 100.0  # 归一化为真实深度值（单位为厘米）
        return depth


    def _get_sem_pred(self, rgb, depth=None):
        if self.args.visualize:
            self.rgb_vis = rgb[:, :, ::-1]  # RGB—>BGR

        sem_pred, sem_vis = self.seg_model.get_prediction(rgb, depth, goal_cat=self.goal_cat)
        return sem_pred.astype(np.float32)  # h*w*10, 非0的位置表示没有这个类别的物体，
    
    
    def _plan(self, planner_inputs):
        """
        Function responsible for planning.

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'obstacle'  (ndarray): (M, M) map prediction
                    'exp_pred'  (ndarray): (M, M) exploration mask 
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found
                    'goal_name' (str): name of target category

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Obstacle map
        map_pred = np.rint(planner_inputs['obstacle'])  # 障碍图，四舍五入处理为 0/1；概率大于0.5的认为是障碍
        
        self.found_goal = planner_inputs['found_goal']  # 是否已找到目标对象（从目标语义图中提取）。
        goal = planner_inputs['goal'] # 表示目标的二值图；

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']  # (start_x, start_y, start_o) 是 agent 的位姿（x, y, 方向）；
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)  # gx1:gx2, gy1:gy2 是局部规划区域在全局地图中的边界。
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start_exact = [r * 100.0 / args.map_resolution - gx1,
                       c * 100.0 / args.map_resolution - gy1]
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)  # 把世界坐标投影到局部 map 中；乘以 100 / map_resolution 实现从米到像素的缩放。

        # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                      int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.last_start = last_start
        if self.timestep != 1:
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                vu.draw_line(last_start, start,
                            self.visited_vis[gx1:gx2, gy1:gy2])  # 在 visited map 上从 last_start 连线到 start 位置，记录 agent 走过的路径。

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4 if self.prev_blocked < self.block_threshold else 2
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 1)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision, 如果当前前进动作导致几乎没有位置变化，说明撞墙了；使用 agent 面朝方向和一定宽度向前投影一段区域，标记为障碍；
                self.prev_blocked += 1  # 统计连续多少次被阻塞，用于判断是否被卡住。
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

            else:
                if self.prev_blocked >= self.block_threshold:  # 如果多次被阻挡，说明 agent 被卡住；启动 untrap.get_action() 策略脱离死角。
                    self.untrap.reset()
                self.prev_blocked = 0

        # Deterministic Local Policy
        stg, stop = self._get_stg(map_pred, start_exact, np.copy(goal),
                                  planning_window)  # 根据障碍图、起点、目标图，返回一个短期子目标点 stg；stop:是否达到了目标。

        if self.forward_after_stop < 0:  # 到达目标区域且目标已识别，执行 STOP；
            self.forward_after_stop = self.forward_after_stop_preset
        if self.forward_after_stop != self.forward_after_stop_preset:
            if self.forward_after_stop == 0:
                self.forward_after_stop -= 1
                action = 0
                # self.stop_reset()  # Reset forward_after_stop
            else:
                self.forward_after_stop -= 1
                action = 1
        elif stop and planner_inputs['found_goal'] == 1:  # 否则根据相对角度判断：往左转 / 右转 / 前进；
            if self.forward_after_stop == 0:
                action = 0  # Stop
                # self.stop_reset()  # Reset forward_after_stop
            else:
                self.forward_after_stop -= 1
                action = 1
        else:
            (stg_x, stg_y) = stg
            
            # Stay within global map
            stg_x = np.clip(stg_x, self.edge_buffer, self.local_w - self.edge_buffer - 1)
            stg_y = np.clip(stg_y, self.edge_buffer, self.local_h - self.edge_buffer - 1)
            
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360
                
            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward
                
        if self.prev_blocked >= self.block_threshold:
            if self._previous_action == 1:
                action = self.untrap.get_action()
            else:
                action = 1
        self._previous_action = action
        return action

    def stop_reset(self):
        self.forward_after_stop_preset = self.args.move_forward_after_stop
        self.forward_after_stop = self.forward_after_stop_preset  # 停止后向前移动一步 1
        self.prev_blocked = 0
        self._previous_action = -1
        self.last_action = None

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        if gx2 == self.full_w:
            grid[x2 - 1] = 1
        if gy2 == self.full_h:
            grid[:, y2 - 1] = 1
            
        if gx1 == 0:
            grid[x1] = 1
        if gy1 == 0:
            grid[y1] = 1
            
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        def surrounded_by_obstacle(mat,i,j):
            i = int(i)
            j = int(j)
            i1 = max(0,i-3)
            i2 = min(mat.shape[0],i+2)
            j1 = max(0,j-3)
            j2 = min(mat.shape[1],j+2)
            return np.sum(mat[i1:i2,j1:j2]) > 0
        

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True

        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1


        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        
        selem = skimage.morphology.disk(8 if self.found_goal == 1 else 2)

        # Smalller radius for toilet
        is_toilet = self.info['goal_name'] == 'toilet'
        if is_toilet:
            selem = skimage.morphology.disk(6 if self.found_goal == 1 else 2)

        goal = skimage.morphology.binary_dilation(
            goal, selem) != True

        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        # assume replan true suggests failure in planning
        stg_x, stg_y, distance, stop, replan = planner.get_short_term_goal(state)
        
        # Failed to plan a path
        if replan:
            if self.args.only_explore:
                self.agent_states.next_preset_goal()

            # Try again with eroded obstacle map
            grid = skimage.morphology.binary_erosion(grid.astype(bool)).astype(int)
            traversible = skimage.morphology.binary_dilation(
                grid[x1:x2, y1:y2],
                self.selem) != True

            traversible[self.collision_map[gx1:gx2, gy1:gy2]
                        [x1:x2, y1:y2] == 1] = 0
            traversible[self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1


            traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                        int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

            traversible = add_boundary(traversible)

            planner = FMMPlanner(traversible)
            planner.set_multi_goal(goal)

            state = [start[0] - x1 + 1, start[1] - y1 + 1]
            
            # assume replan true suggests failure in planning
            stg_x, stg_y, distance, stop, replan = planner.get_short_term_goal(state)
            

        #If we fail to plan a path to the goal, make goal larger
        if self.found_goal == 1 and distance > self.args.magnify_goal_when_hard:
            radius = 2
            step = 0

            while distance > 100:
                step += 1
                if step > 8 or (is_toilet and step > 2):
                    break
                selem = skimage.morphology.disk(radius)
                goal = skimage.morphology.binary_dilation(
                    goal, selem) != True
                goal = 1 - goal * 1.
                planner.set_multi_goal(goal)

                # assume replan true suggests failure in planning
                stg_x, stg_y, distance, stop, replan = planner.get_short_term_goal(
                    state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
        self.stg = (stg_x, stg_y)
        return (stg_x, stg_y), stop


    def _visualize(self, inputs, agent_id):
        """Generate visualization and save."""

        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no - 1)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        map_pred = inputs['obstacle']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        goal = inputs['goal']

        sem_map = inputs['sem_map_pred']
        
        self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5
        sem_map[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 14
        if int(self.stg[0]) < self.local_w and int(self.stg[1]) < self.local_h:
            sem_map[int(self.stg[0]),int(self.stg[1])] = 15

        no_cat_mask = sem_map == args.num_sem_categories + 4
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        
        # 先得到“基础语义图”（不含轨迹）
        base_sem_map = sem_map.copy()
        base_sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask); base_sem_map[m1] = 2
        m2 = np.logical_and(no_cat_mask, map_mask); base_sem_map[m2] = 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        # Draw goal dot
        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True
        
        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

        def render_sem_map(sm):
            color_pal = [int(x * 255.) for x in color_palette]
            sm_vis = Image.new("P", (sm.shape[1], sm.shape[0]))
            sm_vis.putpalette(color_pal)
            sm_vis.putdata(sm.flatten().astype(np.uint8))
            sm_vis = sm_vis.convert("RGB")
            sm_vis = np.flipud(sm_vis)[:, :, [2, 1, 0]]
            return cv2.resize(sm_vis, (480, 480), interpolation=cv2.INTER_NEAREST)
        
        # color_pal = [int(x * 255.) for x in color_palette]
        # sem_map_vis = Image.new("P", (sem_map.shape[1],
        #                               sem_map.shape[0]))
        # sem_map_vis.putpalette(color_pal)
        # sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        # sem_map_vis = sem_map_vis.convert("RGB")
        # sem_map_vis = np.flipud(sem_map_vis)

        # sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        # sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
        #                          interpolation=cv2.INTER_NEAREST)
        
        # 渲染两张图
        sem_map_vis = render_sem_map(sem_map)            # 含红色路径（原图）
        sem_map_vis_no_traj   = render_sem_map(base_sem_map)       # 不含红色路径
        
        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis
        
                        
        right_panel = self.vis_image[:, -250:] 
                                   
        my_cm = matplotlib.cm.get_cmap('Purples')
        data = self.agent_states.target_pred
        if data is not None:
            # data[data == -100000.0] = 0.0
            # import pdb; pdb.set_trace()
            normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
                                   
            mapped_data = my_cm(normed_data)[::-1, :, [2, 1, 0]] * 255
            
            white_idx = np.where(np.sum(sem_map_vis, axis=2) == 255 * 3) 
            mapped_data_vis = cv2.resize(mapped_data, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
            self.vis_image[50:530, 670:1150][white_idx] = mapped_data_vis[white_idx]


            data = self.agent_states.value
            normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            mapped_data = my_cm(normed_data)[::-1, :, [2, 1, 0]] * 255
            mapped_data_vis = cv2.resize(mapped_data, (240, 240),
                             interpolation=cv2.INTER_NEAREST)
            right_panel[290:530, :240] = mapped_data_vis

            data = self.agent_states.dd_wt
            normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            mapped_data = my_cm(normed_data)[::-1, :, [2, 1, 0]] * 255
            mapped_data_vis = cv2.resize(mapped_data, (240, 240),
                             interpolation=cv2.INTER_NEAREST)
            right_panel[50:290, :240] = mapped_data_vis
            
            border_color = [100] * 3
            right_panel[49, :240] = border_color
            right_panel[530, :240] = border_color
            right_panel[49:531, 240] = border_color
            
                            
        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        # Draw agent as an arrow
        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1) 
        
        # 生成箭头轮廓（注意 origin 设为小图的左上角）
        arrow_local = vu.get_contour_points(pos, origin=(0, 0))

        # 在“含轨迹”的图上画
        cv2.drawContours(sem_map_vis, [arrow_local], 0, color, -1)
        # 在“不含轨迹”的图上画
        # cv2.drawContours(sem_map_vis_no_traj, [arrow_local], 0, color, -1)
        
        if args.visualize == 1:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        elif args.visualize == 2:
            # Saving the image
            fn = '{}/episodes/thread_{}/eps_{}/agent-{}-{}-{}-Vis-{}.jpg'.format(
                dump_dir, self.rank, self.episode_no - 1, agent_id,
                self.rank, self.episode_no - 1, self.timestep)

            cv2.imwrite(fn, self.vis_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            fn_clean = '{}/episodes/thread_{}/eps_{}/agent-{}-{}-{}-Sem-NoTraj-{}.jpg'.format(
                dump_dir, self.rank, self.episode_no - 1, agent_id,
                self.rank, self.episode_no - 1, self.timestep)
            cv2.imwrite(fn_clean, sem_map_vis_no_traj, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            fn_clean = '{}/episodes/thread_{}/eps_{}/agent-{}-{}-{}-Sem-Traj-{}.jpg'.format(
                dump_dir, self.rank, self.episode_no - 1, agent_id,
                self.rank, self.episode_no - 1, self.timestep)
            cv2.imwrite(fn_clean, sem_map_vis, [cv2.IMWRITE_JPEG_QUALITY, 100])


