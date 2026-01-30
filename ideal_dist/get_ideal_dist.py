import os
import json
import gzip
import time
from itertools import product, permutations
from typing import List, Dict, Tuple, Any, Optional, Sequence

def load_distance_matrix(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    point2index = {eval(k): v for k, v in data['point2index'].items()}
    matrix = data['matrix']
    return matrix, point2index



def brute_force_optimal_assignment(
        initial_agent_positions: Sequence[float],
        target_positions: Dict[str, Sequence[Sequence[float]]], matrix, point2index
    ):
        """
        穷举所有(分配 + 顺序 + 目标实例选择)，返回“所有目标都找到”时的最小总路径。
        参数：
        - initial_agent_positions: [pos_a0, pos_a1, ...]，每个 pos 是 [x,y,z]
        - target_positions: { goal_name: [[x,y,z], [x,y,z], ...], ... }
        - geodesic_distance_fn: 你的 geodesic_distance(a, b, episode)
        - episode: 传给 geodesic_distance 的可选参数（用于缓存/路径器）
        返回：
        - best_total: float 最小总路径
        - best_plan:  每个智能体的方案列表：
            [
                {
                "agent_id": int,
                "goal_order": [goal_name, ...],            # 访问顺序
                "instance_choice": [int, ...],             # 各目标选用的候选位置下标
                "cost": float
                }, ...
            ]
        """
        A = initial_agent_positions
        goal_names: List[str] = list(target_positions.keys())
        Gpos = [target_positions[g] for g in goal_names]

        M = len(A)
        N = len(goal_names)

        def dist(a, b) -> float:
            # 只用“单点→单点”的距离
            # aa = np.array(a, dtype=np.float32)
            # bb = np.array(b, dtype=np.float32)
            aa = tuple(a)
            bb = tuple(b)
            aa_index = point2index[aa]
            bb_index = point2index[bb]
            # return float(self._sim.geodesic_distance(a, b))
            return matrix[aa_index][bb_index]

        # 对固定“目标顺序”，在每个目标可能位置之间做分层 DP，求最短路及实例选择
        def best_path_for_order(agent_idx: int, goal_idx_order: List[int]):
            """
            层0: 起点 (只有1个状态)
            层1: 目标 g1 的所有候选位置
            层2: 目标 g2 的所有候选位置
            ...
            转移: 层t-1的任一实例 j -> 层t的任一实例 k
            返回: (最小代价 cost, 各层选中的实例下标列表 choices)
            """
            if not goal_idx_order:
                return 0.0, []

            start = A[agent_idx]

            # 第1层：起点 -> 目标1的所有候选
            g0 = goal_idx_order[0]
            insts0 = Gpos[g0]  # List[Pos]  第一个目标的所有候选位置（可能有多个）
            dp_prev = [float('inf')] * len(insts0)  # 记录“走到第一个目标的第 i 个实例”的最短路径代价。
            back_prev = [-1] * len(insts0)  # 记录“到达当前实例时，它是从上一层哪个实例走来的”。第一层没前驱，所以设成 -1。
            for i, p in enumerate(insts0):
                dp_prev[i] = dist(start, p)

            backs = [back_prev]  # 这是一个二维结构：backs[t][k] 表示第 t 层的第 k 个实例，是从上一层的哪个实例转移过来的。

            # 后续层：逐层松弛
            for t in range(1, len(goal_idx_order)):
                gt = goal_idx_order[t]  # 表示目标顺序中的第 t 个目标。
                insts_t = Gpos[gt]  # 这个目标的所有候选实例。
                dp_cur = [float('inf')] * len(insts_t)  # 到达这个目标的第 k 个实例的最短路径代价。初始化为无穷大。
                back_cur = [-1] * len(insts_t)  # 到达这个实例时，是从上一层哪个实例转移来的。

                prev_goal = goal_idx_order[t - 1]
                insts_prev = Gpos[prev_goal]

                for j, pj in enumerate(insts_prev):
                    base = dp_prev[j]
                    if base == float('inf'):
                        continue
                    for k, pk in enumerate(insts_t):
                        cand = base + dist(pj, pk)
                        if cand < dp_cur[k]:
                            dp_cur[k] = cand
                            back_cur[k] = j

                dp_prev = dp_cur
                backs.append(back_cur)

            # 终层取最小
            last_cost = min(dp_prev)
            last_k = dp_prev.index(last_cost)

            # 回溯实例选择
            chosen = [0] * len(goal_idx_order)
            chosen[-1] = last_k
            for t in range(len(goal_idx_order) - 1, 0, -1):
                chosen[t - 1] = backs[t][chosen[t]]

            return last_cost, chosen

        # 对固定“该智能体负责的目标集合”，穷举所有访问顺序，取最短
        def best_for_agent(agent_idx: int, goal_indices: List[int]):
            if not goal_indices:
                return 0.0, [], []
            best_cost = float('inf')
            best_order = None
            best_choice = None
            for order in permutations(goal_indices):
                c, inst_choice = best_path_for_order(agent_idx, list(order))
                if c < best_cost:
                    best_cost = c
                    best_order = list(order)
                    best_choice = inst_choice
            return best_cost, best_order, best_choice

        # 穷举全局“目标→智能体”的分配
        best_total = float('inf')
        best_plan = None

        # assign[g] = 分配目标 g 给哪个 agent（0..M-1）
        for assign in product(range(M), repeat=N):  # M^N, assign[i]长度为N，表示N个目标分别分配给哪个智能体
            # 收集每个智能体的目标索引
            agent_goals = [[] for _ in range(M)]
            for gi, aidx in enumerate(assign):
                agent_goals[aidx].append(gi)

            total = 0.0
            plan = []
            feasible = True

            for a in range(M):
                cost_a, order_a, inst_choice_a = best_for_agent(a, agent_goals[a])
                if cost_a == float('inf'):
                    feasible = False
                    break
                total += cost_a
                plan.append({
                    "agent_id": a,
                    "goal_order": [goal_names[i] for i in (order_a or [])],
                    "instance_choice": inst_choice_a or [],
                    "cost": cost_a,
                })

            if not feasible:
                continue

            if total < best_total:
                best_total = total
                best_plan = plan

        return best_total, best_plan     

def save_ideal_dist(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    
    scene_dir = '/Data/hm3d/datasets/objectnav/objectnav_hm3d_v2/val/content'
    scene_paths = []
    for file in os.listdir(scene_dir):
        scene_paths.append(os.path.join(scene_dir, file))
    
    output_data = {}    
    for scene_path in scene_paths:
        print("scane_path is: ", scene_path)
        with gzip.open(scene_path, "rt") as f:
            json_str = f.read()
        content = json.loads(json_str)
        episodes = content['episodes']
        file = scene_path.split('/')[-1].split('.')[0]
        path = '/home/huangbw/navigation/PEANUT/ideal_dist/Point_Point_Distance/' + file + '.json'
        matrix, point2index = load_distance_matrix(path)
        
        target_positions = {}
        goals = content['goals_by_category']
        category_to_task_category_id = content['category_to_task_category_id']
        for k, v in goals.items():
            name = k[22:]
            name_id = category_to_task_category_id[name]
            for view in v:
                view_points = view['view_points']
                for point in view_points:
                    if name_id not in target_positions.keys():
                        target_positions[name_id] = [point['agent_state']['position']]
                    else:
                        target_positions[name_id].append(point['agent_state']['position'])
            
        
        for episode in episodes:
            view_points = {}
            episode_id = file + '_' + episode['episode_id']
            initial_agent_position = episode['start_position'][:2]
            object_category = [category_to_task_category_id[c] for c in episode['object_category']]
            
            length = 0
            for c in object_category:
                view_points[c] = target_positions[c]
                length += len(view_points[c])
            
            print(f'The current episode is: {episode_id}')
            print(f'The object_category is {object_category}')
            print(f'The number of target_positions is: {length}')
            
            start_time = time.time()
            ideal_dist, _ = brute_force_optimal_assignment(initial_agent_position, view_points, matrix, point2index)
            print(f'Episode {episode_id} took {time.time() - start_time:.2f} seconds.')
            print(f'The ideal_dist in episdoe {episode_id} is {ideal_dist}')
            
            output_data[episode_id] = ideal_dist
    
    p = '/home/huangbw/navigation/PEANUT/ideal_dist/agent2_ideal_dist.json'
    save_ideal_dist(output_data, p)
            
