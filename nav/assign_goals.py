import numpy as np
from constants import hm3d_names, hm3d_to_coco
import numpy as np
from scipy.optimize import linear_sum_assignment
import random







def reassign_goals(num_agents, goal2position, agent2goal, assigned_goal, full_map, dist_wt,
                   candidate_goals, goal_weights, is_can_assigned_goal, step):

    num_goals = len(candidate_goals)
    utility_matrix = np.zeros((num_agents, num_goals))

    for agent_id in range(num_agents):
        dist_map = dist_wt[agent_id]

        for gi, goal_id in enumerate(candidate_goals):
            if goal2position[goal_id] is not None:  # 已找到
                continue
            
            cat_id = hm3d_to_coco[goal_id]  # 转成coco格式的id
            prob_map = full_map[agent_id][cat_id]  # 假设目标通道从0开始
            temp = np.nan_to_num(prob_map - dist_map + 1.0, nan=0, posinf=1e6, neginf=-1e6)
            # raw_gain = np.max(temp)  # [0,2], 这里选最大值
            raw_gain = np.sum(temp)/(960*960)  # 鼓励探索距离远，概率高的物体
            # weight = goal_weights[goal_id]
            # weight = 1.0
            expected_utility = raw_gain  # 加权收益
            utility_matrix[agent_id, gi] = expected_utility
            
           
    # import pdb; pdb.set_trace()
    for agent_id in range(num_agents):
        old_goal = agent2goal[agent_id]
        best_gain = -10000
        best_goal = -1

        for gi, goal_id in enumerate(candidate_goals):
            if goal2position[goal_id] is not None:  # 已找到
                continue
            # if is_can_assigned_goal[goal_id] > current_min_level:
            #     continue
            gain = utility_matrix[agent_id, gi]
            
           
            if goal_id not in agent2goal:
                gain *= (1 + 0.1)  
            
            if gain > best_gain:
                best_gain = gain
                best_goal = goal_id

        if best_goal != -1:
            old_gain = 0
            old_idx = candidate_goals.index(old_goal)
            old_gain = utility_matrix[agent_id, old_idx]

            if goal2position[old_goal] is not None:
                agent2goal[agent_id] = best_goal
                assigned_goal[best_goal] = True
            else:
                if best_gain > old_gain * 1.2:  
                    
                    if agent2goal.count(old_goal) == 1:
                        assigned_goal[old_goal] = False
                    agent2goal[agent_id] = best_goal
                    assigned_goal[best_goal] = True
                    print("Agent %d reassigning from %s to %s" % (agent_id, old_goal, best_goal))
                else:
                    agent2goal[agent_id] = old_goal
        else:
            if goal2position[old_goal] is not None:
                valuable_goals = [g for g in candidate_goals if goal2position[g] is None]
                agent2goal[agent_id] = np.random.choice(valuable_goals)  
                print("Agent %d's old goal %s is found, random reassigning to %s" % (agent_id, old_goal, agent2goal[agent_id]))
                print("utility_matrix:", utility_matrix[agent_id])
            else:
                agent2goal[agent_id] = old_goal
                print("No better goal found for agent %d, keeping %s" % (agent_id, old_goal if old_goal != -1 else "None"))
            
    for i in range(num_agents):
        print("Step %d: agent %d assigned to goal %s" % (step, i, agent2goal[i] if agent2goal[i] != -1 else "None"))
        
        
