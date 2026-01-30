import argparse
import os
import random
import habitat

from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
import torch
import sys
import cv2
from arguments import get_args
from habitat.core.env import Env
# from constants import hm3d_names
import numpy as np
import matplotlib.pyplot as plt

from agent.peanut_agent import PEANUT_Agent
from agent.multi_agent_env import Multi_Agent_Env
import habitat_sim
from collections import defaultdict
from constants import hm3d_names, hm3d_to_coco, names_hm3d
from assign_goals import reassign_goals
import copy
from agent.agent_state import Agent_State

import json
import Distance

@habitat.registry.register_action_space_configuration
class PreciseTurn(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.TURN_LEFT_S] = habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )
        config[HabitatSimActions.TURN_RIGHT_S] = habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )

        return config


def save_ideal_dist(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_ideal_dist(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():

    args = get_args()
    args.only_explore = 0  
    HabitatSimActions.extend_action_space("TURN_LEFT_S")
    HabitatSimActions.extend_action_space("TURN_RIGHT_S")
    
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    config.defrost()
    config.TASK.POSSIBLE_ACTIONS = config.TASK.POSSIBLE_ACTIONS + [
        "TURN_LEFT_S",
        "TURN_RIGHT_S",
    ]
    config.TASK.ACTIONS.TURN_LEFT_S = habitat.config.Config()
    config.TASK.ACTIONS.TURN_LEFT_S.TYPE = "TurnLeftAction_S"
    config.TASK.ACTIONS.TURN_RIGHT_S = habitat.config.Config()
    config.TASK.ACTIONS.TURN_RIGHT_S.TYPE = "TurnRightAction_S"
    config.SIMULATOR.ACTION_SPACE_CONFIG = "PreciseTurn"
    
    
    config.SEED = 100
    # config.SEED = 0
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sem_gpu_id
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 1
    config.DATASET.SPLIT = 'val'
    
    config.freeze()
    
    # hab_env = Env(config=config)
    hab_env = Multi_Agent_Env(config)
    
    num_agents = config.SIMULATOR.NUM_AGENTS
    
    # nav_agent = PEANUT_Agent(args=args,task_config=config)
    
    agent = []
    for i in range(num_agents):
        agent.append(PEANUT_Agent(args=args, task_config=config, agent_id=i))
    print(config.DATASET.SPLIT, 'split')
    print(len(hab_env.episodes), 'episodes in dataset')
    
    num_episodes = 1000
    start = args.start_ep
    end = args.end_ep if args.end_ep > 0 else num_episodes
    
    # sucs, spls, ep_lens, dtgs = [], [], [], []
    multi_agent_multi_goals = []
    
    ep_i = 0
    Distance.calculated_ideal_dist = load_ideal_dist('./ideal_dist/agent2_ideal_dist.json')
    
    while ep_i < min(num_episodes, end):
        observations = hab_env.reset()
        # init_goal = observations[0]['objectgoal'][0]
        # previous_agent2goal = hab_env.task.agent2goal.copy()
        previous_agent2goal = copy.deepcopy(hab_env.task.agent2goal)
        for i in range(num_agents):
            agent[i].reset(observations[i])
        # nav_agent.reset()
        print('-' * 40)
        sys.stdout.flush()
        
        
        if ep_i >= start and ep_i < end:
            for i in range(num_agents):
                print('Episode %d | Target: %s' % (ep_i, hm3d_names[observations[i]['objectgoal'][0]]))
            print('Scene: %s' % hab_env._current_episode.scene_id)

            step_i = 0
            seq_i = 0
            
            while not hab_env.episode_over:
                # action = [0, 0]
                action = [0]*num_agents
                infos = []
                for i in range(num_agents):
                    info = agent[i].mapping(observations[i])
                    infos.append(info)
                    
                for i in range(num_agents):
                    action[i] = agent[i].act2(infos[i])['action']
                # action = nav_agent.act(observations)
                observations = hab_env.step(action)
                
                if all(v is not None for v in hab_env.task.goal2position.values()):
                    break
                
                if step_i ==0 or step_i % args.update_goal_freq == args.update_goal_freq - 1 or hab_env.task.flag:
                    p_map = []
                    dist_wt = []
                    for i in range(num_agents):
                        p_map.append(agent[i].agent_states.probability_preds)    
                        dist_wt.append(agent[i].agent_states.dist_wt)
                        
                    reassign_goals(num_agents=num_agents, goal2position=hab_env.task.goal2position, agent2goal=hab_env.task.agent2goal, assigned_goal=hab_env.task.assigned_goals,
                                   full_map=p_map, dist_wt=dist_wt, candidate_goals=hab_env.task.goals, goal_weights=hab_env.task.goal_weight,
                                   is_can_assigned_goal=hab_env.task.goal_priority, step=step_i)
                    if hab_env.task.flag:
                        hab_env.task.flag = False
                
                
                for i in range(num_agents):
                    observations[i]['objectgoal'] = np.array([hab_env.task.agent2goal[i]])
                for i in range(num_agents):
                    if previous_agent2goal[i] != hab_env.task.agent2goal[i]:  
                        print('Agent %d reassigned to goal %s at step %d, previous goals is %d' % (i, hab_env.task.agent2goal[i], step_i, previous_agent2goal[i]))
                        
                        agent[i].agent_helper.col_width = 1
                        agent[i].agent_helper.prev_blocked = 0
                        agent[i].agent_helper.forward_after_stop = agent[i].agent_helper.forward_after_stop_preset
                        
                        previous_agent2goal[i] = hab_env.task.agent2goal[i]  
                        
                        
                          
                if step_i % 100 == 0:
                    print('step %d...' % step_i)
                    for i in range(num_agents):
                        print('Agent %d | Step %d | Goal: ' %(i, step_i), observations[i]['objectgoal'])
                    sys.stdout.flush()

                step_i += 1
            
            Agent_State.global_map = None
            Agent_State.origin_pose = None        
            if args.only_explore == 0:
                
                print('ended at step %d' % step_i)
                
                # Navigation metrics
                metrics = hab_env.get_metrics()
                print(metrics)
                
                
                multi_agent_multi_goals.append(metrics['multi_agent_multi_goals'])
                print('-' * 40)
                metric_lists = defaultdict(list)
                for ep_metric in multi_agent_multi_goals:
                    for k, v in ep_metric.items():
                        if k == 'asr' or k == 'avgsr' or k == 'tsr' or k == 'tspl' or k == 'tspl_max':
                            if isinstance(v, (int, float)):
                                metric_lists[k].append(v)
                metric_means = {k: float(np.mean(v)) for k, v in metric_lists.items()}
                print('ASR: %.4f | AvgSR: %.4f | TSR: %.4f | TSPL: %.4f | TSPL_MAX: %.4f' %
                    (metric_means.get('asr', 0.0),
                    metric_means.get('avgsr', 0.0),
                    metric_means.get('tsr', 0.0),
                    metric_means.get('tspl', 0.0),
                    metric_means.get('tspl_max', 0.0)))
                print('-' * 40)
                sys.stdout.flush()
                
        ep_i += 1
    

if __name__ == "__main__":
    main()
