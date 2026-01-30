from habitat.sims import make_sim
from nav.arguments import get_args
import os
import habitat
import json
import gzip
from habitat.datasets.pointnav.pointnav_dataset import DEFAULT_SCENE_PATH_PREFIX
import time

sdir = '/Data/hm3d/scene_datasets_v0.2'
default_dir = './data/scene_datasets/'

def load_distance_matrix(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 将字符串 "(1.2, 3.4, 0.0)" 转换回 tuple
    point2index = {eval(k): v for k, v in data['point2index'].items()}
    matrix = data['matrix']
    return matrix, point2index

def old_get_all_points_distance(points, sim):
    content = {}
    point2index = {}
    n = 0
    for point in points:
        p = tuple(point)
        if p not in point2index.keys():
            point2index[p]=n
            n += 1
    matrix = [[-1 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        a = tuple(points[i])
        a_index = point2index[a]
        for j in range(n):
            b = tuple(points[j])
            b_index = point2index[b]
            if a_index == b_index:
                matrix[a_index][b_index]=0
            else:
                if matrix[a_index][b_index] == -1:
                    matrix[a_index][b_index] = sim.geodesic_distance(points[i], points[j])
                    matrix[b_index][a_index] = matrix[a_index][b_index]
                else:
                    matrix[a_index][b_index] = matrix[b_index][a_index]
    
    converted_point2index = {str(k): v for k, v in point2index.items()}
    content = {
        "point2index": converted_point2index,
        "matrix": matrix  # list[list[float or int]]
    }
            
    return content

def get_all_points_distance(points, sim):
    point2index = {}
    unique_points = []
    
    for point in points:
        p = tuple(point)
        if p not in point2index:
            point2index[p] = len(unique_points)
            unique_points.append(p)

    n = len(unique_points)
    print(f'The number of points is {n}')
    matrix = [[-1 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
            elif matrix[i][j] == -1:
                dist = sim.geodesic_distance(unique_points[i], unique_points[j])
                matrix[i][j] = dist
                matrix[j][i] = dist  # 对称矩阵

    # Convert keys to string for JSON serialization
    converted_point2index = {str(k): v for k, v in point2index.items()}

    return {
        "point2index": converted_point2index,
        "matrix": matrix
    }


def from_json(json_str):
    deserialized = json.loads(json_str)
    points = []
    
    episodes = deserialized['episodes']
    if episodes[0]['scene_dataset_config'].startswith(default_dir):
        scene_dataset_config = episodes[0]['scene_dataset_config'][len(DEFAULT_SCENE_PATH_PREFIX): ]
    else:
        scene_dataset_config = episodes[0]['scene_dataset_config']
    scene_dataset_config = os.path.join(sdir,scene_dataset_config)
    
    if episodes[0]['scene_id'].startswith(DEFAULT_SCENE_PATH_PREFIX):
        scene_id = episodes[0]['scene_id'][
            len(DEFAULT_SCENE_PATH_PREFIX) :
        ]
    else:
        scene_id = episodes[0]['scene_id']

    scene_id = os.path.join(sdir, scene_id)
    # scene_id = episodes[0]['scene_id']
    # additional_obj_config_paths = episodes[0]['additional_obj_config_paths']
    
    
    for ep in episodes:
        # import pdb; pdb.set_trace()
        start_pisitions = ep['start_position']
        for sp in start_pisitions:
            points.append(sp)
        
    goals = deserialized['goals_by_category']
    for k, v in goals.items():
        for view in v:
            view_points = view['view_points']
            for point in view_points:
                points.append(point['agent_state']['position'])
    return points, scene_dataset_config, scene_id
                

def save_ideal_dist(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)            
    
if __name__ == "__main__":
    args = get_args()
    args.only_explore = 0  
    
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    config.defrost()
    
    
    config.SEED = 100
    # config.SEED = 0
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sem_gpu_id
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 1
    config.DATASET.SPLIT = 'val'
    config.freeze()
    
    
    scene_dir = '/Data/hm3d/datasets/objectnav/objectnav_hm3d_v2/val/content_samerotation508'
    scene_paths = []
    for file in os.listdir(scene_dir):
        scene_paths.append(os.path.join(scene_dir, file))
    
    
    for scene_path in scene_paths:
        print("scane_path is: ", scene_path)
        with gzip.open(scene_path, "rt") as f:
            json_str = f.read()
        points, scene_dataset_config, scene_id = from_json(json_str)  # /Data/hm3d/scene_datasets_v0.2/
        additional_obj_config_paths = []
        config.defrost()
        config.SIMULATOR.SCENE_DATASET = (
            scene_dataset_config
        )  # /data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json
        config.SIMULATOR.SCENE = scene_id  # hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb
        config.SIMULATOR.ADDITIONAL_OBJECT_PATHS = (
            additional_obj_config_paths
        )  # []
        config.freeze()
        with make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR) as sim:
        
        
            start_time = time.time()
            dist_matrix = get_all_points_distance(points, sim)
            end_time = time.time()
            print(f'Scene {scene_path} took {end_time - start_time:.2f} seconds.')
            
            fname = scene_path.split('/')[-1].split('.')[0] + '.json'
            output_path = os.path.join('/home/huangbw/navigation/PEANUT/ideal_dist/Point_Point_Distance', fname)
            
            start_time = time.time()
            save_ideal_dist(dist_matrix, output_path)
            end_time = time.time()
            print(f'Save {fname} took {end_time - start_time:.2f} seconds.')
        
        