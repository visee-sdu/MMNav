import os
import numpy as np
import torch
from torch.utils.data import Dataset

from agent.prediction import PEANUT_Prediction_Model
from arguments import get_args
class PSNetRefineDataset(Dataset):
    def __init__(self, root_dir, max_time_step=19):

        self.root_dir = root_dir
        self.filenames = sorted([f for f in os.listdir(root_dir) if f.endswith('.npz')])
        self.max_time = max_time_step

      
        
        self.img_path = []
        self.gt = []
        self.timestep = []
        args = get_args()
        self.prediction_model = PEANUT_Prediction_Model(args)
        
        for file in os.listdir(root_dir):
            
            for t in range(10):
                self.timestep.append(t)
                path = os.path.join(root_dir, file)
                self.img_path.append(path)
        
        print(f'Loaded {len(self.img_path)} samples from {root_dir}')
                

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        path = self.img_path[idx]
        time = self.timestep[idx]
        data = np.load(path)['maps']
        image = data[time]/255.0  # [14, 960, 960]
        output = self.prediction_model.get_prediction(image)
        output = torch.from_numpy(output).float()
        
        goals = range(4, 4 + 6)  # channels of semantic map
        mask = (image[1, :, :] > 0)
        gt_img = (data[-1, goals] * (1 - mask))
        gt_img = torch.from_numpy(gt_img).float()
        gt_img = gt_img / 255.0  # Normalize to [0, 1]
        
        
        return output, gt_img, torch.LongTensor([time])

if __name__ == "__main__":
    args = get_args()
    prediction_model = PEANUT_Prediction_Model(args)
    
    root_dir = '/Data/hm3d/saved_maps/train/'
    dst_dir = '/Data/hm3d/saved_maps/train_output/'
    for file in os.listdir(root_dir):
        for t in range(19):
            path = os.path.join(root_dir, file)
            data = np.load(path)['maps']
            image = data[t]/255.0
            output = prediction_model.get_prediction(image)
            
            output_name = file.split('.')[0]+f'_{t}'+'.npz'
            output_path = os.path.join(dst_dir, output_name)
            np.savez(output_path, output=output)
            print(f'Saved prediction for {output_name}')
            
            
    

