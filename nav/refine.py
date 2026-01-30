import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import PSNetRefineDataset

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        attn = self.fc2(F.relu(self.fc1(avg_pool)))
        return x * self.sigmoid(attn)


class UAGRM_noconf(nn.Module):
    def __init__(self, in_channels=6, t_embed_dim=6):
        super().__init__()
        self.t_embed = nn.Embedding(20, t_embed_dim)

        self.refine_block = nn.Sequential(
            nn.Conv2d(in_channels + t_embed_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ChannelAttention(64),  # 简单注意力机制
            nn.Conv2d(64, in_channels, 1)
        )


    def forward(self, x, t):
        """
        x: [B, C, H, W], predicted map
        t: [B], int time step
        """
        B, C, H, W = x.shape
        t_vec = self.t_embed(t)  # [B, d]
        t_map = t_vec.view(B, -1, 1, 1).expand(-1, -1, H, W)

        x_cat = torch.cat([x, t_map], dim=1)
        refined = self.refine_block(x_cat)
        return refined  # 类似 residual mask blending


def computer_loss(output, gt):
    loss = F.binary_cross_entropy_with_logits(output, gt, reduction='mean')
    return loss


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(B, C)
        attn = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(B, C, 1, 1)
        x = x * attn

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial
        return x



def test1():
    dataset = PSNetRefineDataset('/Data/hm3d/saved_maps/train/')
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)

    uagrm = UAGRM_noconf().cuda()
    
    uagrm.train()
    optimizer = torch.optim.Adam(uagrm.parameters(), lr=1e-4)
    num_epochs = 5
    for epoch in range(num_epochs):
        step = 0
        for x, gt, t in loader:
            x, gt, t = x.cuda(), gt.cuda(), t.cuda()
            optimizer.zero_grad()  # 

            refined = uagrm(x, t)
            

            loss = computer_loss(refined, gt)
            loss.backward()
            optimizer.step()
            

            step += 1
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        torch.save(uagrm.state_dict(), f'nav/uagrm_pred60k/uagrm_epoch_{epoch+1}.pth')


        
if __name__ == "__main__":
    test1()
    
    
    
    
    


