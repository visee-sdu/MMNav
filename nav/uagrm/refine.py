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

class UAGRM(nn.Module):
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

        self.confidence_head = nn.Sequential(
            nn.Conv2d(in_channels + t_embed_dim, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),     # [B, 1, H, W]
            nn.Sigmoid()
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
        conf = self.confidence_head(x_cat)
        y = refined * conf + x * (1 - conf)
        import pdb; pdb.set_trace()
        conf = 1.0
        return refined * conf + x * (1 - conf)  # 类似 residual mask blending


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


class UAGRM_Large(nn.Module):
    def __init__(self, in_channels=6, t_embed_dim=6):  # ⬅️ 时间嵌入维度提高一倍
        super().__init__()
        self.t_embed = nn.Embedding(20, t_embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + t_embed_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            CBAM(64),  # ⬅️ 更强的注意力机制

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, in_channels, 1)  # 输出和输入同维度，便于 residual blend
        )

        self.confidence_head = nn.Sequential(
            nn.Conv2d(in_channels + t_embed_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x, t):
        """
        x: [B, C, H, W]
        t: [B] int
        """
        B, C, H, W = x.shape
        t_vec = self.t_embed(t)  # [B, D]
        t_map = t_vec.view(B, -1, 1, 1).expand(-1, -1, H, W)

        x_cat = torch.cat([x, t_map], dim=1)
        refined = self.encoder(x_cat)
        conf = self.confidence_head(x_cat)

        return refined * conf + x * (1 - conf)  # ⬅️ residual mask blending


def test1():
    dataset = PSNetRefineDataset('/Data/hm3d/saved_maps/val/')
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)

    uagrm = UAGRM().cuda()
    
    # state_dict = torch.load(
    #     '/home/huangbw/navigation/PEANUT/nav/uagrm/uagrm_epoch_2_step_6000.pth',
    # )
    state_dict = torch.load(
        '/home/huangbw/navigation/PEANUT/nav/uagrm_pred60k/uagrm_epoch_5.pth',
    )
    uagrm.load_state_dict(state_dict)
    uagrm.train()
    optimizer = torch.optim.Adam(uagrm.parameters(), lr=1e-4)
    num_epochs = 5
    for epoch in range(num_epochs):
        step = 0
        for x, gt, t in loader:
            x, gt, t = x.cuda(), gt.cuda(), t.cuda()
            optimizer.zero_grad()  # ✅ 清空梯度

            refined = uagrm(x, t)
            

            loss = computer_loss(refined, gt)
            loss.backward()
            optimizer.step()
            
            # if step%500 ==0:
            #     print(f'Epoch {epoch+1}/{num_epochs}, Step {step}, Loss: {loss.item()}')
            
            # if step % 5000 == 0:
            #     print(f'Saving model at epoch {epoch+1}, step {step}')
            #     torch.save(uagrm.state_dict(), f'nav/uagrm_/uagrm_epoch_{epoch+1}_step_{step}.pth')
            step += 1
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        torch.save(uagrm.state_dict(), f'nav/uagrm_pred60k/uagrm_epoch_{epoch+1}.pth')


def test2():
    # 加载（同样进行过滤）：
    path1 = '/home/huangbw/navigation/PEANUT/nav/uagrm_pred60k/uagrm_epoch_5.pth'
    path2 = '/home/huangbw/navigation/PEANUT/nav/uagrm_pred60k/uagrm_epoch_5_noconf.pth'
    state_dict = torch.load(path1)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("confidence_head")}

    model = UAGRM_noconf()
    model.load_state_dict(state_dict, strict=False)        
    torch.save(state_dict, path2)
        
if __name__ == "__main__":
    test2()
    
    
    
    
    
# def compute_loss(psnet_out, refined_out, confidence, gt):
#     # BCE loss between refined output and GT
#     gt_norm = gt / 255.  # Ensure in range [0, 1]
#     pred_loss = F.binary_cross_entropy_with_logits(refined_out, gt_norm)

#     # Confidence supervision label from PSPNet output
#     with torch.no_grad():
#         pseudo_conf = 1. - F.binary_cross_entropy_with_logits(
#             psnet_out, gt_norm, reduction='none'
#         ).mean(dim=1, keepdim=True)  # [B, 1, H, W]
#         pseudo_conf = pseudo_conf.clamp(0., 1.)

#     # MSE loss between predicted confidence and pseudo label
#     conf_loss = F.mse_loss(confidence, pseudo_conf)

#     # Total loss
#     alpha = 0.2
#     total_loss = pred_loss + alpha * conf_loss

#     return total_loss, pred_loss, conf_loss

# final_out, refined_out, conf = model(psnet_out, t)

# loss_main = F.binary_cross_entropy_with_logits(final_out, gt)
# loss_refined = F.binary_cross_entropy_with_logits(refined_out, gt)

# with torch.no_grad():
#     pseudo_conf = 1. - F.binary_cross_entropy_with_logits(psnet_out, gt, reduction='none').mean(dim=1, keepdim=True)
#     pseudo_conf = pseudo_conf.clamp(0., 1.)
# loss_conf = F.mse_loss(conf, pseudo_conf)

# total_loss = loss_main + 0.5 * loss_refined + 0.2 * loss_conf



# psrefine = PSRefineModel(base_psnet=psnet)
# optimizer = torch.optim.Adam(psrefine.parameters(), lr=1e-4)

# for data in dataloader:
#     img = data['img'].cuda()           # [B, 14, H, W]
#     gt = data['gt_semantic_seg'].cuda()  # [B, 6, H, W]
#     t = data['step_t'].float().cuda()  # [B] 当前时间步

#     psnet_out, refined_out, confidence = psrefine(img, t)
#     total_loss, pred_loss, conf_loss = compute_loss(psnet_out, refined_out, confidence, gt)

#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()

