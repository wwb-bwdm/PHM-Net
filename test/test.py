import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from typing import Union, List
import math
from glob import glob
from scipy.spatial.distance import cdist

class TrainingConfig:
    pass

# ===================== Axial Attention模块 =====================
class AxialAttention(nn.Module):
    def __init__(self, in_channels, axis='height'):
        super().__init__()
        self.axis = axis
        self.in_channels = in_channels
        self.qkv_channels = max(in_channels // 8, 8)
        self.query_conv = nn.Conv2d(in_channels, self.qkv_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, self.qkv_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        num_groups = min(32, in_channels // 4) if in_channels >= 32 else max(1, in_channels // 4)
        self.norm = nn.GroupNorm(num_groups, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        if self.axis == 'height':
            Q = Q.permute(0, 2, 1, 3).reshape(B * H, self.qkv_channels, W)
            K = K.permute(0, 2, 1, 3).reshape(B * H, self.qkv_channels, W)
            V = V.permute(0, 2, 1, 3).reshape(B * H, C, W)
            attn = torch.bmm(Q.transpose(1, 2).float(), K.float())
            attn = attn / math.sqrt(self.qkv_channels)
            attn = F.softmax(attn, dim=-1).type_as(V)
            out = torch.bmm(V.float(), attn.float()).type_as(V)
            out = out.reshape(B, H, C, W).permute(0, 2, 1, 3)
        else:
            Q = Q.permute(0, 3, 1, 2).reshape(B * W, self.qkv_channels, H)
            K = K.permute(0, 3, 1, 2).reshape(B * W, self.qkv_channels, H)
            V = V.permute(0, 3, 1, 2).reshape(B * W, C, H)
            attn = torch.bmm(Q.transpose(1, 2).float(), K.float())
            attn = attn / math.sqrt(self.qkv_channels)
            attn = F.softmax(attn, dim=-1).type_as(V)
            out = torch.bmm(V.float(), attn.float()).type_as(V)
            out = out.reshape(B, W, C, H).permute(0, 2, 3, 1)
        return self.norm(out)

# ===================== RedECA模块 =====================
class RedECA_Module(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, red_map):
        y = self.avg_pool(x)
        red_weight = self.avg_pool(red_map)
        y = y * (1 + red_weight)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# ===================== 基础模块 =====================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        return self.relu(self.bn(self.conv(x)))

class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))

# ===================== RSU 模块 =====================
class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))
        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)
        self.red_eca = RedECA_Module(out_ch)
        self.axial_h = AxialAttention(out_ch, axis='height')
        self.axial_w = AxialAttention(out_ch, axis='width')
        self.ax_gamma_h = nn.Parameter(torch.zeros(1))
        self.ax_gamma_w = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, red_map: torch.Tensor = None) -> torch.Tensor:
        if red_map is None:
            if x.size(1) == 4:
                red_map = x[:, 3:4, :, :]
                x = x[:, :3, :, :]
            else:
                red_map = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        x_in = self.conv_in(x)
        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)
        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)
        x = x + x_in
        x = self.red_eca(x, red_map)
        x = x + self.ax_gamma_h * self.axial_h(x) + self.ax_gamma_w * self.axial_w(x)
        return x

class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([
            ConvBNReLU(out_ch, mid_ch),
            ConvBNReLU(mid_ch, mid_ch, dilation=2),
            ConvBNReLU(mid_ch, mid_ch, dilation=4),
            ConvBNReLU(mid_ch, mid_ch, dilation=8)
        ])
        self.decode_modules = nn.ModuleList([
            ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
            ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
            ConvBNReLU(mid_ch * 2, out_ch)
        ])
        self.red_eca = RedECA_Module(out_ch)
        self.axial_h = AxialAttention(out_ch, axis='height')
        self.axial_w = AxialAttention(out_ch, axis='width')
        self.ax_gamma_h = nn.Parameter(torch.zeros(1))
        self.ax_gamma_w = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, red_map: torch.Tensor = None) -> torch.Tensor:
        if red_map is None:
            if x.size(1) == 4:
                red_map = x[:, 3:4, :, :]
                x = x[:, :3, :, :]
            else:
                red_map = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        x_in = self.conv_in(x)
        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)
        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))
        x = x + x_in
        x = self.red_eca(x, red_map)
        x = x + self.ax_gamma_h * self.axial_h(x) + self.ax_gamma_w * self.axial_w(x)
        return x

# ===================== U2Net模型 =====================
class U2Net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1):
        super().__init__()
        self.encode_num = len(cfg["encode"])
        encode_list, side_list = [], []
        for c in cfg["encode"]:
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)
        decode_list = []
        for c in cfg["decode"]:
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if x.size(1) == 4:
            self.red_map = x[:, 3:4, :, :]
        else:
            self.red_map = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            if i == 0:
                x = m(x, self.red_map)
            else:
                if i != self.encode_num - 1:
                    x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
                red_map_resized = F.interpolate(self.red_map, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = m(x, red_map_resized)
            encode_outputs.append(x)
        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            red_map_resized = F.interpolate(self.red_map, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = m(torch.cat([x, x2], dim=1), red_map_resized)
            decode_outputs.insert(0, x)
        side_outputs = []
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)
        x = self.out_conv(torch.cat(side_outputs, dim=1))
        return torch.sigmoid(x)

def u2net_lite_4ch(out_ch: int = 1):
    cfg = {
        "encode": [[7, 4, 16, 64, False, False],
                   [6, 64, 16, 64, False, False],
                   [5, 64, 16, 64, False, False],
                   [4, 64, 16, 64, False, False],
                   [4, 64, 16, 64, True, False],
                   [4, 64, 16, 64, True, True]],
        "decode": [[4, 128, 16, 64, True, True],
                   [4, 128, 16, 64, False, True],
                   [5, 128, 16, 64, False, True],
                   [6, 128, 16, 64, False, True],
                   [7, 128, 16, 64, False, True]]
    }
    return U2Net(cfg, out_ch)

# ===================== 4通道数据预处理 =====================
class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensorWithRedMask:
    def __call__(self, image, target):
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        target = torch.from_numpy(target).unsqueeze(0).float()
        red_mask = self.generate_red_mask(image)
        red_mask_tensor = torch.from_numpy(red_mask).unsqueeze(0).float()
        image_4ch = torch.cat([image_tensor, red_mask_tensor], dim=0)
        return image_4ch, target

    @staticmethod
    def generate_red_mask(rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        return red_mask.astype(np.float32) / 255.0

class Resize4Ch:
    def __init__(self, size): self.size = size if isinstance(size, (list, tuple)) else (size, size)
    def __call__(self, image, target):
        rgb = image[:3]
        red = image[3:4]
        rgb = F.interpolate(rgb.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        red = F.interpolate(red.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        return torch.cat([rgb, red], dim=0), target

class Normalize4Ch:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    def __call__(self, image, target):
        return (image - self.mean) / self.std, target

class SODPresetEval4Ch:
    def __init__(self, base_size):
        self.transforms = Compose([
            ToTensorWithRedMask(),
            Resize4Ch(base_size),
            Normalize4Ch(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.25))
        ])
    def __call__(self, img, target): return self.transforms(img, target)

# ===================== 测试数据集 =====================
def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

class TestDataset4Ch(data.Dataset):
    def __init__(self, image_dir: str, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in ['.jpg','.jpeg','.png','.bmp']]
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        image = cv2.imread(os.path.join(self.image_dir, img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        dummy = np.zeros((h, w), np.float32)
        if self.transforms: image, dummy = self.transforms(image, dummy)
        return image, torch.zeros(1), img_file, (h, w)
    @staticmethod
    def collate_fn(batch):
        imgs, _, fns, sizes = list(zip(*batch))
        return cat_list(imgs), torch.zeros(1), fns, sizes

# ===================== 推理函数 =====================
@torch.no_grad()
def predict_mask(model, image_dir, save_dir, device, input_size=[320, 320]):
    os.makedirs(save_dir, exist_ok=True)
    dataset = TestDataset4Ch(image_dir, SODPresetEval4Ch(input_size))
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=TestDataset4Ch.collate_fn)
    model.eval()
    for images, _, filenames, sizes in tqdm(loader, desc="分割中", leave=False):
        images = images.to(device)
        preds = model(images)
        h, w = sizes[0]
        for i, pred in enumerate(preds):
            pred = pred.cpu().squeeze().numpy()
            pred = cv2.resize(pred, (w, h)) * 255
            cv2.imwrite(os.path.join(save_dir, filenames[i]), pred.astype(np.uint8))

# ===================== 条纹位移计算 =====================
def detect_stripe_centers(mask, min_width=10):
    h, w = mask.shape
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    white_pixels = np.where(binary == 255)
    if len(white_pixels[0]) == 0: raise ValueError("无条纹")
    x_coords = sorted(white_pixels[1])
    stripe_groups = []
    current_group = [x_coords[0]]
    for x in x_coords[1:]:
        if x - current_group[-1] <= 1: current_group.append(x)
        else:
            stripe_groups.append(current_group)
            current_group = [x]
    stripe_groups.append(current_group)
    centers = []
    for g in stripe_groups:
        L, R = min(g), max(g)
        if R-L+1 >= min_width:
            centers.append((L+R)/2)
    return centers

def select_16_benchmark_stripes_center(c):
    if len(c) < 16: raise ValueError("不足16条")
    s = (len(c)-16)//2
    return c[s:s+16], c[s:s+16][-1]

def match_benchmark_rightmost_to_target_left(r, t, th=50):
    tl = [x for x in t if x <= r]
    if not tl: raise ValueError("无左侧条纹")
    d = cdist([[r]], [[x] for x in tl])[0]
    i = np.argmin(d)
    if d[i]>th: raise ValueError("距离超限")
    return t.index(tl[i])

def target_select_16_from_rightmost(t, i):
    if i-15<0: raise ValueError("左向不足16条")
    return t[i-15:i+1]

def calculate_diffs_and_stats(b, t):
    return np.mean([abs(b[i]-t[i]) for i in range(16)])

def get_matched_image_pairs(a, b):
    exts = ["*.png","*.jpg","*.jpeg","*.bmp"]
    d1 = {os.path.basename(p):p for e in exts for p in glob(os.path.join(a,e))}
    d2 = {os.path.basename(p):p for e in exts for p in glob(os.path.join(b,e))}
    common = sorted(set(d1.keys()) & set(d2.keys()))
    return [(d1[n], d2[n], n) for n in common]

def calculate_displacement(bench_folder, target_folder, min_width=10, max_dist=50):
    pairs = get_matched_image_pairs(bench_folder, target_folder)
    for idx, (bp, tp, name) in enumerate(pairs, 1):
        try:
            bimg = cv2.imread(bp, 0)
            timg = cv2.imread(tp, 0)
            h, w = bimg.shape[:2]
            cf = (800 * 4.5) / w
            bc = detect_stripe_centers(bimg, min_width)
            b16, br = select_16_benchmark_stripes_center(bc)
            tc = detect_stripe_centers(timg, min_width)
            ti = match_benchmark_rightmost_to_target_left(br, tc, max_dist)
            t16 = target_select_16_from_rightmost(tc, ti)
            avg = calculate_diffs_and_stats(b16, t16) * cf
            print(f"{idx} {avg:.6f}")
        except:
            continue

# ===================== 一键运行配置 =====================
if __name__ == "__main__":
    # ========== 只需修改这里 ==========
    IMAGE_DIR_BENCH   = r"E:/spy/images2_800"    # 基准原图文件夹
    IMAGE_DIR_TARGET = r"E:/spy/images3_800"    # 目标原图文件夹
    MASK_DIR_BENCH    = r"test_masks_4"    # 基准掩码输出
    MASK_DIR_TARGET  = r"test_masks_5"    # 目标掩码输出
    MODEL_PATH       = r"E:\spy\测试代码\save_weights_red_Axial_new\model_best.pth"
    DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
    # =================================

    # 加载模型
    model = u2net_lite_4ch().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)


    # 1. 分割基准图
    predict_mask(model, IMAGE_DIR_BENCH, MASK_DIR_BENCH, DEVICE)
    # 2. 分割目标图
    predict_mask(model, IMAGE_DIR_TARGET, MASK_DIR_TARGET, DEVICE)
    # 3. 计算位移（只打印序号+位移）
    calculate_displacement(MASK_DIR_BENCH, MASK_DIR_TARGET)
