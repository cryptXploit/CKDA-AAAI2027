import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
import sys
import errno

# ========== Seed & AverageMeter ==========

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# ========== Memory‑Efficient Prototype Extraction ==========

def get_old_proto_chunked(model, data_path, img_size, chunk_size=100, batch_size=1):
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    # Memory‑mapped loading
    rgb_imgs = np.load(os.path.join(data_path, 'train_rgb_resized_img.npy'), mmap_mode='r')
    rgb_labels = np.load(os.path.join(data_path, 'train_rgb_resized_label.npy'), mmap_mode='r')
    ir_imgs = np.load(os.path.join(data_path, 'train_ir_resized_img.npy'), mmap_mode='r')
    ir_labels = np.load(os.path.join(data_path, 'train_ir_resized_label.npy'), mmap_mode='r')

    all_feat_vis, all_id_vis = [], []
    all_feat_inf, all_id_inf = [], []
    
    # ফিচার ডাইমেনশন ধরার জন্য ভ্যারিয়েবল
    feat_dim = None

    for imgs, labs, mod_val, name in [(rgb_imgs, rgb_labels, 1, 'VIS'), (ir_imgs, ir_labels, 0, 'IR')]:
        n = len(labs)
        with tqdm(total=n, desc=f'{name} total', unit='img') as pbar:
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk_imgs = imgs[start:end]
                chunk_labels = labs[start:end]

                chunk_t = torch.from_numpy(chunk_imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
                lbl_t = torch.from_numpy(chunk_labels.astype(np.int64)).long()
                mod_t = torch.full((len(lbl_t),), mod_val, dtype=torch.long)

                dataset = TensorDataset(chunk_t, lbl_t, mod_t)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

                for inp, lbl_batch, mod_batch in loader:
                    inp = inp.cuda()
                    mod_batch = mod_batch.cuda()
                    inp = (inp - mean) / std

                    with torch.no_grad():
                        feats = model(inp, mod_batch)    # (scores, features, prompt)
                        
                        if feat_dim is None:
                            feat_dim = feats.shape[1]          # e.g., 768
                        
                        # Per‑sample L2 clipping
                        C = 1.0
                        norms = feats.norm(p=2, dim=-1, keepdim=True)
                        scale = torch.clamp(C / (norms + 1e-12), max=1.0)
                        feats = feats * scale

                    if mod_val == 1:
                        all_feat_vis.append(feats.cpu())
                        all_id_vis.extend(lbl_batch.tolist())
                    else:
                        all_feat_inf.append(feats.cpu())
                        all_id_inf.extend(lbl_batch.tolist())

                    pbar.update(len(lbl_batch))

    def make_proto(feat_list, ids):
        if not feat_list:
            return torch.empty(0, feat_dim if feat_dim is not None else 0), []
        feats = torch.cat(feat_list, dim=0)
        unique = sorted(set(ids))
        means = []
        counts = []
        for lbl in unique:
            mask = [i for i, x in enumerate(ids) if x == lbl]
            means.append(feats[mask].mean(dim=0, keepdim=True))
            counts.append(len(mask))
        return torch.cat(means, dim=0), unique, torch.tensor(counts).cuda()

    dim = feat_dim if feat_dim is not None else 0
    vis_proto, vis_names, vis_counts = make_proto(all_feat_vis, all_id_vis)
    inf_proto, inf_names, inf_counts = make_proto(all_feat_inf, all_id_inf)
    
    # 🟢 6 টি ভ্যালু রিটার্ন করছে, যা train.py এর লেয়ার ৩ এর জন্য একদম পারফেক্ট
    return vis_proto, vis_names, inf_proto, inf_names, vis_counts, inf_counts

# ========== Affinity and Cosine Similarity ==========

def get_normal_affinity(x, Norm=100):
    pre_matrix_origin = cosine_similarity(x, x)
    pre_affinity_matrix = F.softmax(pre_matrix_origin * Norm, dim=1)
    return pre_affinity_matrix

def cosine_similarity(input1, input2):
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = torch.mm(input1_normed, input2_normed.t())
    return distmat

# ========== Logger ==========

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
            self.file.flush()

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
