# train.py

from dataloader import SYSUData, RegDBData, LLCMData, VCMData, TrainData, TestData_Vis, TestData_Inf, GenIdx, IdentitySampler
from datamanager import process_gallery_sysu, process_query_sysu, process_test_regdb, process_query_llcm, process_gallery_llcm, process_query_vcm, process_gallery_vcm, process_train_regdb, process_train_sysu, process_train_llcm, process_train_vcm
import numpy as np
import torch.utils.data as data
from torch.autograd import Variable
import torch
from torch.cuda import amp
import torch.nn as nn
import os.path as osp
import os
import math
from model.make_model import build_vision_transformer
import time
from scheduler import create_scheduler
from loss.Triplet import TripletLoss
from loss.TripletCross import TripletCrossLoss
from loss.MSEL import MSEL
from loss.DCL import DCL
from utils import AverageMeter, set_seed, get_normal_affinity, cosine_similarity, get_old_proto_chunked
from transforms import transform_rgb, transform_rgb2gray, transform_thermal, transform_test
from optimizer import make_optimizer
from config.config import cfg
from eval_metrics import eval_sysu, eval_regdb
import argparse
from datamanager import VCM, VideoDataset_test_Inf, VideoDataset_test_Vis
import torch.nn.functional as F
import sys
from utils import Logger
from tqdm import tqdm


######################################################################################

parser = argparse.ArgumentParser(description="PMT Training")
parser.add_argument('--config_file', default='config/SYSU.yml',
                    help='path to config file', type=str)
parser.add_argument('--trial', default=1,
                    help='only for RegDB', type=int)
parser.add_argument('--resume', '-r', default='',
                    help='resume from checkpoint', type=str)
parser.add_argument('--model_path', default='save_model/',
                    help='model save path', type=str)
parser.add_argument('--num_workers', default=8,
                    help='number of data loading workers', type=int)
parser.add_argument('--start_test', default=0,
                    help='start to test in training', type=int)
parser.add_argument('--test_batch', default=128,
                    help='batch size for test', type=int)
parser.add_argument('--test_epoch', default=2,
                    help='test model every 2 epochs', type=int)
parser.add_argument('--save_epoch', default=2,
                    help='save model every 2 epochs', type=int)
parser.add_argument('--gpu', default='0',
                    help='gpu device ids for CUDA_VISIBLE_DEVICES', type=str)
parser.add_argument("opts", help="Modify config options using the command-line",
                    default=None,nargs=argparse.REMAINDER)
parser.add_argument("--logs-dir", help="Modify config options using the command-line",
                    default='logs_test',type=str)
parser.add_argument('--setting', type=int, default=1, choices=[1, 2, 3, 4, 5], help="training order setting")
parser.add_argument('--ema_weight', type=float, default=0.7)
parser.add_argument('--proto_weight', type=float, default=1.0)
parser.add_argument('--inter_weight', type=float, default=0.5)
parser.add_argument('--new_weight', type=float, default=1.0)
parser.add_argument('--cross_weight', type=float, default=0.5)
parser.add_argument('--prompt_weight', type=float, default=1.0)

args = parser.parse_args()

######################################################################################

if os.path.exists(args.logs_dir) is False:
    os.makedirs(args.logs_dir)
log_name = 'log.txt'
sys.stdout = Logger(osp.join(args.logs_dir, log_name))
# sys.stderr = sys.stdout   # tqdm যাতে লগে না ঢোকে, এটা বন্ধ রাখা ভালো
cfg.merge_from_list(args.opts)


cfg.MAX_EPOCH = 20   # অল্প এপকের জন্য


cfg.freeze()
print("==========\nArgs:{}\n==========".format(args))

set_seed(cfg.SEED)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

if args.setting == 1:
    training_set = ['sysu']
    #training_set = ['regdb', 'sysu', 'llcm', 'vcm']

for idx, dataset_name in enumerate(training_set):
    print ("[!INFO]: Start Training on:", dataset_name)
    if dataset_name == 'sysu':
        data_path = cfg.DATA_PATH_SYSU
        trainset_rgb = SYSUData(data_path, transform1=transform_rgb, transform2=transform_thermal)
    elif dataset_name == 'regdb':
        data_path = cfg.DATA_PATH_RegDB
        trainset_rgb = RegDBData(data_path, args.trial, transform1=transform_rgb, transform2=transform_thermal)
    elif dataset_name == 'llcm':
        data_path = cfg.DATA_PATH_LLCM
        trainset_rgb = LLCMData(data_path, transform1=transform_rgb, transform2=transform_thermal)
    elif dataset_name == 'vcm':
        data_path = cfg.DATA_PATH_VCM
        trainset_rgb = VCMData(data_path, transform1=transform_rgb, transform2=transform_thermal)
    color_pos_rgb, thermal_pos_rgb = GenIdx(trainset_rgb.train_color_label, trainset_rgb.train_thermal_label)

    num_classes = len(np.unique(trainset_rgb.train_color_label))
    model = build_vision_transformer(num_classes=num_classes,cfg = cfg)
    model.to(device)

    if idx > 0:
        args.resume = training_set[idx-1] + '.pth'
        model_path = os.path.join(args.logs_dir, args.resume)
        model.load_param(model_path)
        print('[!INFO] Loaded old model: {}'.format(args.resume))
        old_param_dict = torch.load(model_path)
        for i in old_param_dict:
            if i == 'classifier.weight':
                old_num_classes = old_param_dict[i].shape
        old_model = build_vision_transformer(num_classes=old_num_classes[0], cfg = cfg)
        old_model.to(device)
        old_model.load_param(model_path)
        print('[!INFO] Froze old model: {}'.format(args.resume))
        old_proto = torch.load(model_path.replace('.pth', '_proto.pth'))
        vis_features_mean = old_proto['vis_features_mean']
        vis_labels_named = old_proto['vis_labels_named']
        inf_features_mean = old_proto['inf_features_mean']
        inf_labels_named = old_proto['inf_labels_named']
    else:
        old_model = None

    criterion_ID = nn.CrossEntropyLoss()
    criterion_Tri = TripletLoss(margin=cfg.MARGIN, feat_norm='no')
    criterion_TriCros = TripletCrossLoss(margin=cfg.MARGINCROSS, feat_norm='no')
    criterion_TriCent = TripletCrossLoss(margin=cfg.MARGINCENTER, feat_norm='no')
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    criterion_PromptKD = nn.L1Loss(reduction='mean')

    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)
    scaler = amp.GradScaler()

    query_loaders, gall_loaders, query_labels, gall_labels, query_cams, gall_cams = [],[],[],[],[],[]
    for ii in range(idx+1):
        if training_set[ii] == 'sysu':
            data_path = cfg.DATA_PATH_SYSU
            query_img, query_label, query_cam = process_query_sysu(data_path, mode='all')
            queryset = TestData_Inf(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
            gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=0, gall_mode='single')
            gallset = TestData_Vis(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        elif training_set[ii] == 'regdb':
            data_path = cfg.DATA_PATH_RegDB
            query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
            queryset = TestData_Vis(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
            gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
            gallset = TestData_Inf(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        elif training_set[ii] == 'llcm':
            data_path = cfg.DATA_PATH_LLCM
            query_img, query_label, query_cam = process_query_llcm(data_path, modal=2)
            queryset = TestData_Vis(query_img, query_label, transform=transform_test, img_size=(cfg.W, cfg.H))
            gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, modal=1)
            gallset = TestData_Inf(gall_img, gall_label, transform=transform_test, img_size=(cfg.W, cfg.H))
        elif training_set[ii] == 'vcm':
            data_path = cfg.DATA_PATH_VCM
            processed_data = VCM()
            queryset = VideoDataset_test_Inf(processed_data.query,1,"video_test",transform_test,processed_data.query_cam)
            gallset = VideoDataset_test_Vis(processed_data.gallery,1,"video_test",transform_test,processed_data.gallary_cam)
            queryset.test_label = processed_data.query_labels
            gallset.test_label = processed_data.gallery_labels

        query_loaders.append(data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers))
        gall_loaders.append(data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers))
        query_labels.append(queryset.test_label)
        gall_labels.append(gallset.test_label)
        query_cams.append(query_cam if training_set[ii] == 'sysu' else None)
        gall_cams.append(gall_cam if training_set[ii] == 'sysu' else None)

    def train(epoch, model, scheduler, optimizer, scaler):
        # (আগের মতোই, কোন পরিবর্তন নেই)
        loss_meter = AverageMeter()
        loss_ce_meter = AverageMeter()
        loss_tri_meter = AverageMeter()
        acc_rgb_meter = AverageMeter()
        acc_ir_meter = AverageMeter()

        loss_meter.reset()
        loss_ce_meter.reset()
        loss_tri_meter.reset()
        acc_rgb_meter.reset()
        acc_ir_meter.reset()

        scheduler.step(epoch)
        model.train()

        for idx_, (input1, input2, label1, label2) in enumerate(trainloader):
            optimizer.zero_grad()
            input1 = input1.to(device)
            input2 = input2.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)
            labels = torch.cat((label1,label2),0)

            with amp.autocast(enabled=True):
                scores, feats, prompt_new = model(torch.cat([input1,input2]), torch.cat([torch.ones_like(label1),torch.zeros_like(label2)]))

                score1, score2 = scores.chunk(2,0)
                feat1, feat2 = feats.chunk(2,0)

                # Privacy noise
                if model.training:
                    noise_std = 0.005
                    feat1 = feat1 + torch.randn_like(feat1) * noise_std
                    feat2 = feat2 + torch.randn_like(feat2) * noise_std

                loss_id = criterion_ID(score1, label1.long()) + criterion_ID(score2, label2.long())
                loss_tri = criterion_Tri(feats, feats, labels)
                loss = loss_id + loss_tri

                feat1_norm = feat1
                feat2_norm = feat2

                def merge_feat(features_all, labels_all):
                    features_collect = {}
                    for feature, label in zip(features_all, labels_all):
                        label = label.cpu().item()
                        if label in features_collect:
                            features_collect[label].append(feature)
                        else:
                            features_collect[label] = [feature]
                    labels_named = list(set(labels_all))
                    labels_named.sort()
                    features_mean=[]
                    for x in labels_named:
                        x = x.cpu().item()
                        if x in features_collect.keys():
                            feats=torch.stack(features_collect[x])
                            feat_mean=feats.mean(dim=0)
                            features_mean.append(feat_mean)
                        else:
                            features_mean.append(torch.zeros_like(features_all[0]))
                    return torch.stack(features_mean), labels_named

                feat1_mean, feat1_mean_label = merge_feat(feat1_norm, label1)
                feat2_mean, feat2_mean_label = merge_feat(feat2_norm, label2)

                loss_tri_cent = 0.5 * criterion_TriCent(feat1_mean, feat2_mean, feat1_mean_label, feat2_mean_label) + 0.5 * criterion_TriCent(feat2_mean, feat1_mean, feat2_mean_label, feat1_mean_label)
                loss_tri_cross = 0.5 * criterion_TriCros(feat1_norm, feat2_mean, label1, feat2_mean_label) + 0.5 * criterion_TriCros(feat1_mean, feat2_norm, feat1_mean_label, label2)

                loss += args.new_weight * ((1 - args.cross_weight) * loss_tri_cent + args.cross_weight * loss_tri_cross)

                if old_model is not None:
                    old_model.eval()
                    with torch.no_grad():
                        feats_old_norm, prompt_old = old_model(torch.cat([input1,input2]), torch.cat([torch.ones_like(label1),torch.zeros_like(label2)]), fkd=True)
                        feats_old_norm, prompt_old = feats_old_norm.detach(), prompt_old.detach()

                        feat1_old, feat2_old = feats_old_norm.chunk(2,0)

                    vis_features_new = cosine_similarity(feat1, vis_features_mean)
                    vis_features_old = cosine_similarity(feat1_old, vis_features_mean)
                    inf_features_new = cosine_similarity(feat2, inf_features_mean)
                    inf_features_old = cosine_similarity(feat2_old, inf_features_mean)

                    vis_features_new = F.softmax(vis_features_new / 0.1, dim=1)
                    vis_features_old = F.softmax(vis_features_old / 0.1, dim=1)
                    inf_features_new = F.softmax(inf_features_new / 0.1, dim=1)
                    inf_features_old = F.softmax(inf_features_old / 0.1, dim=1)

                    vis_rel_new = get_normal_affinity(vis_features_new, 0.1)
                    vis_rel_old = get_normal_affinity(vis_features_old, 0.1)
                    inf_rel_new = get_normal_affinity(inf_features_new, 0.1)
                    inf_rel_old = get_normal_affinity(inf_features_old, 0.1)

                    div_1 = 0.5 * KLDivLoss(torch.log(vis_rel_new), vis_rel_old) + 0.5 * KLDivLoss(torch.log(inf_rel_new), inf_rel_old)

                    vis_features_new_ = cosine_similarity(feat1, inf_features_mean)
                    vis_features_old_ = cosine_similarity(feat1_old, inf_features_mean)
                    inf_features_new_ = cosine_similarity(feat2, vis_features_mean)
                    inf_features_old_ = cosine_similarity(feat2_old, vis_features_mean)

                    vis_features_new_ = F.softmax(vis_features_new_ / 0.1, dim=1)
                    vis_features_old_ = F.softmax(vis_features_old_ / 0.1, dim=1)
                    inf_features_new_ = F.softmax(inf_features_new_ / 0.1, dim=1)
                    inf_features_old_ = F.softmax(inf_features_old_ / 0.1, dim=1)

                    vis_rel_new_ = get_normal_affinity(vis_features_new_, 0.1)
                    vis_rel_old_ = get_normal_affinity(vis_features_old_, 0.1)
                    inf_rel_new_ = get_normal_affinity(inf_features_new_, 0.1)
                    inf_rel_old_ = get_normal_affinity(inf_features_old_, 0.1)

                    div_2 = 0.5 * KLDivLoss(torch.log(vis_rel_new_), vis_rel_old_) + 0.5 * KLDivLoss(torch.log(inf_rel_new_), inf_rel_old_)

                    loss += args.proto_weight * ((1 - args.inter_weight) * div_1 + args.inter_weight * div_2)
                    loss += args.prompt_weight * criterion_PromptKD(prompt_new, prompt_old)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            acc_rgb = (score1.max(1)[1] == label1).float().mean()
            acc_ir = (score2.max(1)[1] == label2).float().mean()

            loss_tri_meter.update(loss_tri.item())
            loss_ce_meter.update(loss_id.item())
            loss_meter.update(loss.item())
            acc_rgb_meter.update(acc_rgb, 1)
            acc_ir_meter.update(acc_ir, 1)

            torch.cuda.synchronize()
            if (idx_ + 1) % len(trainloader) == 0 :
                print('Epoch[{}] Iteration[{}/{}]'
                      ' Loss: {:.3f}, Tri:{:.3f} CE:{:.3f}, '
                      'Acc_RGB: {:.3f}, Acc_IR: {:.3f}, '
                      'Base Lr: {:.2e} '.format(epoch, (idx_+1),
                      len(trainloader), loss_meter.avg, loss_tri_meter.avg,
                      loss_ce_meter.avg, acc_rgb_meter.avg, acc_ir_meter.avg,
                      optimizer.state_dict()['param_groups'][0]['lr']))
        return model


    def test(model, query_loader, gall_loader, dataset = 'sysu', query_label=None, gall_label=None, query_cam=None, gall_cam=None):
        model.eval()
        print('[!INFO] Testing...')
        gall_feat = []
        with torch.no_grad():
            for batch_idx, (input, label, cam) in enumerate(gall_loader):
                input = Variable(input.cuda())
                feat = model(input, cam)
                gall_feat.append(feat.detach().cpu().numpy())
        gall_feat = np.concatenate(gall_feat, axis=0)

        query_feat = []
        with torch.no_grad():
            for batch_idx, (input, label, cam) in enumerate(query_loader):
                input = Variable(input.cuda())
                feat = model(input, cam)
                query_feat.append(feat.detach().cpu().numpy())
        query_feat = np.concatenate(query_feat, axis=0)

        distmat = -np.matmul(query_feat, np.transpose(gall_feat))
        if dataset == 'sysu':
            cmc, mAP, mInp = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        else:
            cmc, mAP, mInp = eval_regdb(distmat, query_label, gall_label)
        return cmc, mAP, mInp

    best_mAP = 0
    print('==> Start Training...')
    start_epoch = cfg.START_EPOCH
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler: scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'[!INFO] Resumed from epoch {checkpoint["epoch"]}')

    for epoch in range(start_epoch, cfg.MAX_EPOCH + 1):
        sampler_rgb = IdentitySampler(trainset_rgb.train_color_label, trainset_rgb.train_thermal_label, color_pos_rgb,thermal_pos_rgb, cfg.BATCH_SIZE, per_img=cfg.NUM_POS)
        trainset_rgb.cIndex = sampler_rgb.index1
        trainset_rgb.tIndex = sampler_rgb.index2
        trainloader = data.DataLoader(trainset_rgb, batch_size=cfg.BATCH_SIZE, sampler=sampler_rgb, num_workers=args.num_workers, drop_last=True, pin_memory=True)

        model = train(epoch, model, scheduler, optimizer, scaler)

        if epoch % 5 == 0 and epoch != 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
            }
            ckpt_path = osp.join(args.logs_dir, f'{dataset_name}_epoch{epoch}.pth')
            torch.save(checkpoint, ckpt_path)
            print(f'[!INFO] Checkpoint saved at {ckpt_path}')

        if epoch == cfg.MAX_EPOCH:
            try:
                if idx > 0:
                    model_path = os.path.join(args.logs_dir, args.resume)
                    model.merge_param(model_path, args.ema_weight)
                    print('[!INFO] Merge old model ......')
                torch.save(model.state_dict(), osp.join(args.logs_dir, training_set[idx] + '.pth'))

                # ========== GPU মেমরি পরিষ্কার ==========
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # =========================================

                print("[INFO] Starting prototype extraction (ultra‑safe, chunk=100, batch=1)...")
                vis_features_mean, vis_labels_named, inf_features_mean, inf_labels_named, vis_counts, inf_counts = \
                    get_old_proto_chunked(model, data_path, img_size=(cfg.W, cfg.H), chunk_size=100, batch_size=1)
                print("[INFO] Prototype extraction completed.")

                C = 1.0                     # L2 norm clipping bound
                target_epsilon = 0.1        # আপনি 10, 8, 5 দিয়ে পরীক্ষা করবেন
                target_delta = 1e-5

                # class‑wise sensitivity: Δf = C / n_i
                with torch.no_grad():
                    # Visible prototypes – সরাসরি গড় ব্যবহার
                    vis_sens = C / vis_counts.float().clamp(min=1)
                    sigma_vis = vis_sens.view(-1, 1).expand_as(vis_features_mean) * math.sqrt(2.0 * math.log(1.25 / target_delta)) / target_epsilon
                    noise_vis = torch.randn_like(vis_features_mean) * sigma_vis
                    vis_priv = F.normalize(vis_features_mean + noise_vis, p=2, dim=-1)

                    # Infrared prototypes
                    inf_sens = C / inf_counts.float().clamp(min=1)
                    sigma_inf = inf_sens.view(-1, 1).expand_as(inf_features_mean) * math.sqrt(2.0 * math.log(1.25 / target_delta)) / target_epsilon
                    noise_inf = torch.randn_like(inf_features_mean) * sigma_inf
                    inf_priv = F.normalize(inf_features_mean + noise_inf, p=2, dim=-1)
                
                
                # ── সংরক্ষণ ──
                proto_type = {
                    'vis_features_mean': vis_priv,
                    'vis_labels_named': vis_labels_named,
                    'inf_features_mean': inf_priv,
                    'inf_labels_named': inf_labels_named
                }
                torch.save(proto_type, osp.join(args.logs_dir, dataset_name + '_proto.pth'))

                # ========== আবার মেমরি পরিষ্কার ==========
                torch.cuda.empty_cache()
                # =========================================

                print("[INFO] Starting evaluation on test sets...")
                R1_list, mAP_list = [], []
                head_str, results_str, copy_str = '|', '|', ''
                mean_R1, mean_mAP = 0, 0
                for ii, testset_name in enumerate(training_set):
                    if ii > idx:
                        continue
                    cmc, mAP, mINP = test(model, query_loaders[ii], gall_loaders[ii],
                                         testset_name, query_labels[ii], gall_labels[ii],
                                         query_cams[ii], gall_cams[ii])
                    R1_list.append(cmc[0])
                    mAP_list.append(mAP)
                    head_str += testset_name + '|\t'
                    results_str += '{:.2f}/{:.2f}|\t'.format(mAP * 100, cmc[0] * 100)
                    copy_str += '{:.2f}\t{:.2f}\t'.format(mAP * 100, cmc[0] * 100)
                    mean_R1 += cmc[0]
                    mean_mAP += mAP
                    print("[INFO] Evaluation finished.")

                mean_R1 /= (idx + 1)
                mean_mAP /= (idx + 1)
                head_str += 'AVG|\t'
                results_str += '{:.2f}/{:.2f}|\t'.format(mean_mAP * 100, mean_R1 * 100)
                copy_str += '{:.2f}\t{:.2f}'.format(mean_mAP * 100, mean_R1 * 100)

                print("Results:")
                print(head_str)
                print(results_str)
                print(copy_str)

            except Exception as e:
                print(f"ERROR during evaluation: {e}")
                import traceback
                traceback.print_exc()
