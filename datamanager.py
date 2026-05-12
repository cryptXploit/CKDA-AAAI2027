from __future__ import print_function, absolute_import
import os
import numpy as np
import random
import pickle
from tqdm import tqdm
from dataloader import load_data
from PIL import Image
import os.path as osp
import torch.utils.data as data
import math
import torch

def process_query_sysu(data_path, mode='all', relabel=False):

    if mode== 'all':
        ir_cameras = ['cam3','cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3','cam6']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)

    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, mode='all', trial=0, relabel=False, gall_mode='single'):

    random.seed(trial)

    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                if gall_mode == 'single':
                    files_rgb.append(random.choice(new_files))
                if gall_mode == 'multi':
                    files_rgb.append(np.random.choice(new_files, 10, replace=False))
    gall_img = []
    gall_id = []
    gall_cam = []

    for img_path in files_rgb:
        if gall_mode == 'single':
            camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            gall_img.append(img_path)
            gall_id.append(pid)
            gall_cam.append(camid)

        if gall_mode == 'multi':
            for i in img_path:
                camid, pid = int(i[-15]), int(i[-13:-9])
                gall_img.append(i)
                gall_id.append(pid)
                gall_cam.append(camid)

    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_test_regdb(img_dir, trial=1, modal='visible'):
    if modal == 'visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal == 'thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'

    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, np.array(file_label)

def process_query_llcm(data_path, modal):
    if modal == 1:
        cameras = ['test_vis/cam1', 'test_vis/cam2', 'test_vis/cam3', 'test_vis/cam4', 'test_vis/cam5', 'test_vis/cam6',
                   'test_vis/cam7', 'test_vis/cam8', 'test_vis/cam9']
    elif modal == 2:
        cameras = ['test_nir/cam1', 'test_nir/cam2', 'test_nir/cam4', 'test_nir/cam5', 'test_nir/cam6', 'test_nir/cam7',
                   'test_nir/cam8', 'test_nir/cam9']

    file_path = os.path.join(data_path, 'idx/test_id.txt')
    files_list = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_list.extend(new_files)

    query_img = []
    query_id = []
    query_cam = []

    for img_path in files_list:
        camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_llcm(data_path, modal, trial = None):
    if modal == 1:
        cameras = ['test_vis/cam1', 'test_vis/cam2', 'test_vis/cam3', 'test_vis/cam4', 'test_vis/cam5', 'test_vis/cam6',
                   'test_vis/cam7', 'test_vis/cam8', 'test_vis/cam9']
    elif modal == 2:
        cameras = ['test_nir/cam1', 'test_nir/cam2', 'test_nir/cam4', 'test_nir/cam5', 'test_nir/cam6', 'test_nir/cam7',
                   'test_nir/cam8', 'test_nir/cam9']

    file_path = os.path.join(data_path, 'idx/test_id.txt')
    files_list = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_list.append(random.choice(new_files))

    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_list:
        camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_query_vcm(data_path, modal):
    if modal == 1:  # rgb
        with open(data_path + 'query_vis_path_img.pkl', 'rb') as f:
            gall = pickle.load(f)
    elif modal == 2:  # ir
        with open(data_path + 'query_ir_path_img.pkl', 'rb') as f:
            gall = pickle.load(f)

    gall_img = []
    gall_id = []
    gall_cam = []
    for i in range(len(gall)):
        pid = gall[i][0]
        for j, imgs in enumerate(gall[i][1]):
            camid = imgs[0]
            files_list = sorted(imgs[1])
            for img_path in files_list:
                gall_img.append(img_path)
                gall_id.append(pid)
                gall_cam.append(camid)

    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_gallery_vcm(data_path, modal, seed = None):
    if modal == 1:  # rgb
        with open(data_path + 'gallery_vis_path_img.pkl', 'rb') as f:
            gall = pickle.load(f)
    elif modal == 2:  # ir
        with open(data_path + 'gallery_ir_path_img.pkl', 'rb') as f:
            gall = pickle.load(f)

    gall_img = []
    gall_id = []
    gall_cam = []
    for i in range(len(gall)):
        pid = gall[i][0]
        for j, imgs in enumerate(gall[i][1]):
            camid = imgs[0]
            files_list = sorted(imgs[1])
            img_path = random.choice(files_list)
            gall_img.append(img_path)
            gall_id.append(pid)
            gall_cam.append(camid)

    return gall_img, np.array(gall_id), np.array(gall_cam)






def process_train_regdb(data_path, trial = 1):

    train_color_list = data_path + 'idx/train_visible_{}'.format(trial) + '.txt'
    train_thermal_list = data_path + 'idx/train_thermal_{}'.format(trial) + '.txt'

    color_img_file, train_color_label = load_data(train_color_list)
    thermal_img_file, train_thermal_label = load_data(train_thermal_list)

    train_color_image = []
    for i in range(len(color_img_file)):
        fp = open(data_path + color_img_file[i], 'rb')
        img = Image.open(fp)
        img = np.array(img)
        train_color_image.append(Image.fromarray(img.astype(np.uint8)))
        fp.close()

    train_thermal_image = []
    for i in range(len(thermal_img_file)):
        fp = open(data_path + thermal_img_file[i], 'rb')
        img = Image.open(fp)
        img = np.array(img)
        train_thermal_image.append(Image.fromarray(img.astype(np.uint8)))
        fp.close()

    return train_color_image + train_color_image, train_color_label + train_thermal_label, [1 for i in train_color_image] + [0 for i in train_thermal_image]

def process_train_sysu(data_path):

    train_color_image = np.load(data_path + 'train_rgb_resized_img.npy')
    train_color_label = np.load(data_path + 'train_rgb_resized_label.npy')
    train_thermal_image = np.load(data_path + 'train_ir_resized_img.npy')
    train_thermal_label = np.load(data_path + 'train_ir_resized_label.npy')

    train_color_list, train_color_label_list = [], []
    for img_array, img_label in zip(train_color_image, train_color_label):
        train_color_list.append(Image.fromarray(img_array.astype(np.uint8)))
        train_color_label_list.append(img_label)
    train_thermal_list, train_thermal_label_list = [], []
    for img_array, img_label in zip(train_thermal_image, train_thermal_label):
        train_thermal_list.append(Image.fromarray(img_array.astype(np.uint8)))
        train_thermal_label_list.append(img_label)

    return train_color_list + train_thermal_list, train_color_label_list + train_thermal_label_list, [1 for i in train_color_image] + [0 for i in train_thermal_image]

def process_train_llcm(data_path):

    train_color_image = np.load(data_path + 'train_rgb_resized_img.npy')
    train_color_label = np.load(data_path + 'train_rgb_resized_label.npy')
    train_thermal_image = np.load(data_path + 'train_ir_resized_img.npy')
    train_thermal_label = np.load(data_path + 'train_ir_resized_label.npy')

    train_color_list, train_color_label_list = [], []
    for img_array, img_label in zip(train_color_image, train_color_label):
        train_color_list.append(Image.fromarray(img_array.astype(np.uint8)))
        train_color_label_list.append(img_label)
    train_thermal_list, train_thermal_label_list = [], []
    for img_array, img_label in zip(train_thermal_image, train_thermal_label):
        train_thermal_list.append(Image.fromarray(img_array.astype(np.uint8)))
        train_thermal_label_list.append(img_label)
    
    return train_color_list + train_thermal_list, train_color_label_list + train_thermal_label_list, [1 for i in train_color_image] + [0 for i in train_thermal_image]

def process_train_vcm(data_path):

    train_color_path = np.load(data_path + 'train_rgb_path_img.npy')
    train_color_label = np.load(data_path + 'train_rgb_path_label.npy')
    train_thermal_path = np.load(data_path + 'train_ir_path_img.npy')
    train_thermal_label = np.load(data_path + 'train_ir_path_label.npy')
    
    train_color_list, train_color_label_list = [], []
    for img_array, img_label in zip(train_color_path, train_color_label):
        train_color_list.append(Image.fromarray(img_array.astype(np.uint8)))
        train_color_label_list.append(img_label)
    train_thermal_list, train_thermal_label_list = [], []
    for img_array, img_label in zip(train_thermal_path, train_thermal_label):
        train_thermal_list.append(Image.fromarray(img_array.astype(np.uint8)))
        train_thermal_label_list.append(img_label)
    
    return train_color_list + train_thermal_list, train_color_label_list + train_thermal_label_list, [1 for i in train_color_path] + [0 for i in train_thermal_path]

class VCM(object):
    # modify the data path here according to yours
    root = '/home/cuizhenyu/VI-LReID/Dataset/HITSZ-VCM/'
    # training data
    train_name_path = osp.join(root,'info/train_name.txt')
    track_train_info_path = osp.join(root,'info/track_train_info.txt')
    # testing data
    test_name_path = osp.join(root,'info/test_name.txt')
    track_test_info_path = osp.join(root,'info/track_test_info.txt')
    query_IDX_path = osp.join(root,'info/query_IDX.txt')

    def __init__(self,min_seq_len=12):
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        track_train = self._get_tracks(self.track_train_info_path)

        # for test
        test_names = self._get_names(self.test_name_path)
        track_test = self._get_tracks(self.track_test_info_path)
        query_IDX =  self._get_query_idx(self.query_IDX_path)
        query_IDX -= 1

        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]

        #---------visible to infrared-----------
        gallery_IDX_1 = self._get_query_idx(self.query_IDX_path)
        gallery_IDX_1 -= 1
        track_gallery_1 = track_test[gallery_IDX_1,:]

        query_IDX_1 = [j for j in range(track_test.shape[0]) if j not in gallery_IDX_1]
        track_query_1 = track_test[query_IDX_1,:]
        #-----------------------------------------

        train_ir, num_train_tracklets_ir,num_train_imgs_ir,train_rgb, num_train_tracklets_rgb,num_train_imgs_rgb,num_train_pids,ir_label,rgb_label = \
          self._process_data_train(train_names,track_train,relabel=True,min_seq_len=min_seq_len)


        query, num_query_tracklets, num_query_pids, num_query_imgs,query_cam,query_labels = \
          self._process_data_test(test_names, track_query, relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs,gallary_cam,gallery_labels = \
          self._process_data_test(test_names, track_gallery, relabel=False, min_seq_len=min_seq_len)


        #--------visible to infrared-----------
        query_1, num_query_tracklets_1, num_query_pids_1, num_query_imgs_1,query_cam1,query_labels1 = \
          self._process_data_test(test_names, track_query_1, relabel=False, min_seq_len=min_seq_len)

        gallery_1, num_gallery_tracklets_1, num_gallery_pids_1, num_gallery_imgs_1,gallery_cam1,gallery_labels1 = \
          self._process_data_test(test_names, track_gallery_1, relabel=False, min_seq_len=min_seq_len)
        #---------------------------------------

        # print("=> VCM loaded")
        # print("Dataset statistics:")
        # print("---------------------------------")
        # print("subset      | # ids | # tracklets")
        # print("---------------------------------")
        # print("train_ir    | {:5d} | {:8d}".format(num_train_pids,num_train_tracklets_ir))
        # print("train_rgb   | {:5d} | {:8d}".format(num_train_pids,num_train_tracklets_rgb))
        # print("query       | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        # print("gallery     | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        # print("---------------------------------")

        self.train_ir = train_ir
        self.train_rgb = train_rgb
        self.ir_label = ir_label
        self.rgb_label = rgb_label

        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_query_tracklets = num_query_tracklets
        self.num_gallery_tracklets = num_gallery_tracklets
        self.query_cam = query_cam
        self.gallary_cam = gallary_cam
        self.query_labels = query_labels
        self.gallery_labels = gallery_labels

        #------- visible to infrared------------
        self.query_1 = query_1
        self.gallery_1 = gallery_1

        self.num_query_pids_1 = num_query_pids_1
        self.num_gallery_pids_1 = num_gallery_pids_1
        self.num_query_tracklets_1 = num_query_tracklets_1
        self.num_gallery_tracklets_1 = num_gallery_tracklets_1
        self.query_cam1 = query_cam1
        self.gallery_cam1 = gallery_cam1
        self.query_labels1 = query_labels1
        self.gallery_labels1 = gallery_labels1
        #---------------------------------------


    def _check_before_run(self):
        """check befor run """
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))

    def _get_names(self,fpath):
        """get image name, retuen name list"""
        names = []
        with open(fpath,'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _get_tracks(self,fpath):
        """get tracks file"""
        names = []
        with open(fpath,'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')

                tmp = new_line.split(' ')[0:]

                tmp = list(map(int, tmp))
                names.append(tmp)
        names = np.array(names)
        return names


    def _get_query_idx(self, fpath):
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')

                tmp = new_line.split(' ')[0:]


                tmp = list(map(int, tmp))
                idxs = tmp
        idxs = np.array(idxs)
        return idxs

    def _process_data_train(self,names,meta_data,relabel=False,min_seq_len=0):
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,3].tolist()))
        num_pids = len(pid_list)

        # dict {pid : label}
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        
        tracklets_ir = []
        num_imgs_per_tracklet_ir = []
        ir_label = []

        tracklets_rgb = []
        num_imgs_per_tracklet_rgb = []
        rgb_label = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            m,start_index,end_index,pid,camid = data
            if relabel: pid = pid2label[pid]

            if m == 1:
                img_names = names[start_index-1:end_index]
                img_ir_paths = [osp.join(self.root,decoder_pic_path("Train",img_name)) for img_name in img_names]
                if len(img_ir_paths) >= min_seq_len:
                    img_ir_paths = tuple(img_ir_paths)
                    ir_label.append(pid)
                    tracklets_ir.append((img_ir_paths,pid,camid))
                    # same id
                    num_imgs_per_tracklet_ir.append(len(img_ir_paths))
            else:
                img_names = names[start_index-1:end_index]
                img_rgb_paths = [osp.join(self.root,decoder_pic_path("Train",img_name)) for img_name in img_names]
                if len(img_rgb_paths) >= min_seq_len:
                    img_rgb_paths = tuple(img_rgb_paths)
                    rgb_label.append(pid)
                    tracklets_rgb.append((img_rgb_paths,pid,camid))
                    #same id
                    num_imgs_per_tracklet_rgb.append(len(img_rgb_paths))

        num_tracklets_ir = len(tracklets_ir)
        num_tracklets_rgb = len(tracklets_rgb)
        num_tracklets = num_tracklets_rgb  + num_tracklets_ir
        ir_label = np.array(ir_label)
        rgb_label = np.array(rgb_label)

        return tracklets_ir, num_tracklets_ir,num_imgs_per_tracklet_ir,tracklets_rgb,num_tracklets_rgb,num_imgs_per_tracklet_rgb,num_pids,ir_label,rgb_label


    def _process_data_test(self,names,meta_data,relabel=False,min_seq_len=0):
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,3].tolist()))
        num_pids = len(pid_list)

        # dict {pid : label}
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []
        camids = []
        test_label = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            m,start_index,end_index,pid,camid = data
            if relabel: pid = pid2label[pid]

            img_names = names[start_index-1:end_index]
            img_paths = [osp.join(self.root,decoder_pic_path("Test",img_name)) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                camids.append(camid)
                test_label.append(pid)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)
        test_label = np.array(test_label)
        camids = np.array(camids)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet,camids,test_label

def decoder_pic_path(mode,fname):
    base = fname[0:4]
    modality = fname[5]
    if modality == '1' :
        modality_str = 'ir'
    else:
        modality_str = 'rgb'
    T_pos = fname.find('T')
    D_pos = fname.find('D')
    F_pos = fname.find('F')
    camera = fname[D_pos:T_pos]
    picture = fname[F_pos+1:]
    path = mode + '/' + base + '/' + modality_str + '/' + camera + '/' + picture
    return path

class VideoDataset_test_Inf(data.Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None,test_cam = None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.test_cam = test_cam

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = range(num)
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list = []
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)

                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)

                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]
            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, 0
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class VideoDataset_test_Vis(data.Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=12, sample='evenly', transform=None,test_cam = None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.test_cam = test_cam

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        S = self.seq_len
        sample_clip_ir = []
        frame_indices_ir = list(range(num))
        if num < S:
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (S - num)
            for s in range(S):
                pool_ir = strip_ir[s * 1:(s + 1) * 1]
                sample_clip_ir.append(list(pool_ir))
        else:
            inter_val_ir = math.ceil(num / S)
            strip_ir = list(range(num)) + [frame_indices_ir[-1]] * (inter_val_ir * S - num)
            for s in range(S):
                pool_ir = strip_ir[inter_val_ir * s:inter_val_ir * (s + 1)]
                sample_clip_ir.append(list(pool_ir))

        sample_clip_ir = np.array(sample_clip_ir)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = range(num)
            indices_list = []
            while num - cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index + self.seq_len])
                cur_index += self.seq_len
            last_seq = frame_indices[cur_index:]
            last_seq = list(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list = []
            for indices in indices_list:
                imgs = []
                for index in indices:
                    index = int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)

                    img = np.array(img)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)

                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            num_ir = len(img_paths)
            frame_indices = range(num_ir)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            indices = list(indices)
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices = np.array(indices)
            imgs_ir = []
            for index in indices:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, camid

        if self.sample == 'video_test':
            number = sample_clip_ir[:, 0]
            imgs_ir = []
            for index in number:
                index = int(index)
                img_path = img_paths[index]
                img = read_image(img_path)

                img = np.array(img)
                if self.transform is not None:
                    img = self.transform(img)

                imgs_ir.append(img)
            imgs_ir = torch.cat(imgs_ir, dim=0)
            return imgs_ir, pid, 1
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img