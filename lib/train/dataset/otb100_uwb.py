import os
import os.path
import torch
import numpy as np
import pandas
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class OTB100UWB(BaseVideoDataset):
    """ OTB100_UWB dataset (视觉部分).
    
    这是 OTB100 的增强版本，包含 UWB 传感器数据。
    本接口仅使用视觉部分（groundtruth.txt），忽略 UWB 数据。
    
    数据集结构:
    data/OTB100_UWB/
    ├── train/
    │   ├── Biker/
    │   │   ├── 00000001.jpg
    │   │   ├── groundtruth.txt
    │   │   └── list.txt
    │   ├── ...
    ├── val/
    └── test/
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train', data_fraction=None):
        """
        args:
            root - path to the OTB100_UWB dataset.
            image_loader (jpeg4py_loader) - The function to read the images.
            split - 'train', 'val', or 'test'
            data_fraction - Fraction of dataset to be used. None means use all.
        """
        root = env_settings().otb100_uwb_dir if root is None else root
        super().__init__('OTB100UWB', root, image_loader)

        self.split = split
        
        # 根据 split 设置路径
        if split == 'train':
            self.split_path = os.path.join(root, 'train')
        elif split == 'val':
            self.split_path = os.path.join(root, 'val')
        elif split == 'test':
            self.split_path = os.path.join(root, 'test')
        else:
            raise ValueError(f'Unknown split: {split}')

        # 读取序列列表
        self.sequence_list = self._build_sequence_list()

        if data_fraction is not None:
            import random
            self.sequence_list = random.sample(self.sequence_list, 
                                              int(len(self.sequence_list) * data_fraction))

        self.seq_per_class = self._build_class_list()

    def _build_sequence_list(self):
        """从 list.txt 读取序列列表"""
        list_file = os.path.join(self.split_path, 'list.txt')
        if os.path.exists(list_file):
            sequence_list = pandas.read_csv(list_file, header=None).squeeze("columns").values.tolist()
        else:
            # 如果 list.txt 不存在，直接列出目录
            sequence_list = [f for f in os.listdir(self.split_path) 
                           if os.path.isdir(os.path.join(self.split_path, f))]
        
        return sequence_list

    def _build_class_list(self):
        """构建类别列表（OTB中每个序列视为一个类别）"""
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            seq_per_class[seq_name] = [seq_id]
        return seq_per_class

    def get_name(self):
        return 'otb100_uwb'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.seq_per_class)

    def _read_bb_anno(self, seq_path):
        """读取 groundtruth.txt 标注"""
        bb_anno_file = os.path.join(seq_path, 'groundtruth.txt')
        
        if not os.path.exists(bb_anno_file):
            raise FileNotFoundError(f'Annotation file not found: {bb_anno_file}')
        
        # OTB 格式可能是逗号、空格或Tab分隔
        try:
            gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32,
                               na_filter=False, low_memory=False).values
        except:
            try:
                gt = pandas.read_csv(bb_anno_file, delimiter='\t', header=None, dtype=np.float32,
                                   na_filter=False, low_memory=False).values
            except:
                gt = pandas.read_csv(bb_anno_file, delim_whitespace=True, header=None, dtype=np.float32,
                                   na_filter=False, low_memory=False).values
        
        # 处理可能的缺失值
        gt = np.array(gt, dtype=np.float32)
        
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        """读取遮挡标注（如果有）"""
        occlusion_file = os.path.join(seq_path, 'occlusion.txt')
        if os.path.exists(occlusion_file):
            try:
                occlusion = pandas.read_csv(occlusion_file, header=None, dtype=np.int32).values
                return torch.tensor(occlusion.flatten(), dtype=torch.int32)
            except:
                pass
        
        # 如果没有 occlusion.txt，返回全1（全部可见）
        bb_anno = self._read_bb_anno(seq_path)
        return torch.ones(bb_anno.shape[0], dtype=torch.int32)

    def _get_image_path(self, seq_path, frame_id):
        """获取图像路径"""
        # OTB100_UWB 使用 8 位数字编号，从 1 开始
        img_filename = f'{frame_id + 1:08d}.jpg'
        return os.path.join(seq_path, img_filename)

    def get_sequence_info(self, seq_id):
        seq_path = os.path.join(self.split_path, self.sequence_list[seq_id])
        
        bbox = self._read_bb_anno(seq_path)
        
        # OTB 格式: x, y, w, h (左上角坐标 + 宽高)
        if bbox.shape[1] == 4:
            # 已经是 x, y, w, h 格式
            pass
        elif bbox.shape[1] == 2:
            raise ValueError('Invalid bbox format: expected 4 columns (x, y, w, h)')
        else:
            raise ValueError(f'Invalid bbox shape: {bbox.shape}')
        
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_path, frame_id):
        """获取单帧图像"""
        img_path = self._get_image_path(seq_path, frame_id)
        img = self.image_loader(img_path)
        return img

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = os.path.join(self.split_path, self.sequence_list[seq_id])
        
        obj_class = self.sequence_list[seq_id]
        
        # 获取图像帧
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        
        # 构建输出
        anno_frames = {}
        for key, value in anno.items():
            # 处理一维和二维张量
            if value.dim() == 1:
                anno_frames[key] = [value[f_id].clone() for f_id in frame_ids]
            else:
                anno_frames[key] = [value[f_id, :].clone() for f_id in frame_ids]
        
        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
