import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
import shutil
import os
import os.path as osp
import glob
import json

import time
import cv2
import numpy as np
import pandas as pd
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from grapher_coco import GraphOCR

class GraphCOCODataset(InMemoryDataset):
    '''
    dataset-root
    |____ annotations
    |        |___dataset1.json
    |        |___dataset2.json
    |____ images
    |        |___dataset1/
    |        |___dataset2/
    |_____OCR
    |       |___dataset1/
    |       |___dataset2/

    '''
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        self.TARGET = ['DocType', 'Item', 'Payment', 'Reciever', 'Remark', 'Sender', 'Signature', 'Summary', 'Table', 'Other']
        self.grapher = GraphOCR(label = self.TARGET)
        self.plot_dir = osp.join(self.root, 'plot_graph/')
        if not osp.isdir(osp.join(self.root, 'processed')):
            os.mkdir(osp.join(self.root, 'processed'))
        super(GraphCOCODataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(osp.join(self.root, 'processed', 'dataset.pt'))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if osp.exists(osp.join(self.root, 'processed', 'dataset.pt')):
            return ['dataset.pt']
        else:
             return []
    
    def download(self):
        pass       

    def process(self):
        if not osp.isdir(osp.join(self.root, 'plot_graph/')):
            os.mkdir(osp.join(self.root, 'plot_graph/'))


        dataset_list = os.listdir(osp.join(self.root, 'images'))
        print(dataset_list)
        data_list = []
        for dataset_name in dataset_list:
            if dataset_name[0] == '.':  #ignore hidden file
                continue
            print(f'++++++++++++++++={dataset_name}+++++++++++++++')
            ann_path = osp.join(self.root, 'annotations', dataset_name + '.json')

            with open(ann_path, 'r') as f:
                ann = json.load(f)

            num_img = len(ann['images'])
            ann_id_img = self.get_ann_id(ann['annotations'], num_img)

            ## image level
            for id, img_info in enumerate(ann['images']):
                start = time.time()
                h = img_info['height']
                w = img_info['width']
                img_id = img_info['id']
                print(osp.join(self.root, 'images', img_info['file_name']))
                img = cv2.imread(osp.join(self.root, 'images', img_info['file_name']))
                gt_dict = self.get_gt_dict(img_id, np.array(ann['annotations']), ann_id_img)

                ## ocr level
                ocr_path = osp.join(self.root, 'OCR', dataset_name, osp.basename(img_info['file_name'].split('.')[0] + '.json')) 

                with open(ocr_path, 'r') as f:
                    ocr = json.load(f)

                row = []
                for ocr_result in ocr['result']:
                    text, bbox = self.get_text(ocr_result)
                    label = self.get_label(bbox, gt_dict)
                    row.append((bbox[0], bbox[1], bbox[2], bbox[3], text, label))
                df = pd.DataFrame(row, columns=['xmin', 'ymin', 'xmax', 'ymax', 'text', 'label'])
                self.visualise_node(img, df)
                print('connecting graph')
                x, y, edge_index, edge_attr, edge_label = self.grapher.connect(df, h, w, gt_dict)

                save_plot_name = osp.join(self.plot_dir, osp.split(img_info['file_name'])[-1])
                self.visualise_graph(img, df, edge_index, edge_attr, edge_label, save_plot_name)

                data = Data(x=torch.tensor(x, dtype=torch.float),
                        y=torch.tensor(y, dtype=torch.long),
                        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                        edge_label = torch.tensor(edge_label, dtype=torch.long),
                        edge_attr = torch.tensor(edge_attr, dtype=torch.float))
                print(f'time use {time.time() - start} seconds')

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data) 
                print(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), osp.join(self.root, 'processed', 'dataset.pt'))

    def get_gt_dict(self, img_id, annotations, ann_id_img):
        """
        get ground truth dict of their region as dictionary
        """
        pos = ann_id_img[img_id]
        ann = annotations[pos]
        gt_dict = {t:np.array([]) for t in self.TARGET}
        for region in ann:
            region_target = self.TARGET[region['category_id'] - 1] #category_id start at 1
            seg = region['segmentation'][0]
            polygon = Polygon([(i,j) for i, j in zip(seg[::2], seg[1::2])])
            gt_dict[region_target] = np.append(gt_dict[region_target], polygon)
        return gt_dict

    def get_ann_id(self, annotations, num_img):
        """
        get annotations id of each images as dictionary
        """
        img_id_list = np.array([an['image_id'] for an in annotations])
        ann_id_img = {i:np.where(img_id_list == i)[0] for i in range(num_img)}
        return ann_id_img

    def get_text(self, ocr_result):
        def find_xy_bbox(quad):
            """
            return xmin, ymin, xmax, ymax
            """
            points = np.array([[i,j] for i, j in zip(quad[::2], quad[1::2])])
            x = points[:,0]
            y = points[:,1]
            xmin = int(min(x))
            ymin = int(min(y))
            xmax = int(max(x))
            ymax = int(max(y))
            return xmin, ymin, xmax, ymax
        text = ocr_result['text']
        xmin, ymin, xmax, ymax = find_xy_bbox(ocr_result['quad'])

        return text, (xmin, ymin, xmax, ymax)      

    def get_label(self, bbox, gt_dict):
        x0 = np.mean([bbox[0], bbox[2]])
        y0 = np.mean([bbox[1], bbox[3]])
        cen = Point(x0, y0)
        label = None
        gt_check = {target:[v.contains(cen) for v in poly] for target,poly in gt_dict.items()} # check if centroid of text box in which ground truth box.
        gt_filter = {target:[c for c in check] for target ,check in gt_check.items() if np.any([c for c in check])}  # check how many gt box its belong to.
        if len(gt_filter) == 1:
            label = list(gt_filter.keys())[0]
        elif len(gt_filter) == 0:
            label = 'Other'
        elif len(gt_filter) > 1:
            if 'Table' in gt_filter.keys() and 'Item' in gt_filter.keys():
                label = 'Item'
            else: # case overlap ground truth box
                label_temp = None
                min_temp = None
                for t, c in gt_filter.items():
                    label_dist = np.min([poly.centroid.distance(cen) for poly in gt_dict[t][c]])
                    if min_temp is None or label_dist < min_temp:
                        min_temp = label_dist
                        label_temp = t
                label = label_temp
        return label


    def visualise_graph(self, img, df, edge_index, edge_attr, edge_label, save_name):
            for (src, dst), attr, lab in zip(edge_index, edge_attr, edge_label):
                # connected in same region
                if lab == 1:
                    cv2.line(img, 
                            (int(df.loc[src, 'xcen']), int(df.loc[src, 'ycen'])), 
                            (int(df.loc[dst, 'xcen']), int(df.loc[dst, 'ycen'])), 
                            (0,255,0), 1)
                else:
                    cv2.line(img, 
                            (int(df.loc[src, 'xcen']), int(df.loc[src, 'ycen'])), 
                            (int(df.loc[dst, 'xcen']), int(df.loc[dst, 'ycen'])), 
                            (0,0,255), 1)  

                cv2.rectangle(img, (df.loc[src, 'xmin'], df.loc[src, 'ymin']), (df.loc[src, 'xmax'], df.loc[src, 'ymax']), (0,255,255),1)

                if df.loc[src, 'label'] == 'positions':
                    cv2.circle(img, (int(df.loc[src, 'xcen']), int(df.loc[src, 'ycen'])), 2, (255,0,0), 2)
                else:
                    cv2.circle(img, (int(df.loc[src, 'xcen']), int(df.loc[src, 'ycen'])), 2, (0,255,255), 2)
                if df.loc[dst, 'label'] == 'positions':
                    cv2.circle(img, (int(df.loc[dst, 'xcen']), int(df.loc[dst, 'ycen'])), 2, (255,0,0), 2)
                else:
                    cv2.circle(img, (int(df.loc[dst, 'xcen']), int(df.loc[dst, 'ycen'])), 2, (0,255,255), 2)
            cv2.imwrite(save_name, img)

    def visualise_node(self, img, df):
        for idx, row in df.iterrows():
            cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (255,0,0), 1)
            cv2.putText(img, row['label'], (row['xmin'], row['ymin']), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    dataset = GraphCOCODataset('findoc-dataset')



