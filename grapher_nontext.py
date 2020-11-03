import os.path as osp
import numpy as np
import pandas as pd

from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

import torch
from train_utils import try_gpu

class GraphOCR:
    def __init__(self, label, edge_limit = (5,5)):
        self.label = label
        self.count = 0
        self.edge_limit = edge_limit # edge limit in vertical and horizontal respectively (tuples)
    
    def connect(self, df, h, w, gt_dict, gt=True):
        """
        param df: dataframe that have following columns: xmin, ymin, xmax, ymax, text, label (if label = True)
        param h: height of image
        param w: weight of image
        param gt: ground truth if True df must have column label and will also return y and edge_label
        """
        if gt:
            assert 'label' in df.columns, 'There is no label column in DataFrame. There must be label column if gt=True'

        self.df = df
        self.gt_region = [i for v in gt_dict.values() for i in v]
        self.df['xcen'] = (self.df['xmin'] + self.df['xmax']) / 2
        self.df['ycen'] = (self.df['ymin'] + self.df['ymax']) / 2
        self.h = h
        self.w = w
        
        x = self.get_feature()
        # text_feature = self.get_text_features_bert()
        edge_index, edge_attr = self.get_edge_index()
        self.df.sort_index(inplace = True)
        if gt:
            y = np.array([self.label.index(l) for l in self.df['label']])
            edge_label = self.get_edge_label(edge_index)
            return x, y, edge_index, edge_attr, edge_label
        else:
            return x, edge_index, edge_attr

    def get_edge_label(self, edge_index):
        def isinregion(x0, y0, region):
            if region[0] < x0 < region[2] and region[1] < y0 < region[3]:
                return 1
            else:
                return 0

        def inregion_index(src, dst):
            x0_src = self.df.loc[src, 'xcen']
            y0_src = self.df.loc[src, 'ycen']
            x0_dst = self.df.loc[dst, 'xcen']
            y0_dst = self.df.loc[dst, 'ycen']
            
            src_region_id = [i.contains(Point(x0_src, y0_src)) for i in self.gt_region]
            dst_region_id = [i.contains(Point(x0_dst, y0_dst)) for i in self.gt_region]
            if np.any(src_region_id) and np.any(dst_region_id):
                return np.argmax(src_region_id) == np.argmax(dst_region_id)
            else:
                return False
            
        edge_label = np.array([inregion_index(src, dst) for src, dst in zip(edge_index[:,0], edge_index[:,1])])
        return edge_label       
    
    def get_edge_index(self):
        self.df.reset_index(inplace = True)

        self.df = self.df.sort_values(by = ['ycen', 'xcen'])
        self.df.reset_index(drop=True,inplace=True)
        edge_verti, edge_attr_verti = self.get_vertical_edge()

        self.df = self.df.sort_values(by = ['xcen', 'ycen'])
        self.df.reset_index(drop=True,inplace=True)
        edge_hori, edge_attr_hori = self.get_horizontal_edge()

        # reordering, key is src; ascending
        edge_index = np.concatenate((edge_hori, edge_verti))
        sort_by = np.argsort(edge_index[:,0])
        edge_index = edge_index[sort_by]

        edge_attr = np.concatenate((edge_attr_hori, edge_attr_verti))[sort_by] # 0 for horizontal edge and 1 for vertical edge
        self.df.set_index('index', inplace = True)
        return edge_index, edge_attr

    def get_vertical_edge(self):
        edge_verti = []
        edge_attr_verti = []
        for src_row, src in self.df.iterrows():
            src_id = src['index']
            src_possible_range = [{'xmin':src['xmin'], 'xmax':src['xmax']}]
            edge_count = 0
            for src_range in src_possible_range:
                for dst_row, dst in self.df.iloc[src_row+1:,:].iterrows():
                    if edge_count > self.edge_limit[0]:
                        break
                    dst_id = dst['index']
                    if src_id == dst_id:
                        continue

                    if (src_id, dst_id) in edge_verti or (dst_id, src_id) in edge_verti:
                        continue

                    if src['ycen'] <= dst['ycen']: # only case where src is above and dst is below.
                        # case 1; src and dst are same lavel
                        if src_range['xmin'] == dst['xmin'] and src_range['xmax'] == dst['xmax']:
                            edge_verti.append((src_id, dst_id))
                            edge_verti.append((dst_id, src_id))
                            edge_attr_verti.append(self.get_edge_attr(mode = 'verti'))
                            edge_attr_verti.append(self.get_edge_attr(mode = 'verti'))
                            edge_count += 1
                            break
                        
                        # case 2; src is left of dst 
                        elif src_range['xmin'] <= dst['xmin'] and src_range['xmax'] <= dst['xmax'] \
                        and src_range['xmax'] >= dst['xmin']:
                            edge_verti.append((src_id, dst_id))
                            edge_verti.append((dst_id, src_id))
                            edge_attr_verti.append(self.get_edge_attr(mode = 'verti'))
                            edge_attr_verti.append(self.get_edge_attr(mode = 'verti'))
                            src_range['xmax'] = dst['xmin']
                            edge_count += 1

                        # case 3; src is right of dst
                        elif src_range['xmin'] >= dst['xmin'] and src_range['xmax'] >= dst['xmax'] \
                        and src_range['xmin'] <= dst['xmax']:
                            edge_verti.append((src_id, dst_id))
                            edge_verti.append((dst_id, src_id))
                            edge_attr_verti.append(self.get_edge_attr(mode = 'verti'))
                            edge_attr_verti.append(self.get_edge_attr(mode = 'verti'))
                            src_range['xmin'] = dst['xmax']
                            edge_count += 1
                            
                        # case 4; src is in middle of dst
                        elif src_range['xmin'] >= dst['xmin'] and src_range['xmax'] <= dst['xmax']:
                            edge_verti.append((src_id, dst_id))
                            edge_verti.append((dst_id, src_id))
                            edge_attr_verti.append(self.get_edge_attr(mode = 'verti'))
                            edge_attr_verti.append(self.get_edge_attr(mode = 'verti'))
                            edge_count += 1
                            break
                            
                        # case 5; dst is in middle of src
                        elif src_range['xmin'] <= dst['xmin'] and src_range['xmax'] >= dst['xmax']:
                            edge_verti.append((src_id, dst_id))
                            edge_verti.append((dst_id, src_id))
                            edge_attr_verti.append(self.get_edge_attr(mode = 'verti'))
                            edge_attr_verti.append(self.get_edge_attr(mode = 'verti'))
                            src_possible_range.append({'xmin':src_range['xmin'], 'xmax':dst['xmin']})
                            src_possible_range.append({'xmin':dst['xmax'], 'xmax':src_range['xmax']})
                            edge_count += 1
                            break

        edge_verti = np.array(edge_verti)
        edge_attr_verti = np.array(edge_attr_verti, dtype=np.float)
        return edge_verti, edge_attr_verti

    def get_horizontal_edge(self):
        edge_hori = []
        edge_attr_hori = []
        for src_row , src in self.df.iterrows():
            src_id = src['index']
            src_possible_range = [{'ymin':src['ymin'], 'ymax':src['ymax']}]
            edge_count = 0
            for src_range in src_possible_range:
                for dst_row, dst in self.df.iloc[src_row+1:].iterrows():
                    if edge_count > self.edge_limit[1]:
                        break
                    dst_id = dst['index']
                    if src_id == dst_id:
                        continue
                    
                    if (src_id, dst_id) in edge_hori or (dst_id, src_id) in edge_hori:
                        continue

                    if src['xcen'] <= dst['xcen']: # only case where src is left and dst is right.
                        # case 1; src and dst are same lavel
                        if src_range['ymin'] == dst['ymin'] and src_range['ymax'] == dst['ymax']:
                            edge_hori.append((src_id, dst_id))
                            edge_hori.append((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(mode = 'hori'))
                            edge_attr_hori.append(self.get_edge_attr(mode = 'hori'))
                            edge_count += 1
                            break
                        
                        # case 2; src is higher 
                        elif src_range['ymin'] <= dst['ymin'] and src_range['ymax'] <= dst['ymax'] \
                        and src_range['ymax'] >= dst['ymin']:
                            edge_hori.append((src_id, dst_id))
                            edge_hori.append((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(mode = 'hori'))
                            edge_attr_hori.append(self.get_edge_attr(mode = 'hori'))
                            src_range['ymax'] = dst['ymin']
                            edge_count += 1
                            
                        # case 3; src is lower
                        elif src_range['ymin'] >= dst['ymin'] and src_range['ymax'] >= dst['ymax'] \
                        and src_range['ymin'] <= dst['ymax']:
                            edge_hori.append((src_id, dst_id))
                            edge_hori.append((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(mode = 'hori'))
                            edge_attr_hori.append(self.get_edge_attr(mode = 'hori'))
                            src_range['ymin'] = dst['ymax']
                            edge_count += 1
                            
                        # case 4; src is in middle of dst
                        elif src_range['ymin'] >= dst['ymin'] and src_range['ymax'] <= dst['ymax']:
                            edge_hori.append((src_id, dst_id))
                            edge_hori.append((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(mode = 'hori'))
                            edge_attr_hori.append(self.get_edge_attr(mode = 'hori'))
                            edge_count += 1
                            break
                            
                        # case 5; dst is in middle of src
                        elif src_range['ymin'] <= dst['ymin'] and src_range['ymax'] >= dst['ymax']:
                            edge_hori.append((src_id, dst_id))
                            edge_hori.append((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(mode = 'hori'))
                            edge_attr_hori.append(self.get_edge_attr(mode = 'hori'))
                            src_possible_range.append({'ymin':src_range['ymin'], 'ymax':dst['ymin']})
                            src_possible_range.append({'ymin':dst['ymax'], 'ymax':src_range['ymax']})
                            edge_count += 1
                            break
        edge_hori = np.array(edge_hori)
        edge_attr_hori = np.array(edge_attr_hori, dtype=np.float)
        return edge_hori, edge_attr_hori

    def get_feature(self):
        self.df['xmin_norm'] = self.df['xmin']/self.w
        self.df['ymin_norm'] = self.df['ymin']/self.h
        self.df['xmax_norm'] = self.df['xmax']/self.w
        self.df['ymax_norm'] = self.df['ymax']/self.h
        feature = np.array(self.df.loc[:,['xmin_norm', 'ymin_norm', 'xmax_norm', 'ymax_norm', 'xmin', 'ymin', 'xmax', 'ymax']])
        return feature

    def get_edge_attr(self, mode = 'hori'):
        edge_attr = torch.tensor([0]) if mode == 'hori' else torch.tensor([1])
        return edge_attr

    # def get_edge_attr(self, src_id, dst_id, mode = 'hori'):
    #     """Get edge attr with dimension = A x D + T = 33
    #     where
    #     A is angle of all four points from src to dst = 16 dims
    #     D is inverse distance of all four points from src to dst = 16 dims
    #     T is type of connection include {horizontal=0, vertical=1} = 2 dims

    #     """
    #     def get_angle(a, b):
    #         """Get angle of 2 points in image
    #         param a: point a with shape (n,2)
    #         param b: point a with shape (n,2)
    #         param mode: 2 mode to choose {'hori', 'verti'}
    #         return: angle in radian with shape (n, )
    #         """
            
    #         if mode == 'hori':
    #             numerator = (b[:,1] - a[:,1])
    #             denominator = (b[:,0] - a[:,0])
    #         else:
    #             numerator = (b[:,0] - a[:,0])
    #             denominator = (b[:,1] - a[:,1])
    #         return np.arctan2(numerator, denominator)

    #     def get_inv_dist(a, b, h, w):
    #         """Get inverse distance of 2 points in image
    #         param a: point a with shape (n,2)
    #         param b: point a with shape (n,2)
    #         param h: height of image
    #         param w: width of image
    #         return: inverse distance range (0,1) with shape (n, )
    #         """
    #         max_dist = np.hypot(h,w)
    #         dist = np.linalg.norm(a-b, axis = 1)
    #         inv_dist = (max_dist - dist)/max_dist
    #         return inv_dist

    #     def get_4_points(row):
    #         """
    #         return array of points in following order
    #         left_top, right_top, left_bottom, right_bottom
    #         """
    #         return np.array([[row['xmin'], row['ymin']],
    #                          [row['xmax'], row['ymin']],
    #                          [row['xmin'], row['ymax']],
    #                          [row['xmax'], row['ymax']]])

    #     src_points = get_4_points(self.df.loc[src_id,:])
    #     dst_points = get_4_points(self.df.loc[dst_id,:])
    #     src_pair = np.repeat(src_points,4,axis=0)
    #     dst_pair = np.tile(dst_points, (4,1))
    #     dist = get_inv_dist(src_pair, dst_pair, self.h, self.w)
    #     angle = get_angle(src_pair, dst_pair)
    #     edge_attr = np.concatenate((dist, angle, [0])) if mode == 'hori' else np.concatenate((dist, angle, [1]))
    #     return edge_attr
    