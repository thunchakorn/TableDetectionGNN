import numpy as np
import pandas as pd
import cv2
import os
# from RVL_parser import RVL_Parser

class GraphOCR:
    def __init__(self, label=['supplier', 'invoice_info', 'receiver', 'positions', 'total', 'other']):
        self.label = label
        self.count = 0
    
    def connect(self, df, h, w, gt=True):
        """
        param df: dataframe that have following columns: xmin, ymin, xmax, ymax, text, label (if label = True)
        param h: height of image
        param w: weight of image
        param gt: ground truth if True df must have column label and will also return y and edge_label
        """
        if gt:
            assert 'label' in df.columns, 'There is no label column in DataFrame. There must be label column if gt=True'

        self.df = df
        self.gt = gt
        self.df['xcen'] = (self.df['xmin'] + self.df['xmax']) / 2
        self.df['ycen'] = (self.df['ymin'] + self.df['ymax']) / 2
        self.h = h
        self.w = w
        
        x = self.get_features()
        edge_index, edge_attr = self.get_edge_index()
        if gt:
            y = np.array([self.label.index(l) for l in self.df['label']])
            edge_label = self.get_edge_label(edge_index)
            return x, y, edge_index, edge_attr, edge_label
        else:
            return x, edge_index, edge_attr
         
    def get_edge_label(self, edge_index):
        edge_label = list(map(lambda src,dst: int(self.df['label'][src] == self.df['label'][dst]), edge_index[:,0], edge_index[:,1]))
        return np.array(edge_label, dtype = np.int)
    
    def get_edge_index(self):
        self.df = self.df.sort_values(by = ['ycen', 'xcen'])
        edge_hori, edge_attr_hori = self.get_horizontal_edge()
        edge_verti, edge_attr_verti = self.get_vertical_edge()
        edge_index = np.concatenate((edge_hori, edge_verti))
        edge_attr = np.concatenate((edge_attr_hori, edge_attr_verti)) # 0 for horizontal edge and 1 for vertical edge
        self.df = self.df.sort_index()
        return edge_index, edge_attr

    def get_horizontal_edge(self):

        edge_hori = set()
        edge_attr_hori = []
        for src_id, src in self.df.iterrows():
            src_possible_range = [{'ymin':src['ymin'], 'ymax':src['ymax']}]
            for src_range in src_possible_range:
                for dst_id, dst in self.df.iloc[src_id+1:,:].iterrows():
                    if src_id == dst_id:
                        continue
                    
                    if src['xcen'] <= dst['xcen']: # only case where src is left and dst is right.
                        # case 1; src and dst are same lavel
                        if src_range['ymin'] == dst['ymin'] and src_range['ymax'] == dst['ymax']:
                            edge_hori.add((src_id, dst_id))
                            edge_hori.add((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(src_id, dst_id, mode = 'hori'))
                            break
                        
                        # case 2; src is higher 
                        elif src_range['ymin'] <= dst['ymin'] and src_range['ymax'] <= dst['ymax'] \
                        and src_range['ymax'] >= dst['ymin']:
                            # update src possible range
                            src_range['ymax'] = dst['ymin']
                            edge_hori.add((src_id, dst_id))
                            edge_hori.add((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(src_id, dst_id, mode = 'hori'))
                            
                        # case 3; src is lower
                        elif src_range['ymin'] >= dst['ymin'] and src_range['ymax'] >= dst['ymax'] \
                        and src_range['ymin'] <= dst['ymax']:
                            # update src possible range
                            src_range['ymin'] = dst['ymax']
                            edge_hori.add((src_id, dst_id))
                            edge_hori.add((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(src_id, dst_id, mode = 'hori'))
                            
                        # case 4; src is in middle of dst
                        elif src_range['ymin'] >= dst['ymin'] and src_range['ymax'] <= dst['ymax']:
                            edge_hori.add((src_id, dst_id))
                            edge_hori.add((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(src_id, dst_id, mode = 'hori'))
                            break
                            
                        # case 5; dst is in middle of src
                        elif src_range['ymin'] <= dst['ymin'] and src_range['ymax'] >= dst['ymax']:
                            edge_hori.add((src_id, dst_id))
                            edge_hori.add((dst_id, src_id))
                            src_possible_range.append({'ymin':src_range['ymin'], 'ymax':dst['ymin']})
                            src_possible_range.append({'ymin':dst['ymax'], 'ymax':src_range['ymax']})
                            edge_attr_hori.append(self.get_edge_attr(src_id, dst_id, mode = 'hori'))
                            break

        edge_hori = list(edge_hori)
        edge_attr_hori = np.array(edge_attr_hori, dtype=np.float)
        return edge_hori, edge_attr_hori

    def get_vertical_edge(self):

        edge_verti = set()
        edge_attr_verti = []
        for src_id, src in self.df.iterrows():
            src_possible_range = [{'xmin':src['xmin'], 'xmax':src['xmax']}]
            for src_range in src_possible_range:
                for dst_id, dst in self.df.iloc[src_id+1:,:].iterrows():
                    if src_id == dst_id:
                        continue
                        
                    if src['ycen'] <= dst['ycen']: # only case where src is above and dst is below.
                        # case 1; src and dst are same lavel
                        if src_range['xmin'] == dst['xmin'] and src_range['xmax'] == dst['xmax']:
                            edge_verti.add((src_id, dst_id))
                            edge_attr_verti.append(self.get_edge_attr(src_id, dst_id, mode = 'verti'))
                            break
                        
                        # case 2; src is left of dst 
                        elif src_range['xmin'] <= dst['xmin'] and src_range['xmax'] <= dst['xmax'] \
                        and src_range['xmax'] >= dst['xmin']:
                            # update src possible range
                            src_range['xmax'] = dst['xmin']
                            edge_verti.add((src_id, dst_id))
                            edge_attr_verti.append(self.get_edge_attr(src_id, dst_id, mode = 'verti'))

                        # case 3; src is right of dst
                        elif src_range['xmin'] >= dst['xmin'] and src_range['xmax'] >= dst['xmax'] \
                        and src_range['xmin'] <= dst['xmax']:
                            # update src possible range
                            src_range['xmin'] = dst['xmax']
                            edge_verti.add((src_id, dst_id))
                            edge_attr_verti.append(self.get_edge_attr(src_id, dst_id, mode = 'verti'))
                            
                        # case 4; src is in middle of dst
                        elif src_range['xmin'] >= dst['xmin'] and src_range['xmax'] <= dst['xmax']:
                            edge_verti.add((src_id, dst_id))
                            edge_attr_verti.append(self.get_edge_attr(src_id, dst_id, mode = 'verti'))
                            break
                            
                        # case 5; dst is in middle of src
                        elif src_range['xmin'] <= dst['xmin'] and src_range['xmax'] >= dst['xmax']:
                            edge_verti.add((src_id, dst_id))
                            src_possible_range.append({'xmin':src_range['xmin'], 'xmax':dst['xmin']})
                            src_possible_range.append({'xmin':dst['xmax'], 'xmax':src_range['xmax']})
                            edge_attr_verti.append(self.get_edge_attr(src_id, dst_id, mode = 'verti'))
                            break
                            
        edge_verti = list(edge_verti)
        edge_attr_verti = np.array(edge_attr_verti, dtype=np.float)
        return edge_verti, edge_attr_verti

    def get_features(self):
        x = np.array([], dtype = np.float32).reshape((0,7)) # num_feature is 7
        pos_row_iter = self.df.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']].iterrows()
        text_row_iter = iter(self.df['text'])
        for pos, text in zip(pos_row_iter , text_row_iter):
            norm_pos = self.get_normalize_pos(pos[1]) #pos[1] => get only position not index (pos[0])
            text_features = self.get_text_features(text)
            x = np.append(x, np.concatenate((norm_pos, text_features)).reshape(1, -1), axis = 0)
        return x

    def get_normalize_pos(self, pos):
        """
        param pos: position as xmin, ymin, xmax, ymax
        """
        return np.array((pos['xmin']/self.w, pos['ymin']/self.h, pos['xmax']/self.w, pos['ymax']/self.h))
        

    def get_text_features(self, text):
        n_alpha = 0
        n_numeric = 0
        n_special = 0
        text = text.replace(' ', '')
        for char in text:
            if char.isalpha():n_alpha  += 1
            elif char.isnumeric():n_numeric += 1
            else:n_special += 1
        
        text_features = [n_alpha, n_numeric, n_special]
        return np.array(text_features)

    def get_edge_attr(self, src_id, dst_id, mode = 'hori'):
        """Get edge attr with dimension = A x D + T = 33
        where
        A is angle of all four points from src to dst = 16 dims
        D is inverse distance of all four points from src to dst = 16 dims
        T is type of connection include {horizontal=0, vertical=1} = 2 dims

        """
        def get_angle(a, b):
            """Get angle of 2 points in image
            param a: point a with shape (n,2)
            param b: point a with shape (n,2)
            param mode: 2 mode to choose {'hori', 'verti'}
            return: angle in radian with shape (n, )
            """
            if mode == 'hori':
                numerator = (b[:,1] - a[:,1])
                denominator = (b[:,0] - a[:,0])
            elif mode == 'verti':
                numerator = (b[:,0] - a[:,0])
                denominator = (b[:,1] - a[:,1])
            return np.arctan2(numerator, denominator)/np.pi*180

        def get_inv_dist(a, b, h, w):
            """Get inverse distance of 2 points in image
            param a: point a with shape (n,2)
            param b: point a with shape (n,2)
            param h: height of image
            param w: width of image
            return: inverse distance range (0,1) with shape (n, )
            """
            max_dist = np.hypot(h,w)
            dist = np.linalg.norm(a-b, axis = 1)
            inv_dist = (max_dist - dist)/max_dist
            return inv_dist

        def get_4_points(row):
            """
            return array of points in following order
            left_top, right_top, left_bottom, right_bottom
            """
            return np.array([[row['xmin'], row['ymin']],
                             [row['xmax'], row['ymin']],
                             [row['xmin'], row['ymax']],
                             [row['xmax'], row['ymax']]])

        src_points = get_4_points(self.df.loc[src_id,:])
        dst_points = get_4_points(self.df.loc[dst_id,:])
        src_pair = np.repeat(src_points,4,axis=0)
        dst_pair = np.tile(dst_points, (4,1))

        dist = get_inv_dist(src_pair, dst_pair, self.h, self.w)
        angle = get_angle(src_pair, dst_pair)
        edge_attr = np.concatenate((dist, angle, [0])) if mode == 'hori' else np.concatenate((dist, angle, [1]))
        return edge_attr

# if __name__ == '__main__':
#     p = RVL_Parser('RVL-Dataset/')
#     a = p.parse()
#     df = next(a)
#     df.columns = ['xmin', 'ymin', 'xmax', 'ymax', 'xmin_norm', 'ymin_norm', 'xmax_norm',
#                     'ymax_norm', 'text', 'label']
#     gocr = GraphOCR()
#     x, y, edge_index, edge_attr, edge_label = gocr.connect(df, 1000, 762)