from transformers import pipeline
import os
import os.path as osp
import numpy as np
import pandas as pd

class GraphOCR:
    def __init__(self, label=[ 'supplier', 'invoice_info', 'receiver', 'positions', 'total', 'other']):
        self.label = label
        self.count = 0
        model_name = 'bert-base-uncased'
        # self.tokenizer = AutoTokenizer.from_pretrained(self.LM_model_name)
        # self.embedding_model = BertModel.from_pretrained(self.LM_model_name, output_hidden_states = False)
        # self.embedding_model.eval()
        self.extractor = pipeline('feature-extraction', model_name)
    
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
        
        x = self.get_features()
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

            src_region_id = np.argmax(list(map(lambda x: isinregion(x0_src, y0_src, x), self.gt_region)))
            dst_region_id = np.argmax(list(map(lambda x: isinregion(x0_dst, y0_dst, x), self.gt_region)))
            return src_region_id == dst_region_id
            
        edge_label = np.array([inregion_index(src, dst) for src, dst in zip(edge_index[:,0], edge_index[:,1])])
        return edge_label
         
    # def get_edge_label(self, edge_index):
    #     edge_label = list(map(lambda src,dst: self.df.loc[src, 'label'] == self.df.loc[dst, 'label'], edge_index[:,0], edge_index[:,1]))
    #     return np.array(edge_label, dtype = np.int)
    
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

    def get_horizontal_edge(self):

        edge_hori = []
        edge_attr_hori = []
        for src_row , src in self.df.iterrows():
            src_id = src['index']
            src_possible_range = [{'ymin':src['ymin'], 'ymax':src['ymax']}]
            for src_range in src_possible_range:
                for dst_row, dst in self.df.iloc[src_row+1:].iterrows():
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
                            edge_attr_hori.append(self.get_edge_attr(src_row, dst_row, mode = 'hori'))
                            edge_attr_hori.append(self.get_edge_attr(dst_row, src_row, mode = 'hori'))
                            break
                        
                        # case 2; src is higher 
                        elif src_range['ymin'] <= dst['ymin'] and src_range['ymax'] <= dst['ymax'] \
                        and src_range['ymax'] >= dst['ymin']:
                            edge_hori.append((src_id, dst_id))
                            edge_hori.append((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(src_row, dst_row, mode = 'hori'))
                            edge_attr_hori.append(self.get_edge_attr(dst_row, src_row, mode = 'hori'))
                            src_range['ymax'] = dst['ymin']
                            
                        # case 3; src is lower
                        elif src_range['ymin'] >= dst['ymin'] and src_range['ymax'] >= dst['ymax'] \
                        and src_range['ymin'] <= dst['ymax']:
                            edge_hori.append((src_id, dst_id))
                            edge_hori.append((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(src_row, dst_row, mode = 'hori'))
                            edge_attr_hori.append(self.get_edge_attr(dst_row, src_row, mode = 'hori'))
                            src_range['ymin'] = dst['ymax']
                            
                        # case 4; src is in middle of dst
                        elif src_range['ymin'] >= dst['ymin'] and src_range['ymax'] <= dst['ymax']:
                            edge_hori.append((src_id, dst_id))
                            edge_hori.append((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(src_row, dst_row, mode = 'hori'))
                            edge_attr_hori.append(self.get_edge_attr(dst_row, src_row, mode = 'hori'))
                            break
                            
                        # case 5; dst is in middle of src
                        elif src_range['ymin'] <= dst['ymin'] and src_range['ymax'] >= dst['ymax']:
                            edge_hori.append((src_id, dst_id))
                            edge_hori.append((dst_id, src_id))
                            edge_attr_hori.append(self.get_edge_attr(src_row, dst_row, mode = 'hori'))
                            edge_attr_hori.append(self.get_edge_attr(dst_row, src_row, mode = 'hori'))
                            src_possible_range.append({'ymin':src_range['ymin'], 'ymax':dst['ymin']})
                            src_possible_range.append({'ymin':dst['ymax'], 'ymax':src_range['ymax']})
                            break
        edge_hori = np.array(edge_hori)
        edge_attr_hori = np.array(edge_attr_hori, dtype=np.float)
        return edge_hori, edge_attr_hori

    def get_vertical_edge(self):

        edge_verti = []
        edge_attr_verti = []
        for src_row, src in self.df.iterrows():
            src_id = src['index']
            src_possible_range = [{'xmin':src['xmin'], 'xmax':src['xmax']}]
            for src_range in src_possible_range:
                for dst_row, dst in self.df.iloc[src_row+1:,:].iterrows():
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
                            edge_attr_verti.append(self.get_edge_attr(src_row, dst_row, mode = 'verti'))
                            edge_attr_verti.append(self.get_edge_attr(dst_row, src_row, mode = 'verti'))
                            break
                        
                        # case 2; src is left of dst 
                        elif src_range['xmin'] <= dst['xmin'] and src_range['xmax'] <= dst['xmax'] \
                        and src_range['xmax'] >= dst['xmin']:
                            edge_verti.append((src_id, dst_id))
                            edge_verti.append((dst_id, src_id))
                            edge_attr_verti.append(self.get_edge_attr(src_row, dst_row, mode = 'verti'))
                            edge_attr_verti.append(self.get_edge_attr(dst_row, src_row, mode = 'verti'))
                            src_range['xmax'] = dst['xmin']

                        # case 3; src is right of dst
                        elif src_range['xmin'] >= dst['xmin'] and src_range['xmax'] >= dst['xmax'] \
                        and src_range['xmin'] <= dst['xmax']:
                            edge_verti.append((src_id, dst_id))
                            edge_verti.append((dst_id, src_id))
                            edge_attr_verti.append(self.get_edge_attr(src_row, dst_row, mode = 'verti'))
                            edge_attr_verti.append(self.get_edge_attr(dst_row, src_row, mode = 'verti'))
                            src_range['xmin'] = dst['xmax']
                            
                        # case 4; src is in middle of dst
                        elif src_range['xmin'] >= dst['xmin'] and src_range['xmax'] <= dst['xmax']:
                            edge_verti.append((src_id, dst_id))
                            edge_verti.append((dst_id, src_id))
                            edge_attr_verti.append(self.get_edge_attr(src_row, dst_row, mode = 'verti'))
                            edge_attr_verti.append(self.get_edge_attr(dst_row, src_row, mode = 'verti'))
                            break
                            
                        # case 5; dst is in middle of src
                        elif src_range['xmin'] <= dst['xmin'] and src_range['xmax'] >= dst['xmax']:
                            edge_verti.append((src_id, dst_id))
                            edge_verti.append((dst_id, src_id))
                            edge_attr_verti.append(self.get_edge_attr(src_row, dst_row, mode = 'verti'))
                            edge_attr_verti.append(self.get_edge_attr(dst_row, src_row, mode = 'verti'))
                            src_possible_range.append({'xmin':src_range['xmin'], 'xmax':dst['xmin']})
                            src_possible_range.append({'xmin':dst['xmax'], 'xmax':src_range['xmax']})
                            break

        edge_verti = np.array(edge_verti)
        edge_attr_verti = np.array(edge_attr_verti, dtype=np.float)
        return edge_verti, edge_attr_verti

    # def get_features(self):
    #     x = np.array([], dtype = np.float32).reshape((0,772)) # num_feature is 772
    #     pos_row_iter = self.df.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']].iterrows()
    #     text_row_iter = iter(self.df['text'])
    #     for pos, text in zip(pos_row_iter , text_row_iter):
    #         norm_pos = self.get_normalize_pos(pos[1]) #pos[1] => get only position not index (pos[0])
    #         text_features = self.get_text_features_bert(text)
    #         x = np.append(x, np.concatenate((norm_pos, text_features)).reshape(1, -1), axis = 0)
    #     return x

    def get_features(self):
        self.df['xmin_norm'] = self.df['xmin']/self.w
        self.df['ymin_norm'] = self.df['ymin']/self.h
        self.df['xmax_norm'] = self.df['xmax']/self.w
        self.df['ymax_norm'] = self.df['ymax']/self.h
        all_texts = list(self.df['text'])
        text_feature = self.get_text_features_bert(all_texts)
        pos_feature = np.array(self.df.loc[:,['xmin_norm', 'ymin_norm', 'xmax_norm', 'ymax_norm']])
        feature = np.concatenate((pos_feature, text_feature), axis = 1)
        return feature


    def get_normalize_pos(self, pos):
        """
        param pos: position as xmin, ymin, xmax, ymax
        """
        return np.array((pos['xmin']/self.w, pos['ymin']/self.h, pos['xmax']/self.w, pos['ymax']/self.h))
    
    def get_text_random(self, all_text):
        return np.random.rand(len(all_text),768)

    def get_text_features(self, all_text):
        text_features = []
        for text in all_text:
            n_alpha = 0
            n_numeric = 0
            n_special = 0
            text = text.replace(' ', '')
            for char in text:
                if char.isalpha():n_alpha  += 1
                elif char.isnumeric():n_numeric += 1
                else:n_special += 1
            
            text_features.append([n_alpha, n_numeric, n_special])
        return np.array(text_features)

    def get_text_features_bert(self, text):
        text_feature = np.array(self.extractor(text)).mean(axis = 1)
        return text_feature

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
            return np.arctan2(numerator, denominator)

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