import torch
from torch_geometric.data import InMemoryDataset
import shutil

class RVL_Dataset(InMemoryDataset):
    def __init__(self, root, dir, transform=None, pre_transform=None): 
        self.dir = dir
        self.targets = ['supplier', 'invoice_info', 'receiver', 'positions', 'total', 'other']
        self.ocr_files = iter(sorted(glob.glob(osp.join(self.dir, '*ocr.xml'))))
        self.gt_files = iter(sorted(glob.glob(osp.join(self.dir, '*gt.xml'))))
        self.image_files = iter(sorted(glob.glob(osp.join(self.dir, '*tif'))))
        self.grapher = GraphOCR()
        super(RVL_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(osp.join(self.root, 'processed', 'RVL_Dataset.pt'))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['RVL_Dataset.pt']
        # return []
    
    def download(self):
        pass

    def process(self):
        if not osp.isdir(osp.join(self.root, 'plot_graph/')):
            os.mkdir(osp.join(self.root, 'plot_graph/'))
        data_list = []
        i = 0
        for ocr_file, gt_file, image_file in zip(self.ocr_files, self.gt_files, self.image_files):
            print(i)
            i+=1
            start = time.time()
            ocr = ET.parse(ocr_file).getroot()
            gt = ET.parse(gt_file).getroot()
            img = cv2.imread(image_file)
            row = []
            h, w = img.shape[1], img.shape[0]
            gt_dict = self.get_gt_dict(gt)
            if np.all([not v for v in gt_dict.values()]): #check if ground truth has value.
                continue
            img = self.visualize_gt(gt_dict, img)
            for word in ocr.iter('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Word'):
                points_string = word.find('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords').get('points')
                conf = word.find('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextEquiv').get('conf')
                if float(conf) < 0.2:
                    continue
                xmin, ymin, xmax, ymax = self.find_xy_bbox(points_string)
                text = list(word.iter('{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Unicode'))[0].text
                label = self.find_label(gt_dict, (xmin, ymin, xmax, ymax))
                row.append((xmin,ymin,xmax,ymax,text,label))
            df = pd.DataFrame(row, columns=['xmin', 'ymin', 'xmax', 'ymax', 'text', 'label'])
            df['xcen'] = (df['xmin'] + df['xmax']) / 2
            df['ycen'] = (df['ymin'] + df['ymax']) / 2            
            if df.shape[0] == 0:
                continue
            x, y, edge_index, edge_attr, edge_label = self.grapher.connect(df, h, w, gt_dict)
            save_name = osp.join(self.root, 'plot_graph', osp.basename(image_file)[:-3]+'jpg')
            self.visualise(img, df, edge_index, edge_attr, edge_label, save_name)
            data = Data(x=torch.tensor(x, dtype=torch.float),
                        y=torch.tensor(y, dtype=torch.long),
                        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                        edge_label = torch.tensor(edge_label, dtype=torch.long),
                        edge_attr = torch.tensor(edge_attr, dtype=torch.float))
            print(f'time use {time.time() - start} seconds')
            

            if self.pre_filter is not None and not self.pre_filter(data):
                os.remove(image_file)
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
        torch.save((data, slices), osp.join(self.root, 'processed', 'RVL_Dataset.pt'))

    def get_gt_dict(self, gtxml):
        """
        get ground truth label of node
        """
        namespace = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}'
        gt_dict = {t:[] for t in self.targets}
        for region in gtxml.iter(namespace + 'TextRegion'):
            points_string = region.find(namespace + 'Coords').get('points')
            xmin, ymin, xmax, ymax = self.find_xy_bbox(points_string)
            prop = region.find(namespace + 'Property')
            if prop is None:
                gt_dict['other'].append((xmin, ymin, xmax, ymax))
            else:
                gt_dict[prop.get('value')].append((xmin, ymin, xmax, ymax))
        return gt_dict

    def find_xy_bbox(self, points_string):
        """
        return xmin, ymin, xmax, ymax
        """
        points = np.array([x.split(',') for x in points_string.split(' ')], dtype = np.float).astype(np.int)
        x = points[:,0]
        y = points[:,1]
        xmin = min(x)
        ymin = min(y)
        xmax = max(x)
        ymax = max(y)
        return xmin, ymin, xmax, ymax

    def find_label(self, gt_dict, bbox):
        x0 = np.mean([bbox[0], bbox[2]])
        y0 = np.mean([bbox[1], bbox[3]])
        label = None
        for region in gt_dict.keys():
            for gt_box in gt_dict[region]:
                if gt_box[0] < x0 < gt_box[2] and gt_box[1] < y0 < gt_box[3]:
                    label = region
        if label is None:
            label = 'other'
        return label

    def visualize_gt(self, gt_dict, img):
        for k, v in gt_dict.items():
            for box in v:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 1)
                cv2.putText(img, k, (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (124,100,245), 1)
        return img
    
    def visualise(self, img, df, edge_index, edge_attr, edge_label, save_name):
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