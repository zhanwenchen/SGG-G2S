from os import environ as os_environ
from os.path import join as os_path_join
from json import load as json_load
from copy import deepcopy
from tqdm import tqdm
from numpy import array as np_array, asarray as np_asarray, rint as np_rint
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, to_pil_image
from PIL.Image import fromarray
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
# DATA_DIR = os_environ['DATA_DIR_VG_RCNN']

# with open(os_path_join(DATA_DIR, 'visual_genome', 'VG-SGG-dicts-with-attri.json'), 'r') as fin:
#     scene_graph_meta = json_load(fin)
# # labels = list(scene_graph_meta['label_to_idx'].keys())
# # preds = list(scene_graph_meta['predicate_to_idx'].keys())
# idx2labels = ['_'] + list(scene_graph_meta['idx_to_label'].values())
# # idx2labels = list(scene_graph_meta['idx_to_label'].values())
# idx2predicates = ['_'] + list(scene_graph_meta['idx_to_predicate'].values())


# def draw_triplets(img, row):
# #     if isinstance(img, Tensor):
# #         img_copy = to_pil_image(img)
# #     else:
# #         img_copy = deepcopy(img)
#     draw = ImageDraw.Draw(img)
#     bbox_subj = row[['subj_gtbox_1', 'subj_gtbox_2', 'subj_gtbox_3', 'subj_gtbox_4']].tolist()
#     draw.rectangle(bbox_subj)
#     obj_subj = idx2labels[row['subj_obj_category_idx']]
#     draw.text([bbox_subj[0], bbox_subj[1]], f'subj={obj_subj}')
    
#     bbox_obj = row[['obj_gtbox_1', 'obj_gtbox_2', 'obj_gtbox_3', 'obj_gtbox_4']].tolist()
#     draw.rectangle(bbox_obj)
#     obj_obj = idx2labels[row['obj_obj_category_idx']]
#     draw.text([bbox_obj[0], bbox_obj[1]], f'obj={obj_obj}')
    
#     rel_category_idx = row['rel_category_idx']
#     rel = idx2predicates[rel_category_idx]
#     draw.text([abs(bbox_subj[0]-bbox_obj[0])/2+min(bbox_subj[0], bbox_obj[0]), abs(bbox_subj[1]-bbox_obj[1])/2+min(bbox_subj[1], bbox_obj[1])], rel)
#     display(img)
    
    
# def draw(source_img, bbox, obj):
# #     source_img = Image.open(file_name).convert("RGBA")
#     draw = ImageDraw.Draw(source_img)
#     draw.rectangle(bbox)
#     draw.text([bbox[0], bbox[1]], obj)
# #     source_img.show()
#     display(source_img)

# NOTE: PIL (W, H) => np (H, W)

names = [
    'example_idx',
    'subj_obj_idx_local',
    'subj_obj_category_idx',
    'subj_gtbox_1',
    'subj_gtbox_2',
    'subj_gtbox_3',
    'subj_gtbox_4',
    'obj_obj_idx_local',
    'obj_obj_category_idx',
    'obj_gtbox_1',
    'obj_gtbox_2',
    'obj_gtbox_3',
    'obj_gtbox_4',
    'rel_local_idx',
    'rel_category_idx',
]


class GraftAugmenterDataset(Dataset):
#     def __init__(self, df_objs, df_objsdf_triplets):
    def __init__(self, cfg):
#         self.df_objs = df_objs
#         self.df_triplets = df_triplets
#         self.groupby = defaultdict(np_array)
#         build
        from maskrcnn_benchmark.data.build import make_data_loader, get_dataset_statistics
        train_data_loader = make_data_loader(
            cfg,
            mode='train',
            is_distributed=False,
            start_iter=0,
            aug=False,
        )

        dataset = train_data_loader.dataset
        self.transforms = deepcopy(dataset.transforms)
        dataset.transforms = None
        # self.img_info = dataset.img_info
        self.get_img_info = dataset.get_img_info
        print('GraftAugmenterDataset.__init__: started running get_dataset_statistics')
        result = get_dataset_statistics(cfg, train_data=dataset, return_lookup=True)
        print('GraftAugmenterDataset.__init__: finished running get_dataset_statistics')
        self.dataset = dataset
        stats = result['stats']
        del result
        print('GraftAugmenterDataset.__init__: started converting stats to df')
        df_stats = DataFrame(stats, columns=names)
        print('GraftAugmenterDataset.__init__: finished converting stats to df')
        # df_stats = df_stats.astype(
        #     {
        #         'example_idx':'int',
        #         'rel_local_idx':'int',
        #         'subj_obj_idx_local':'int',
        #         'subj_obj_category_idx':'int',
        #         'obj_obj_idx_local':'int',
        #         'obj_obj_category_idx':'int',
        #         'rel_local_idx': 'int',
        #         'rel_category_idx': 'int',
        #     }
        #     , copy=True
        # )
        del stats
        print('GraftAugmenterDataset.__init__: started converting stats to df')
        rels_bottom_k = set(df_stats.groupby(['rel_category_idx']).count()['example_idx'].nsmallest(30).index.tolist())
        self.df_stats = df_stats
        print('GraftAugmenterDataset.__init__: started querying for least frequent relations')
        self.df_stats_bottom_k = df_stats.query("rel_category_idx.isin(@rels_bottom_k).values")
        self.img_info = [dataset.img_info[i] for i in self.df_stats_bottom_k['example_idx']]
        print('GraftAugmenterDataset.__init__: finished querying for least frequent relations')

#     def exchange_subj(self, row_id):
#         '''
#         Takes triplet id, outputs a new row with 
#         '''
#         # TODO: should cache these
#         df_stats[].sample

    def swap(self, idx_og, subj_or_obj):
        '''
        swap operates on each rare triplet. # sample size = num_iamges * num_reare_triplet_in_image
        '''
        # TODO: need relations. Keep ones that don't overlap or is that particular one. Drop ones that do.
#         image3 = 
#         relations = 
    #     self.df_objs[obj='']
        # TODO: maybe it's df_stats vs df_stats_bottom_k indexing mismatch?
        # print(f'swap: workingon idx_og={idx_og} and subj_or_obj={subj_or_obj}')
        dataset = self.dataset
        df_stats = self.df_stats
        df_stats_bottom_k = self.df_stats_bottom_k
#         img_og, target_og, index_og = dataset[idx_og]
        # dataset, df_stats_bottom_k mismatch
        row_og = df_stats_bottom_k.loc[idx_og, :]
        # print(f'row_og={row_og}')
#         example_idx_og = df_stats_bottom_k.at[idx_og, 'example_idx']
        example_idx_og = row_og['example_idx']
        # Since we are using the build_dataset implicitly, it's already doing a transform
        img_og, target_og, index_og = dataset[example_idx_og]
#         try:
#             img_og, target_og, index_og = dataset[example_idx_og]
#         except:
#             import pdb; pdb.set_trace()
#         img_og, target = dataset.transforms(img_og, target_og)
#         try:
#             img_og, target = dataset.transforms(img_og, target_og)
#         except:
#             import pdb; pdb.set_trace()
        # print(f'img_og.size={img_og.size}')
        img_og_np = np_array(img_og)
        # print(f'img_og_np.shape={img_og_np.shape}')

        # draw_triplets(img_og, row_og)
        if subj_or_obj == 'subj':
            obj_idx_local_og = row_og['subj_obj_idx_local']
            obj_category_idx_og = row_og['subj_obj_category_idx']
            gtbox_1_og = row_og['subj_gtbox_1']
            gtbox_2_og = row_og['subj_gtbox_2']
            gtbox_3_og = row_og['subj_gtbox_3']
            gtbox_4_og = row_og['subj_gtbox_4']
        else:
            obj_idx_local_og = row_og['obj_obj_idx_local']
            obj_category_idx_og = row_og['obj_obj_category_idx']
            gtbox_1_og = row_og['obj_gtbox_1']
            gtbox_2_og = row_og['obj_gtbox_2']
            gtbox_3_og = row_og['obj_gtbox_3']
            gtbox_4_og = row_og['obj_gtbox_4']
#         obj_idx_local_og = int(obj_idx_local_og)
#         rel_local_idx_og = row_og['rel_local_idx']
#         print(row_og)
#         draw(img_og, [gtbox_1_og, gtbox_2_og, gtbox_3_og, gtbox_4_og], idx2labels[obj_category_idx_og])
#         import pdb; pdb.set_trace()
#         obj_idx_local = row['subj_obj_idx_local']
#         obj_category_idx = row['subj_obj_category_idx']
#         gtbox_1 = row_og['subj_gtbox_1']
#         gtbox_2 = row_og['subj_gtbox_2']
#         gtbox_3 = row_og['subj_gtbox_3']
#         gtbox_4 = row_og['subj_gtbox_4']

#         rows = df_stats_bottom_k[(df_stats_bottom_k['subj_obj_category_idx'] == obj_category_idx_og) | (df_stats_bottom_k['obj_obj_category_idx'] == obj_category_idx_og)]
        rows = df_stats[(df_stats['subj_obj_category_idx'] == obj_category_idx_og) | (df_stats['obj_obj_category_idx'] == obj_category_idx_og)]
#         candidates = rows.sample(n=n, replace=False)
        row_new = rows.sample(n=1, ignore_index=True)
        row_new = row_new.loc[0, :]
    
        idx_new = row_new['example_idx']
        img_new, target_new, index_new = dataset[idx_new]
#         try:
#             img_new, target = dataset.transforms(img_new, target_new)
#         except:
#             import pdb; pdb.set_trace()
        
#         img_new, target_new, index_new = dataset[row_new['example_idx']]
        # print(f'img_new.size={img_new.size}')
#         draw(img_og, [gtbox_1_og, gtbox_2_og, gtbox_3_og, gtbox_4_og], idx2labels[obj_category_idx_og])
        # draw_triplets(img_new, row_new)
        
        subj_or_obj_new = 'subj' if row_new['subj_obj_category_idx'] == obj_category_idx_og else 'obj'
        
        if subj_or_obj_new == 'subj':
            obj_idx_local_new = row_new['subj_obj_idx_local']
            obj_category_idx_new = row_new['subj_obj_category_idx']
            gtbox_1_new = row_new['subj_gtbox_1']
            gtbox_2_new = row_new['subj_gtbox_2']
            gtbox_3_new = row_new['subj_gtbox_3']
            gtbox_4_new = row_new['subj_gtbox_4']
        else:
            obj_idx_local_new = row_new['obj_obj_idx_local']
            obj_category_idx_new = row_new['obj_obj_category_idx']
            gtbox_1_new = row_new['obj_gtbox_1']
            gtbox_2_new = row_new['obj_gtbox_2']
            gtbox_3_new = row_new['obj_gtbox_3']
            gtbox_4_new = row_new['obj_gtbox_4']
#         rel_local_idx_new = row_new['rel_local_idx']
        
#         assert roi_og is an instance of PIL or Tensor (whichever one is consistent)
        img_new_np = np_asarray(img_new)
        # print(f'img_new_np.shape={img_new_np.shape}')
#         roi_new = img_new[:, gtbox_1_new:gtbox_3_new, gtbox_2_new:gtbox_4_new]
#         roi_new = img_new.crop(box=[gtbox_1_new, gtbox_2_new, gtbox_3_new, gtbox_4_new])
#         img_og = img_og.paste(roi_new, box=[gtbox_1_og, gtbox_2_og, gtbox_3_og, gtbox_4_og])
#         try:
#             roi_og = img_og[:, gtbox_1_og:gtbox_3_og, gtbox_2_og:gtbox_4_og]
#         except:
#             import pdb; pdb.set_trace()
#         roi_og.paste(im[, box, mask])
#         roi_new_np = img_new_np[:, gtbox_1_new:gtbox_3_new, gtbox_2_new:gtbox_4_new]
#         roi_new_np = img_new_np[gtbox_1_new:gtbox_3_new, gtbox_2_new:gtbox_4_new, :]
        roi_new_np = img_new_np[gtbox_2_new:gtbox_4_new, gtbox_1_new:gtbox_3_new, :]
        # print(f'roi_new_np.shape={roi_new_np.shape}')
        try:
            roi_new = fromarray(roi_new_np)
            # print(f'roi_new.size={roi_new.size}')
        except:
            import pdb; pdb.set_trace()
        roi_new_resized = resize(roi_new, [gtbox_4_og-gtbox_2_og, gtbox_3_og-gtbox_1_og]) # takes H,W
        # print(f'roi_new_resized.size={roi_new_resized.size}')
#         roi_new_resized = resize(roi_new, [gtbox_3_og-gtbox_1_og, gtbox_4_og-gtbox_2_og]) # takes H,W
        roi_new_resized_np = np_asarray(roi_new_resized)
        # print(f'roi_new_resized_np.shape={roi_new_resized_np.shape}')
#         img_og_np[gtbox_1_og:gtbox_3_og, gtbox_2_og:gtbox_4_og, :] = roi_new_resized_np
        try:
            img_og_np[gtbox_2_og:gtbox_4_og, gtbox_1_og:gtbox_3_og, :] = np_rint(0.5 * img_og_np[gtbox_2_og:gtbox_4_og, gtbox_1_og:gtbox_3_og, :] + 0.5 * roi_new_resized_np) #(1024, 683, 3) => (311, 173, 3)
#             img_og_np[gtbox_1_og:gtbox_3_og, gtbox_2_og:gtbox_4_og, :] = roi_new_resized_np.swapaxes(0,1)
        except:
            import pdb; pdb.set_trace()
        img_og_modified = fromarray(img_og_np)
        # print(f'img_og_modified.size={img_og_modified.size}')
        # draw_triplets(img_og_modified, row_og)
#         draw(img_og_modified, [gtbox_1_og, gtbox_2_og, gtbox_3_og, gtbox_4_og], idx2labels[obj_category_idx_og])
#         img_og.show()
#         img_og_modified.show()
        return img_og_modified, target_og, None # set index to None
    
    def __getitem__(self, index):
        # df_stats_bottom_k differs from df_stats. Triplet index.
        # QUESTION: is this different from example index?
        # BUG: for each triplet, output an example. This is probolematic.
        # SOLUTION: 
        idx_og = self.df_stats_bottom_k.index[index]
        img, target, index = self.swap(idx_og, 'subj')
        img, target = self.transforms(img, target)
        return img, target, idx_og
        # rows_new = []
#         for rel_bottom_k in rels_bottom_k:
#         from tqdm.notebook import tqdm
#         swap = self.swap
# #         tqdm.pandas()
#         for idx, _ in tqdm(self.df_stats_bottom_k.iterrows()):
#             row_new_subj = swap(idx, 'subj')
#             rows_new.append(row_new_subj)
#             row_new_obj = swap(idx, 'obj')
#             rows_new.append(row_new_obj)

        return rows_new
        
    def __len__(self):
        return len(self.df_stats_bottom_k)
    
    def run(self):
        rows_new = []
#         for rel_bottom_k in rels_bottom_k:
#         from tqdm.notebook import tqdm
        swap = self.swap
#         tqdm.pandas()
        for idx, _ in tqdm(self.df_stats_bottom_k.iterrows()):
            row_new_subj = swap(idx, 'subj')
            rows_new.append(row_new_subj)
            row_new_obj = swap(idx, 'obj')
            rows_new.append(row_new_obj)

        return rows_new
                
        # TODO: make sure H, W order is correct
#         pass
#         for _ in range(self.num_mix):
#             r = np.random.rand(1)
#             if self.beta <= 0 or r > self.prob
#                 continue

#             # generate mixed sample
#             lam = np.random.beta(self.beta, self.beta)
#             rand_index = random.choice(range(len(self)))

#             img2, lb2 = self.dataset[rand_index]
#             lb2_onehot = onehot(self.num_class, lb2)

#             bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
#             img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
#             lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
#             lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

# Actually won't affect relations because of the resizeing. This augmentation is entirely visual

# So the problem is that we need to make a new image with both existing and new triplets. 
# We need to make sure that the new ones don't overlap with the old ones.


# self.dataset.gt_boxes[row_og['example_idx']][row_og['subj_obj_idx_local']] if subj_or_obj == 'subj' else self.dataset.gt_boxes[row_og['example_idx']][row_og['obj_obj_idx_local']]