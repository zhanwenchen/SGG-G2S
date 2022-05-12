from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

from json import load as json_load
from random import random as random_random

from torch import FloatTensor as torch_FloatTensor, LongTensor as torch_LongTensor
from torch.utils.data import Dataset, DataLoader


class SGEncoding(Dataset):
    """ SGEncoding dataset """
    def __init__(self, train_ids, test_ids, sg_data, test_on=False, val_on=False, num_test=5000, num_val=5000):
        super(SGEncoding, self).__init__()
        data_path = './datasets/vg/'
        with open(data_path+'vg_capgraphs_anno.json') as f:
            cap_graph = json_load(f)

        with open(data_path+'VG-SGG-dicts-with-attri.json') as f:
            vg_dict = json_load(f)
        self.img_txt_sg = sg_data
        self.key_list = list(self.img_txt_sg.keys())
        self.key_list.sort()
        self.train_ids = train_ids
        self.test_ids = test_ids
        if test_on:
            self.key_list = self.test_ids[:num_test]
        elif val_on:
            self.key_list = self.test_ids[num_test:num_test + num_val]
        else:
            self.key_list = self.test_ids[num_test + num_val:] + self.train_ids
        print('number of list:', len(self.key_list))

        # generate union predicate vocabulary
        self.sgg_rel_vocab = list(set(cap_graph['idx_to_meta_predicate'].values()))
        self.txt_rel_vocab = list(set(cap_graph['cap_predicate'].keys()))

        # generate union object vocabulary
        self.sgg_obj_vocab = list(set(vg_dict['idx_to_label'].values()))
        self.txt_obj_vocab = list(set(cap_graph['cap_category'].keys()))

        # vocabulary length
        self.num_sgg_rel = len(self.sgg_rel_vocab)
        self.num_txt_rel = len(self.txt_rel_vocab)
        self.num_sgg_obj = len(self.sgg_obj_vocab)
        self.num_txt_obj = len(self.txt_obj_vocab)

    def _to_tensor(self, inp_dict):
        return {'entities': torch_LongTensor(inp_dict['entities']),
                'relations': torch_LongTensor(inp_dict['relations'])}

    def _generate_tensor_by_idx(self, idx):
        img = self._to_tensor(self.img_txt_sg[self.key_list[idx]]['img'])
        img_graph = torch_FloatTensor(self.img_txt_sg[self.key_list[idx]]['image_graph'])

        txt = self._to_tensor(self.img_txt_sg[self.key_list[idx]]['txt'])
        txt_graph = torch_FloatTensor(self.img_txt_sg[self.key_list[idx]]['text_graph'])

        img['graph'] = img_graph
        txt['graph'] = txt_graph
        return img, txt

    def __getitem__(self, item):
        fg_img, fg_txt = self._generate_tensor_by_idx(item)
        # generate negative sample
        bg_idx = item
        while (bg_idx == item):
            bg_idx = int(random_random() * len(self.key_list))
        bg_img, bg_txt = self._generate_tensor_by_idx(bg_idx)
        return fg_img, fg_txt, bg_img, bg_txt

    def __len__(self):
        return len(self.key_list)


class SimpleCollator(object):
    def __call__(self, batch):
        return list(zip(*batch))


def get_loader(cfg, train_ids, test_ids, sg_data, test_on=False, val_on=False, num_test=5000, num_val=1000):
    """ Returns a data loader for the desired split """
    split = SGEncoding(train_ids, test_ids, sg_data=sg_data, test_on=test_on, val_on=val_on, num_test=num_test,
                       num_val=num_val)

    return DataLoader(split,
                        batch_size=cfg.SOLVER.IMS_PER_BATCH,
                        shuffle=not (test_on or val_on),  # only shuffle the data in training
                        pin_memory=True,
                        num_workers=8,
                        collate_fn=SimpleCollator(),
                       )
