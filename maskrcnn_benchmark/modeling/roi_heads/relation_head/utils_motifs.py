import sys
from array import array as array_array
from os import makedirs as os_makedirs
from os.path import isfile as os_path_isfile, join as os_path_join, exists as os_path_exists, basename as os_path_basename
import zipfile
from six import binary_type as six_binary_type
from six.moves.urllib.request import urlretrieve
from torch import save as torch_save, sigmoid as torch_sigmoid, float32 as torch_float32, zeros as torch_zeros, nonzero as torch_nonzero, as_tensor as torch_as_tensor, cat as torch_cat, int64 as torch_int64, device as torch_device, rand as torch_rand, arange as torch_arange, LongTensor as torch_LongTensor, Tensor as torch_Tensor, load as torch_load, sort as torch_sort
from numpy import cumsum as np_cumsum, concatenate as np_concatenate, save as np_save
from tqdm import tqdm
from maskrcnn_benchmark.modeling.utils import cat


def normalize_sigmoid_logits(orig_logits):
    orig_logits = torch_sigmoid(orig_logits)
    return orig_logits / (orig_logits.sum(1).unsqueeze(-1) + 1e-12)

def generate_attributes_target(attributes, device, max_num_attri, num_attri_cat):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert max_num_attri == attributes.shape[1]
        num_obj = attributes.shape[0]

        with_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch_zeros((num_obj, num_attri_cat), device=device, dtype=torch_float32)

        for idx in torch_nonzero(with_attri_idx).squeeze(1).tolist():
            for k in range(max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, with_attri_idx

def transpose_packed_sequence_inds(lengths):
    """
    Get a TxB indices from sorted lengths.
    Fetch new_inds, split by new_lens, padding to max(new_lens), and stack.
    Returns:
        new_inds (np.array) [sum(lengths), ]
        new_lens (list(np.array)): number of elements of each time step, descending
    """
    new_inds = []
    new_lens = []
    cum_add = np_cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer+1)].copy())
        cum_add[:(length_pointer+1)] += 1
        new_lens.append(length_pointer+1)
    new_inds = np_concatenate(new_inds, 0)
    return new_inds, new_lens


def sort_by_score(proposals, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_rois = [len(b) for b in proposals]
    num_im = len(num_rois)

    scores = scores.split(num_rois, dim=0)
    ordered_scores = []
    for i, (score, num_roi) in enumerate(zip(scores, num_rois)):
        ordered_scores.append( score - 2.0 * float(num_roi * 2 * num_im + i) )
    ordered_scores = cat(ordered_scores, dim=0)
    _, perm = torch_sort(ordered_scores, 0, descending=True)

    num_rois = sorted(num_rois, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(num_rois)  # move it to TxB form
    inds = torch_as_tensor(inds, dtype=torch_int64, device=scores[0].device)
    ls_transposed = torch_LongTensor(ls_transposed)

    perm = perm[inds] # (batch_num_box, )
    _, inv_perm = torch_sort(perm)

    return perm, inv_perm, ls_transposed


def to_onehot(vec, num_classes, fill=1000):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = fill

    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :return:
    """
    vec_size_0 = vec.size(0)
    onehot_result = vec.new_full((vec_size_0, num_classes), -fill, dtype=torch_float32)
    arange_inds = torch_arange(0, vec_size_0, dtype=torch_int64, device=vec.device)

    onehot_result.view(-1)[vec.long() + num_classes*arange_inds] = fill
    return onehot_result


def get_dropout_mask(dropout_probability, tensor_shape, device):
    """
    once get, it is fixed all the time
    """
    binary_mask = (torch_rand(tensor_shape, device=device, dtype=torch_float32) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    return binary_mask.to(device, dtype=torch_float32).div(1.0 - dropout_probability)


def center_x(proposals):
    assert proposals[0].mode == 'xyxy'
    boxes = cat([p.bbox for p in proposals], dim=0)
    c_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    return c_x.view(-1)


def encode_box_info(proposals):
    """
    encode proposed box information (x1, y1, x2, y2) to
    (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """
    assert proposals[0].mode == 'xyxy'
    boxes_info = []
    for proposal in proposals:
        boxes = proposal.bbox
        img_size = proposal.size
        wid = img_size[0]
        hei = img_size[1]
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        w, h = wh.split([1,1], dim=-1)
        x, y = xy.split([1,1], dim=-1)
        x1, y1, x2, y2 = boxes.split([1,1,1,1], dim=-1)
        assert wid * hei != 0
        info = torch_cat([w/wid, h/hei, x/wid, y/hei, x1/wid, y1/hei, x2/wid, y2/hei,
                          w*h/(wid*hei)], dim=-1).view(-1, 9)
        boxes_info.append(info)

    return torch_cat(boxes_info, dim=0)


def obj_edge_vectors(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch_Tensor(len(names), wv_dim)
    vectors.normal_(0,1)

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Try the longest word
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print("fail on {}".format(token))
    vectors_np = vectors.cpu().numpy()
    if len(vectors_np)<60:
        np_save('./misc/predicates_w2v.npy', vectors_np)
    return vectors

def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os_path_join(root, wv_type + '.' + dim)

    fname_pt = f'{fname}.pt'
    if os_path_isfile(fname_pt):
        print('loading word vectors from', fname_pt)
        try:
            return torch_load(fname_pt, map_location=torch_device("cpu"))
        except Exception as e:
            print("Error loading the model from {}{}".format(fname_pt, str(e)))
            sys.exit(-1)
    fname_txt = f'{fname}.txt'
    if os_path_isfile(fname_txt):
        with open(fname_txt, 'rb') as f:
            cm = list(f)
    elif os_path_basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os_path_basename(fname)
        if not os_path_exists(root):
            os_makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os_path_isfile(fname_txt):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array_array('d'), None
    if cm is not None:
        for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six_binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch_Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch_save(ret, fname_pt)
    return ret

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]
    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner
