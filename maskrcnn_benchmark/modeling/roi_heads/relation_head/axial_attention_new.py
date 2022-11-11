from torch import (
    nn,
    bmm as torch_bmm,
    sum as torch_sum,
)
# from torch.cuda import empty_cache
from operator import itemgetter

# helper functions

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    # total_dimensions = num_dimensions + 1
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        # breakpoint()
        axial = x.permute(*self.permutation).contiguous() # Old: torch.Size([1, 506, 506, 768]). New: [0, 1 ,2]. self.permutation: Old: [0, 2, 1, 3]
        del x
        # breakpoint()
        shape = axial.shape
        # breakpoint()
        *_, t, d = shape

        # merge all but axial dimension
        # breakpoint()
        axial = axial.reshape(-1, t, d)

        # attention
        # breakpoint()
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        # breakpoint()
        axial = axial.reshape(*shape)
        # breakpoint()
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


# attention
class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        print(f'x.size()={x.size()}')
        # breakpoint()
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads
        # lol_b, lol_t, lol_d, lol_h, lol_e = *q.shape, self.heads, self.dim_heads

        print('PRE:')
        print(f'q.size()={q.size()}')
        print(f'k.size()={k.size()}')
        print(f'v.size()={v.size()}')
        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        # breakpoint()
        # breakpoint()
        q, k, v = map(merge_heads, (q, k, v))
        #
        print('POST:')
        print(f'q.size()={q.size()}')
        print(f'k.size()={k.size()}')
        print(f'v.size()={v.size()}')
        print(f'q.is_contiguous()={q.is_contiguous()}')
        print(f'k.is_contiguous()={k.is_contiguous()}')
        # breakpoint()
        # empty_cache()
        # print(f'q.size()={q.size()}')
        q = q.contiguous()
        k = k.transpose_(1, 2).contiguous()
        dots = torch_bmm(q, k)
        dots *= e ** -0.5

        del q, k
        # dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        # breakpoint()
        # dots = dots.softmax(dim=-1)
        dots.exp_()
        # torch.exp(dots, out=t)
        summed = torch_sum(dots, dim=-1, keepdim=True)
        dots /= summed
        del summed
        # breakpoint()
        out = torch_bmm(dots, v)
        del dots, v
        # out = torch.einsum('bij,bje->bie', dots, v)
        # breakpoint()
        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        # out_new = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        # breakpoint()
        out = self.to_out(out)
        # breakpoint()
        return out

# axial attention class

class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        # self.total_dimensions = num_dimensions + 1
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            # breakpoint()
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        # breakpoint()
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            breakpoint()
            out = axial_attn(out)
        return out
