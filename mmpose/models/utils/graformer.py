from __future__ import absolute_import
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_upsample_layer
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Variable
from typing import Optional, Sequence, Tuple, Union
from mmpose.registry import KEYPOINT_CODECS, MODELS
from torch import Tensor, nn
OptIntSeq = Optional[Sequence[int]]
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z
class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

class _GraphConv1(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv1, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        #in here if our batch size equal to 64

        # x = self.gconv(x).transpose(1, 2).contiguous()
        x = self.gconv(x)
        # x = self.bn(x).transpose(1, 2).contiguous()
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x

class _GraphConv_no_bn(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv_no_bn, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)

    def forward(self, x):
        #in here if our batch size equal to 64
        # x = self.gconv(x).transpose(1, 2).contiguous()
        x = self.gconv(x)
        return x



class _ResGraphConv_Attention(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv_Attention, self).__init__()

        self.gconv1 = _GraphConv1(adj, input_dim, hid_dim//2, p_dropout)


        self.gconv2 = _GraphConv_no_bn(adj, hid_dim//2, output_dim, p_dropout)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.attention = Node_Attention(output_dim)

    def forward(self,x,joint_features=None):
        if joint_features is None:
            residual = x
        else:
            # joint_features = joint_features.transpose(1,2).contiguous()
            x = torch.cat([joint_features,x],dim=2)
            residual = x

        out = self.gconv1(x)
        out = self.gconv2(out)

        # out = self.bn(residual.transpose(1,2).contiguous() + out)
        out = self.bn(residual + out)
        out = self.relu(out)

        # out = self.attention(out).transpose(1,2).contiguous()
        out = self.attention(out)
        return out

class Node_Attention(nn.Module):
    def __init__(self,channels):
        '''
        likely SElayer
        '''
        super(Node_Attention,self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.squeeze = nn.Sequential(
            nn.Linear(channels,channels//4),
            nn.ReLU(),
            nn.Linear(channels//4,17),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.avg(x).squeeze(2)
        out = self.squeeze(out)
        out = out[:,None,:]
        out = out
        out = (x+x*out)
        return out

class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        #very useful demo means this is Parameter, which can be adjust by bp methods
        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):

        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])



        adj = -9e15 * torch.ones_like(self.adj).to(input.device)

        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)




        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N]
        mul_L = self.cheb_polynomial(L).unsqueeze(1)   # [K, 1, N, N]
        mul_L=mul_L.float()
        inputs = inputs.float()

        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]

        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:

            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L
class _GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = ChebConv(input_dim, output_dim, K=2)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, adj):
        x = self.gconv(x, adj)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResChebGC(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResChebGC, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x, self.adj)
        out = self.gconv2(out, self.adj)
        return residual + out

fc_out = 256
fc_unit = 1024
class refine(nn.Module):
    def __init__(self, dim):
        super().__init__()

        fc_in = dim * 2 * 1 * 17
        fc_out = 17 * 2

        self.post_refine = nn.Sequential(
            nn.Linear(fc_in, fc_unit),
            nn.GELU(),
            # nn.ReLU(),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(fc_unit, fc_out),
            nn.Sigmoid()
        )

    def forward(self, x, x_1):
        B, J, _ = x.size()
        x_in = torch.cat((x, x_1), -1)
        x_in = x_in.view(B, -1)

        score = self.post_refine(x_in).view(B, J, 2)
        score_1 = Variable(torch.ones(score.size()), requires_grad=False).cuda() - score
        x_out = x.clone()
        x_out[:,:, :2] = score * x[:, :, :2] + score_1 * x_1[:, :, :2]

        return x_out

edges = torch.tensor([[0, 1], [0,2], [1, 3],[2, 4],[1,2],
                              [5, 6],[6,8],[8,10],[5,7],[7,9],[4,6],[3,5],
                              [5,11],[11,13],[13,15],[11,12],
                              [6, 12], [12, 14], [14, 16]], dtype=torch.long)

# edges = torch.tensor([[12,13], [13,0], [13,1],[0,2],[2, 4],
#                               [1, 3],[3,5],[2,3],[2,6],[6,8],[8,10],
#                            [3,7],[7,9],[9,11]], dtype=torch.long)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float,device='cuda')
    return adj_mx


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features=layer.size=512
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(size, 3 * size, bias=True)
        )

    def forward(self, x, sublayer, c):
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=2)
        return x + gate * self.dropout(sublayer(modulate(self.norm(x), shift, scale)))


class GraAttenLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GraAttenLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, c):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask), c)
        return self.sublayer[1](x, self.feed_forward, c)


def attention(Q, K, V, mask=None, dropout=None):
    # Query=Key=Value: [batch_size, 8, max_len, 64]
    d_k = Q.size(-1)
    # Q * K.T = [batch_size, 8, max_len, 64] * [batch_size, 8, 64, max_len]
    # scores: [batch_size, 8, max_len, max_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, V), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        Q, K, V = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)

        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True]]])


class LAM_Gconv(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(LAM_Gconv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GraphNet(nn.Module):

    def __init__(self, in_features=2, out_features=2, n_pts=21):
        super(GraphNet, self).__init__()

        self.A_hat = Parameter(torch.eye(n_pts).float(), requires_grad=True)  # 生成一个对角矩阵
        self.gconv1 = LAM_Gconv(in_features, in_features * 2)
        self.gconv2 = LAM_Gconv(in_features * 2, out_features, activation=None)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1



class GraFormer(nn.Module):
    def __init__(self, hid_dim=128, num_layers=4,
                 n_head=4, dropout=0.1, n_pts=17,final_layer: dict = dict(kernel_size=1)):
        super(GraFormer, self).__init__()
        self.n_layers = num_layers
        self.n_pts=n_pts

        self.adj = adj_mx_from_edges(num_pts=self.n_pts, edges=edges, sparse=False)

        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                                        True, True, True, True, True, True, True]]])
        # self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
        #                                 True, True, True, True]]])
        self.src_mask = self.src_mask.to('cuda')

        _gconv_input = ChebConv(in_c=2, out_c=hid_dim, K=2)

        _gconv_cond = ChebConv(in_c=2, out_c=hid_dim, K=2)

        _gconv_cond1= ChebConv(in_c=self.n_pts, out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hid_dim),
            nn.Linear(hid_dim, hid_dim * 2),
            # nn.GELU(),
            nn.SiLU(),
            nn.Linear(hid_dim * 2, hid_dim),
        )

        dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                            hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_input = _gconv_input
        self.gconv_cond = _gconv_cond
        self.gconv_cond1=_gconv_cond1
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)

        self.gconv_output = ChebConv(in_c=dim_model, out_c=2, K=2)
        self.fusion = refine(2)

        self.gconv_layers1 = _ResGraphConv_Attention(self.adj, 128, 128, 128, p_dropout=None)
        self.gconv_layers2 = _ResGraphConv_Attention(self.adj, 128+self.n_pts, 128+self.n_pts, 128+self.n_pts,
                                                     p_dropout=None)
        self.gconv_layers3 = _ResGraphConv_Attention(self.adj, 128 + 64*48, 128 + 64*48, 128 + 64*48,
                                                     p_dropout=None)


        self.FC = nn.Sequential(
            nn.Sequential(nn.Linear(64*48, 1024), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(inplace=True)),
            nn.Linear(1024, 2))

        self.fc_last = nn.Linear(128+self.n_pts, 128)
        self.fc_last1 = nn.Linear(128 + 64*48, 128)
        self.fc_cond = nn.Linear(64*48,128)

        cfg = dict(
                type='Conv2d',
                in_channels=32,
                out_channels=17,
                kernel_size=1)
        cfg.update(final_layer)
        self.final_layer = build_conv_layer(cfg)

    def conv_head(self,feats: Tuple[Tensor]) -> Tensor:
        x = feats[-1]
        # 得到B,17,64,48
        x = self.final_layer(x)
        return x

    #     self.deconv_head = self._make_deconv_head(
    #         in_channels=32,
    #         out_channels=17,
    #         deconv_type='heatmap',
    #         deconv_out_channels=None,
    #         deconv_kernel_sizes=(4, 4, 4),
    #         deconv_num_groups=(16, 16, 16),
    #         conv_out_channels=None,
    #         conv_kernel_sizes=None,
    #         final_layer=dict(kernel_size=1))
    #
    # def _make_deconv_head(
    #     self,
    #     in_channels: Union[int, Sequence[int]],
    #     out_channels: int,
    #     deconv_type: str = 'heatmap',
    #     deconv_out_channels: OptIntSeq = (256, 256, 256),
    #     deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
    #     deconv_num_groups: OptIntSeq = (16, 16, 16),
    #     conv_out_channels: OptIntSeq = None,
    #     conv_kernel_sizes: OptIntSeq = None,
    #     final_layer: dict = dict(kernel_size=1)
    # ) -> nn.Module:
    #     """Create deconvolutional layers by given parameters."""
    #
    #     if deconv_type == 'heatmap':
    #         deconv_head = MODELS.build(
    #             dict(
    #                 type='HeatmapHead',
    #                 in_channels=in_channels,
    #                 out_channels=out_channels,
    #                 deconv_out_channels=deconv_out_channels,
    #                 deconv_kernel_sizes=deconv_kernel_sizes,
    #                 conv_out_channels=conv_out_channels,
    #                 conv_kernel_sizes=conv_kernel_sizes,
    #                 final_layer=final_layer))
    #     else:
    #         deconv_head = MODELS.build(
    #             dict(
    #                 type='ViPNASHead',
    #                 in_channels=in_channels,
    #                 out_channels=out_channels,
    #                 deconv_out_channels=deconv_out_channels,
    #                 deconv_num_groups=deconv_num_groups,
    #                 conv_out_channels=conv_out_channels,
    #                 conv_kernel_sizes=conv_kernel_sizes,
    #                 final_layer=final_layer))
    #
    #     return deconv_head

    # def forward(self, x_t,x_c, t,heatmap):
    #     B,J, _ = x_t.shape
    #     x= self.gconv_input(x_t, self.adj)  #(B,J,128)
    #     _, J, C = x.shape #C=hid_dim
    #
    #     #节点特征提取
    #     heat_map_intergral = self.FC(heatmap.view(B * 17, -1)).view(B, 17 * 2)
    #     # heat_map_intergral = self.FC(heatmap.view(bs * 14, -1)).view(bs, 28)
    #     hm_4 = heat_map_intergral.view(-1, 17, 2)
    #     # hm_4 = heat_map_intergral.view(-1, 14, 2)
    #     j_1_4 = F.grid_sample(heatmap, hm_4[:, None, :, :]).squeeze(2)  # B 17 17
    #
    #     #节点特征结合条件特征
    #     out = self.gconv_cond(x_c, self.adj).reshape(B, J, C) # 32 17 128
    #     out = self.gconv_layers1(out)  # 32 17 128
    #     out = self.gconv_layers2(out, joint_features=j_1_4)
    #     out = out.unsqueeze(2)  # 32 17 1 128+17
    #     out = out.permute(0, 3, 2, 1)  # 32 128+17 1 17
    #     # out = self.non_local(out)  # 32 128+17 1 17  空的
    #     out = out.permute(0, 3, 1, 2)  # 32 17 128+17 1
    #     out = out.squeeze()  # 32 17 128+17
    #     cond_embed=self.fc_last(out)
    #
    #     # 直接节点特征引导生成
    #     # cond_embed = self.gconv_cond1(j_1_4, self.adj).reshape(B, J, C)
    #
    #
    #     # # 条件特征
    #     # cond_embed= self.gconv_cond(x_c, self.adj).reshape(B, J, C)  # 32 17 128
    #
    #     time_embed = self.time_mlp(t)[:,None, :].repeat(1,J, 1)
    #
    #     c = time_embed + cond_embed
    #     c = c.reshape(B, J, C)
    #
    #     for i in range(self.n_layers):
    #         x = self.atten_layers[i](x, self.src_mask, c)
    #         x = self.gconv_layers[i](x)
    #
    #     x = self.gconv_output(x, self.adj)
    #
    #     x = x.reshape(B, J, -1)
    #     #是否融合
    #     # x_t = x_t.reshape(B, J, -1)
    #     # x = self.fusion(x, x_t)
    #     return x

    def forward(self, x_t, x_c, t, feats, heatmap):
    # def forward(self, x_t, x_c, t):

        # feats = self.conv_head(feats)  # b,17,64,48
        # feats= torch.flatten(feats, 2) #B,17,64*48
        # feats=self.fc_cond(feats)#B,17,128

        # feat=feats[-1]
        # batch_size=heatmap.shape[0]
        # num_joints=heatmap.shape[1]
        # normalized_heatmap = F.softmax(heatmap.reshape(batch_size, num_joints, -1), dim=-1) #B,J,64*48
        # feat =torch.flatten(feat, 2) #B,C,64*48
        # attended_feat=torch.matmul(normalized_heatmap, feat.transpose(2, 1)) #B,J,C
        # attended_feat = self.fc_cond(attended_feat)  # B,17,128

        B, J, _ = x_t.shape
        x = self.gconv_input(x_t, self.adj)  # (B,J,128)
        _, J, C = x.shape  # C=hid_dim

        # #
        # # 图片特征结合关键点特征
        # out = self.gconv_cond(x_c, self.adj).reshape(B, J, C)  # 32 17 128
        # out = self.gconv_layers1(out)  # 32 17 128
        # # out = self.gconv_layers2(out, joint_features=j_1_4)
        # out = self.gconv_layers3(out, joint_features=feats)  # 32 17 128+64*48
        # out = out.unsqueeze(2)  # 32 17 1 128+64*48
        # out = out.permute(0, 3, 2, 1)  # 32 128+64*48 1 17
        # # out = self.non_local(out)  # 32 128+17 1 17  空的
        # out = out.permute(0, 3, 1, 2)  # 32 17 128+64*48 1
        # out = out.squeeze()  # 32 17 128+64*48
        # # cond_embed = self.fc_last(out)
        # cond_embed = self.fc_last1(out)

        # 直接节点特征引导生成
        # cond_embed = self.gconv_cond1(j_1_4, self.adj).reshape(B, J, C)

        # # 条件特征  准的点
        cond_embed= self.gconv_cond(x_c, self.adj).reshape(B, J, C)  # 32 17 128

        time_embed = self.time_mlp(t)[:, None, :].repeat(1, J, 1)

        # c = time_embed + feats + cond_embed
        # c = time_embed + feats
        # c = time_embed +attended_feat+ cond_embed
        c = time_embed + cond_embed
        c = c.reshape(B, J, C)

        for i in range(self.n_layers):
            x = self.atten_layers[i](x, self.src_mask, c)
            x = self.gconv_layers[i](x)

        x = self.gconv_output(x, self.adj)

        x = x.reshape(B, J, -1)
        # 是否融合
        # x_t = x_t.reshape(B, J, -1)
        # x = self.fusion(x, x_t)
        return x


