# Copyright 2022 The Nine Turn Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pytorch based sequential decoder. Designed specially for dynamic graph learning."""


from typing import List, Tuple, Union
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import MultiHeadAttention
from nineturn.dtdg.models.decoder.tf.sequentialDecoder.baseModel import SlidingWindowFamily
from nineturn.core.layers import Time2Vec
from nineturn.core.layers import TSA
from nineturn.dtdg.models.decoder.tf.simpleDecoder import SimpleDecoder
from nineturn.core.types import nt_layers_list


def FTSA(num_heads:int,  input_d:int,embed_dims:List[int], n_nodes:int, window_size: int, time_kernel:int, 
        time_agg:str, simple_decoder: SimpleDecoder, **kwargs):
    model = None
    if time_agg == 'concate':
        model = FTSAConcate(num_heads,input_d,embed_dims,n_nodes, window_size, time_kernel, simple_decoder, **kwargs)
    elif time_agg == 'sum':
        model = FTSASum(num_heads,input_d,embed_dims,n_nodes, window_size, time_kernel, simple_decoder, **kwargs)
    return model



class FTSAConcate(SlidingWindowFamily):
    
    def __init__(self,num_heads:int,  input_d:int,embed_dims:List[int], n_nodes:int, window_size: int, time_kernel:int,
            simple_decoder: SimpleDecoder, **kwargs):
        """Create a sequential decoder.

        Args:
            num_heads: int, number of attention heads
            key_dim: int, dimension of input key
            hidden_d: int, the hidden state's dimension.
            window_size: int, the length of the sliding window
            time_kernel: int, kernel size of the time2vec embedding 
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(input_d, n_nodes, window_size, simple_decoder)
        self.nn_layers = nt_layers_list()
        for emb in embed_dims:
            self.nn_layers.append(TSA(out_dim=emb, num_heads=num_heads, **kwargs))
        self.time2vec = Time2Vec(time_kernel, **kwargs)
        self.time_dimention = tf.range(window_size, dtype=tf.float32)
        self.time_kernel = time_kernel
        self.input_d = input_d

    def build(self, input_shape):
        self.wt = self.add_weight(
            shape=(self.input_d + self.time_kernel , self.input_d),
            initializer='uniform',
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, in_state: Tuple[Tensor, List[int]]):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids = in_state
        ids = ids.numpy()
        node_embs = tf.gather(node_embs, ids)
        self.memory.update_window(node_embs.numpy(),ids)
        input_windows = tf.convert_to_tensor(self.memory.get_memory(ids),dtype=tf.float32)  #N, W, D N serve as batch size in this case
        time_encoding = self.time2vec(self.time_dimention)
        #input_with_time dimension: [N, W, D+T]
        input_with_time = tf.concat([input_windows, [time_encoding for i in
            range(node_embs.shape[0])]], -1) 
        input_with_time = tf.matmul(input_with_time, self.wt) #[N, W, D]
        for layer in self.nn_layers:
            input_with_time = layer(input_with_time,input_with_time)
        last_sequence = tf.slice(input_with_time,
                [0,self.window_size-1, 0],
                [input_with_time.shape[0], 1, input_with_time.shape[2]])
        out = self.simple_decoder((tf.reshape(last_sequence, [-1, input_with_time.shape[2]]), ids))
        return out



class FTSASum(SlidingWindowFamily):
    
    def __init__(self,num_heads:int,  input_d:int,embed_dims:List[int], n_nodes:int, window_size: int, time_kernel:int,
            simple_decoder: SimpleDecoder, **kwargs):
        """Create a sequential decoder.

        Args:
            num_heads: int, number of attention heads
            key_dim: int, dimension of input key
            hidden_d: int, the hidden state's dimension.
            window_size: int, the length of the sliding window
            time_kernel: int, kernel size of the time2vec embedding 
            simple_decoder: SimpleDecoder, the outputing simple decoder.
        """
        super().__init__(input_d, n_nodes, window_size, simple_decoder)
        self.nn_layers = nt_layers_list()
        for emb in embed_dims:
            self.nn_layers.append(TSA(out_dim=emb, num_heads=num_heads, **kwargs))
        self.time2vec = Time2Vec(time_kernel,input_d, **kwargs)
        self.time_dimention = tf.reshape(tf.range(window_size, dtype=tf.float32), [1,-1])
        self.time_kernel = time_kernel
        self.feature_size = input_d


    def build(self, input_shape):
        print(self.time_kernel)
        self.wk = self.add_weight(
            shape=(self.time_kernel,1),
            initializer='uniform',
            trainable=True
        )


    def call(self, in_state: Tuple[Tensor, List[int]]):
        """Forward function."""
        # node_embs: [|V|, |hidden_dim|]
        # sequence length = 1
        # the sequential model processes each node at each step
        # each snapshot the input number of nodes will change, but should only be increasing for V invariant DTDG.
        node_embs, ids = in_state
        ids = ids.numpy()
        node_embs = tf.gather(node_embs, ids)
        self.memory.update_window(node_embs.numpy(),ids)
        input_windows = tf.convert_to_tensor(self.memory.get_memory(ids),dtype=tf.float32)  #N, W, D N serve as batch size in this case
        time_encoding = self.time2vec(self.time_dimention) # 1, W, D, K
        time_encoding = tf.reshape(time_encoding, [-1, self.feature_size, self.time_kernel]) #W D K
        input_with_time = tf.einsum('abc, bch->abch',input_windows ,time_encoding) # N W D K
        input_with_time = tf.transpose(input_with_time, [0,1,3,2]) # N W k D
        input_with_time = tf.reshape(input_with_time, [-1, self.window_size* self.time_kernel, self.feature_size]) 
        # N W*K D
        for layer in self.nn_layers:
            input_with_time = layer(input_with_time,input_with_time)

        input_with_time = tf.reshape(input_with_time, [node_embs.shape[0], self.window_size, self.time_kernel, -1]) 
        input_with_time = tf.tensordot(input_with_time, self.wk, [[2],[0]])
        input_with_time = tf.reshape(input_with_time, [node_embs.shape[0], self.window_size, -1]) 
        last_sequence = tf.slice(input_with_time,
                [0,self.window_size-1, 0],
                [input_with_time.shape[0], 1, input_with_time.shape[2]])
        out = self.simple_decoder((tf.reshape(last_sequence, [-1, input_with_time.shape[2]]), ids))
        return out
