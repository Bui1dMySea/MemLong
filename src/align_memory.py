import numpy as np
import faiss
import faiss.contrib.torch_utils
import torch
from typing import Literal, Tuple, List, Optional, Union
from .utils import MemCache
from dataclasses import dataclass


"""
记忆与存储机制
** Memory **
- 维护一个固定大小的记忆单元，用于存储历史信息
- 利用 save 模块来更新记忆
- 每次传入的记忆单元是一个 mem_update , 其包含了长度为不固定的 SigMemCache 对象
- save 模块需要完成的操作包括：将SigMemCache对象拆解为文本，检索嵌入，Key-Value-Mask三元组；利用拆解对象，维护一个MemCache用于之后的检索
** Retrieval **
- 利用 retrieve 模块来检索记忆
- 记忆长度分析：如果 bsz 间长度不一致，则用 PAD 填充
"""

@dataclass
class MemRecoder:
    counters:List[List[int]]
    dstore_idx:List[int]
    length:List[List[int]]

class ChunkMemory:
    def __init__(self, model_config, toolkit_config, device=None):
        # llama config
        self.hidden_size = model_config.hidden_size
        self.num_heads = model_config.num_attention_heads
        self.head_dim = int(self.hidden_size / self.num_heads)
        self.memory_size = model_config.memory_size
        self.pooling_tokens = model_config.pooling_tokens
        self.dtype = model_config.mem_dtype
        boundary = model_config.update_boundary
        # toolkit config
        self.task = toolkit_config.task
        self.ret_embeddings_dim = toolkit_config.ret_embeddings_dim
        # other config
        if boundary is not None:
            self.lower, self.upper = list(map(int, boundary.split(",")))
        else:
            self.lower , self.upper = 0 , 100
        assert self.lower < self.upper and self.lower >= 0 and self.upper <= 100
        # init
        self.device = device
        self.mem_caches = MemCache()
        self.mem_caches.texts = []
        self.mem_caches.embeddings = []
        self.res = None
        if device != torch.device("cpu"):
            self.res = faiss.StandardGpuResources()
            
        self.mem_recoder = MemRecoder(
            counters = [],
            dstore_idx = [],
            length = []
        )
        
        self.log_time = 1

    def to(self,device):
        old_device = self.device
        self.device = device
        if self.device != torch.device("cpu"):
            if self.res == None :
                self.res = faiss.StandardGpuResources() 
            
            for i,index in enumerate(self.mem_caches.embeddings):
                print(f"put index from {old_device} to {self.device}")
                if index == torch.device("cpu"):
                    index = faiss.index_cpu_to_gpu(self.res, self.device.index, index)
                else:
                    temp = faiss.index_gpu_to_cpu(index)
                    index = faiss.index_cpu_to_gpu(self.res, self.device.index, temp)
                    
                self.mem_caches.embeddings[i] = index
        
        if self.mem_caches.keys is not None:
            self.mem_caches.keys = self.mem_caches.keys.to(device)
            self.mem_caches.values = self.mem_caches.values.to(device)
            self.mem_caches.masks = self.mem_caches.masks.to(device)
        
    
    def reset(self):
        self.mem_caches = MemCache()
        self.mem_caches.texts = []
        self.mem_caches.embeddings = []
        self.res = None
        if self.device != torch.device("cpu"):
            self.res = faiss.StandardGpuResources()      
        self.mem_recoder = MemRecoder(
            counters = [],
            dstore_idx = [],
            length = []
        )
        
    #DELETE
    def _fill_and_update(self, kv_list) -> List:
        device = self.device
        pooling_list = []
        for i, kv in enumerate(kv_list):
            if kv is None:
                pooling_list.append(None)
                continue
            k = kv["k"]
            v = kv["v"]
            pad_len = 0
            if k.shape[0] % self.mem_granularity != 0:
                pad_len = self.mem_granularity - k.shape[0] % self.mem_granularity
                # fill
                k = torch.cat((k,torch.zeros((pad_len, self.num_heads, self.head_dim),device=device,),),dim=0,)
                v = torch.cat((v,torch.zeros((pad_len, self.num_heads, self.head_dim),device=device,),),dim=0,)

            update_nums = k.shape[0] // self.mem_granularity
            k = k.view(update_nums, self.mem_granularity, self.num_heads, self.head_dim)
            v = v.view(update_nums, self.mem_granularity, self.num_heads, self.head_dim)
            pooling_keys = [torch.zeros((update_nums, self.capacity, self.head_dim),device=device,)for _ in range(self.num_heads)]
            pooling_values = [torch.zeros(
(update_nums, self.capacity, self.head_dim),
                    device=device,
                )
                for _ in range(self.num_heads)
            ]
            for i in range(self.num_heads):
                # TODO: optimize
                k_norm = [
                    torch.mean(
                        k[
                            : update_nums - 1,
                            num * self.pooling_tokens : (num + 1) * self.pooling_tokens,
                            i,
                            :,
                        ],
                        dim=1,
                    )
                    for num in range(self.capacity)
                ]
                k_chg = [
                    (
                        torch.mean(
                            k[
                                update_nums - 1,
                                num
                                * self.pooling_tokens : (num + 1)
                                * self.pooling_tokens,
                                i,
                                :,
                            ],
                            dim=0,
                        )
                        if num != self.capacity
                        else torch.mean(
                            k[
                                update_nums - 1,
                                num
                                * self.pooling_tokens : (num + 1)
                                * self.pooling_tokens
                                - pad_len,
                                i,
                                :,
                            ],
                            dim=0,
                        )
                    )
                    for num in range(self.capacity)
                ]
                v_norm = [
                    torch.mean(
                        v[
                            : update_nums - 1,
                            num * self.pooling_tokens : (num + 1) * self.pooling_tokens,
                            i,
                            :,
                        ],
                        dim=1,
                    )
                    for num in range(self.capacity)
                ]
                v_chg = [
                    (
                        torch.mean(
                            v[
                                update_nums - 1,
                                num
                                * self.pooling_tokens : (num + 1)
                                * self.pooling_tokens,
                                i,
                                :,
                            ],
                            dim=0,
                        )
                        if num != self.capacity
                        else torch.mean(
                            v[
                                update_nums - 1,
                                num
                                * self.pooling_tokens : (num + 1)
                                * self.pooling_tokens
                                - pad_len,
                                i,
                                :,
                            ],
                            dim=0,
                        )
                    )
                    for num in range(self.capacity)
                ]

                merged_k = [
                    torch.cat((item1, item2.unsqueeze(0)), dim=0)
                    for item1, item2 in zip(k_norm, k_chg)
                ]
                merged_v = [
                    torch.cat((item1, item2.unsqueeze(0)), dim=0)
                    for item1, item2 in zip(v_norm, v_chg)
                ]
                pooling_keys[i] = torch.stack(
                    merged_k,
                    dim=1,
                )
                pooling_values[i] = torch.stack(
                    merged_v,
                    dim=1,
                )
                # add
            pooling_list.append(
                {
                    "pooling_keys": torch.stack(pooling_keys, dim=-2),
                    "pooling_values": torch.stack(pooling_values, dim=-2),
                }
            )
        return pooling_list

    def get_key_embeddings(self):
        return self.ret_embeddings

    def achieve_condition(self, condition=5):
        for idx in self.dstore_idx:
            if idx < condition:
                return False
        return True
    
    def get_min_number(self):
        return min(self.mem_recoder.dstore_idx) if self.mem_recoder.dstore_idx != [] else 0
    
    def save(self,mem_update):
        if self.log_time>0:
            print('use mem!!!')
            self.log_time -= 1
        # 1. Text
        texts = mem_update.texts
        # 2. embeddings
        embeddings = mem_update.embeddings
        # 3. key-value-mask
        keys , values, masks = mem_update.keys , mem_update.values , mem_update.masks
        
        if len(keys.shape) != 4 or len(values.shape) != 4 or len(masks.shape) != 4:
            raise ValueError(f"Memory cache content should be consistent in shape got {keys.shape} {values.shape} {masks.shape}")
        
        # No pooling
        if self.pooling_tokens is None:
            # check overflow
            if self.mem_caches.keys is not None and self.mem_caches.keys.shape[-2] + keys.shape[-2] > self.memory_size:
                self.mem_caches.keys = self.mem_caches.keys[...,-self.memory_size//2:,:]
                self.mem_caches.values = self.mem_caches.values[...,-self.memory_size//2:,:]
                self.mem_caches.masks = self.mem_caches.masks[...,-self.memory_size//2:,:]

                for i in range(len(self.mem_caches.keys)):
                    if self.device != torch.device("cpu"):
                        if self.log_time == 0:
                            print("remove index to cpu")
                            self.log_time -= 1
                        temp = faiss.index_gpu_to_cpu(self.mem_caches.embeddings[i])
                    else:
                        if self.log_time == 0:
                            print("remove in cpu")
                            self.log_time -= 1
                        temp = self.mem_caches.embeddings[i]
                    assert temp.ntotal == len(self.mem_caches.texts[i]) == len(self.mem_recoder.length[i]) == self.mem_recoder.dstore_idx[i]
                    rm_indices = torch.arange(0,temp.ntotal//2)
                    rm_total = temp.ntotal // 2
                    temp.remove_ids(rm_indices.cpu().numpy())
                    if self.device != torch.device("cpu"):
                        self.mem_caches.embeddings[i] = faiss.index_cpu_to_gpu(self.res, self.device.index, temp)
                    else:
                        self.mem_caches.embeddings[i] = temp
                    self.mem_caches.texts[i] = self.mem_caches.texts[i][-rm_total:]
                    self.mem_recoder.length[i] = self.mem_recoder.length[i][-rm_total:]
                    self.mem_recoder.dstore_idx[i] = self.mem_recoder.dstore_idx[i]//2

            self.mem_caches.keys = torch.cat((self.mem_caches.keys, keys), dim=-2) if self.mem_caches.keys is not None else keys
            self.mem_caches.values = torch.cat((self.mem_caches.values, values), dim=-2) if self.mem_caches.values is not None else values
            self.mem_caches.masks = torch.cat((self.mem_caches.masks, masks), dim=-2) if self.mem_caches.masks is not None else masks
            
            bsz = len(texts)
            for i in range(bsz):
                if len(self.mem_caches.texts) < i + 1:
                    # init
                    index = faiss.IndexFlatIP(self.ret_embeddings_dim)
                    if self.device != torch.device("cpu"):
                        print(f"put index {i} from cpu to gpu {self.device}")
                        index = faiss.index_cpu_to_gpu(self.res, self.device.index, index)
                    self.mem_caches.embeddings.append(index)
                    self.mem_caches.embeddings[i].add(embeddings[i].view(1,-1).to(torch.float32))
                    self.mem_caches.texts.append(texts[i])
                    self.mem_recoder.length.append([keys.shape[-2]])
                    self.mem_recoder.dstore_idx.append(1)
                else:    
                    self.mem_caches.texts[i].append(texts[i])
                    self.mem_caches.embeddings[i].add(embeddings[i].view(1,-1).to(torch.float32))
                    self.mem_recoder.length[i].append(keys.shape[-2])
                    self.mem_recoder.dstore_idx[i] += 1
                
        else:
            # TODO: Pooling
            pass
    
    def _expand_index_tensor(self, x):
        return torch.cat(list(map(lambda x: torch.arange(x * self.div, (x + 1) * self.div), x)))

    def retrieve_index(self, query_embeddings_list, k):
        return [self.mem_caches.embeddings[i].search(query_embeddings_list[i].to(torch.float32), k) for i in range(len(query_embeddings_list))]    
        
    def expand_elements_2d(self,length_list,indices_list):
        expand_indices = []
        for length , indices in zip(length_list,indices_list):
            indices = indices.tolist()[0]
            tmp = []
            for i in indices:
                if i == 0:
                    tmp.extend(list(range(length[i])))
                else:
                    total = sum(length[:i])
                    tmp.extend(list(range(total,length[i] + total)))
            expand_indices.append((tmp))
        return expand_indices
        
    def get(self,indices_list:Optional[List[torch.Tensor]]) -> MemCache:
        indices_list = [torch.sort(indices,dim=1)[0] for indices in indices_list]
        mem_caches = MemCache()
        expand_indices = self.expand_elements_2d(self.mem_recoder.length,indices_list)
        expand_indices = torch.LongTensor(expand_indices).to(self.device)
        kv_expand_indices = expand_indices.unsqueeze(1).unsqueeze(-1).expand(-1,self.num_heads,-1,self.head_dim)
        mask_expand_indices = expand_indices.unsqueeze(1).unsqueeze(-1)
        mem_caches.keys =  torch.gather(self.mem_caches.keys,2,kv_expand_indices)
        mem_caches.values = torch.gather(self.mem_caches.values,2,kv_expand_indices)
        mem_caches.masks = torch.gather(self.mem_caches.masks,2,mask_expand_indices)
        
        # TODO: ret_counters
        
        return mem_caches
        

