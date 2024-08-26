from .ret_embedder import llm_embedder, st_embedder,bge_embedder
from .align_memory import ChunkMemory
from typing import List
import torch


class ToolKit:
    def __init__(self, model_config, toolkit_config,device=None):
        self.device = device
        embedder_name = toolkit_config.embedder_name
        if embedder_name == "llm_embedder":
            self.embedder = llm_embedder(toolkit_config=toolkit_config,device=device)
        elif embedder_name == "st_embedder":
            self.embedder = st_embedder(toolkit_config=toolkit_config,device=device)
        elif embedder_name =="bge_embedder":
            self.embedder = bge_embedder(toolkit_config=toolkit_config,device=device)
        else:
            raise NotImplementedError
        self.max_ret_length = self.embedder.get_max_seq_length()
        self.tokenizer = None
        self.chunk_memory = ChunkMemory(model_config=model_config, toolkit_config=toolkit_config,device=device)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        
    def to(self, device):
        self.device = device
        self.embedder.to(device)
        self.chunk_memory.to(device)
    
    def update(self, mem_update):
        bsz_texts = mem_update.texts
        embeddings = []
        for i, texts in enumerate(bsz_texts):
            if texts is None:
                embeddings.append(None)
            else:
                embeddings.append(self.embedder.get_embeddings(examples=texts, mode="key"))
        mem_update.embeddings = embeddings
        self.chunk_memory.save(mem_update)
    
    def get_texts(self , input_ids):
        if self.tokenizer == None:
            raise ValueError("Before using MemLong, you should set tokenizer by calling set_toolkit_tokenizer(tokenizer)")
        
        texts_list = [[] for _ in range(input_ids.shape[0])]
        for i in range(input_ids.shape[0]):
            texts_list[i].append(self.tokenizer.decode(input_ids[i], skip_special_tokens=True))
        return texts_list
        
    # 1. 如果不满足，则返回mem中所有的cache
    # 2. 如果满足，进行检索

    def reset(self):
        self.chunk_memory.reset()
    
    def retrieve(self, queries: List[str], k: int = 5):
        k = min(k, self.chunk_memory.get_min_number())
        if k == 0:
            return None
        queries_list = [self.embedder.get_embeddings(examples=query, mode="query").to(torch.float32) for query in queries]
        result = self.chunk_memory.retrieve_index(query_embeddings_list=queries_list, k=k)
        scores_list = list(map(lambda x: x[0], result))
        indices_list = list(map(lambda x: x[1], result))
        mem_caches = self.chunk_memory.get(indices_list=indices_list)
        return mem_caches
    
    def get_max_seq_length(self):
        return self.embedder.get_max_seq_length()

    def reset_by_batch(self, batch_indices_to_clear):
        self.chunk_memory.reset_by_batch(batch_indices_to_clear)

    def reset(self):
        self.chunk_memory.reset()
