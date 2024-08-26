from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
import torch
from typing import List, Literal
from faiss import normalize_L2

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    },
    "chat": {
        "query": "Embed this dialogue to find useful historical dialogues: ",
        "key": "Embed this historical dialogue for retrieval: ",
    },
    "lrlm": {
        "query": "Embed this text chunk for finding useful historical chunks: ",
        "key": "Embed this historical text chunk for retrieval: ",
    },
    "tool": {
        "query": "Transform this user request for fetching helpful tool descriptions: ",
        "key": "Transform this tool description for retrieval: ",
    },
    "convsearch": {
        "query": "Encode this query and context for searching relevant passages: ",
        "key": "Encode this passage for retrieval: ",
    },
}


class llm_embedder:
    def __init__(self, toolkit_config,device):
        self.embedder_model = AutoModel.from_pretrained(toolkit_config.embedder_path).to(device)
        self.embedder_tokenizer = AutoTokenizer.from_pretrained(toolkit_config.embedder_path)
        self.instruction = INSTRUCTIONS[toolkit_config.task]
    
    def to(self,device):
        print(f"Move Retriever from {self.embedder_model.device} to {device}")
        self.embedder_model.to(device)

    def get_embeddings(self, examples: List[str], mode: Literal["query", "key"]):
        examples = [self.instruction[mode] + example for example in examples]
        inputs = self.embedder_tokenizer(examples, padding=True, truncation=True, return_tensors="pt")
        inputs = {name: tensor.to(self.embedder_model.device) for name, tensor in inputs.items()}
        with torch.no_grad():
            outputs = self.embedder_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            # normalize_L2(embeddings)
        return embeddings

    def get_max_seq_length(self):
        return 512

class st_embedder:
    def __init__(self, toolkit_config):
        self.embedder_model = SentenceTransformer(toolkit_config.embedder_path, device=toolkit_config.device)
        self.max_seq_length = self.embedder_model.max_seq_length

    def to(self,device):
        print(f"Move Retriever from {self.embedder_model.device} to {device}")
        self.embedder_model.to(device)
    
    def get_embeddings(self, examples: List[str], mode: Literal["query", "key"]):
        embeddings = self.embedder_model.encode(examples,convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False,)
        # normalize_L2(embeddings)
        return embeddings

    def get_max_seq_length(self):
        return self.max_seq_length

class bge_embedder:
    def __init__(self,toolkit_config,device) -> None:
        self.embedder_model = BGEM3FlagModel(toolkit_config.embedder_path, use_fp16=True,device=device)   
        self.max_seq_length = 8192
    
    def to(self,device):
        print(f"Move Retriever from {self.embedder_model.device} to {device}")
        self.embedder_model.device = device
        self.embedder_model.model.to(device)
    
    def get_embeddings(self, examples: List[str], mode=None):
        embeddings = self.embedder_model.encode(examples,max_length=self.max_seq_length)['dense_vecs']
        return torch.from_numpy(embeddings).to(self.embedder_model.device)
    
    def get_max_seq_length(self):
        return self.max_seq_length
        
