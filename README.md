<h1 align="center">Welcome to MemLong 👋</h1>
<p>
</p>


> MemLong is a method that utilizes explicit retrievers to extend the context length of language models. It is compatible with any current Decoder-Only architecture model and requires only a small amount of fine-tuning data to achieve ultra-long length extension. 

### 🏠 [Homepage](https://github.com/Bui1dMySea/MemLong) 📄

<div align='center'>
     <p>
        <a href='https://arxiv.org/abs/2408.16967'><img src='https://img.shields.io/badge/arXiv-2408.16967-red'></a>
        <img src='https://img.shields.io/github/stars/Bui1dMySea/MemLong?style=social&color=green
' />
        <img src="https://img.shields.io/badge/python->=3.9.11-blue">
        <a href="https://pypi.org/project/lightrag-hku/"><img src="https://img.shields.io/pypi/v/lightrag-hku.svg"></a>
    </p>



![MemLong](./asset/illustration.png)

# What's MemLong?

**Chunking**: For sequences of arbitrary length, we chunk them into fixed lengths (in our experiments, we used lengths of 256 and 512  ). 
**Memory and Retrieval**: For the retrieval method, we innovatively proposed using an external retriever to search for the current chunk. The benefit of this approach is that it leverages the powerful retrieval capabilities of current models like bge-m3. For memory, we introduced dynamic memory planning. Specifically, our strategy differs from conventional FIFO (First In, First Out) by using a counter to calculate the trigger frequency of each chunk. When the memory length is exceeded, we prioritize deleting chunks with lower trigger frequencies until the required number of deletions is met. **Positional Encoding and Memory Fusion**: Our experiments found that if we reassign positional information for retrieved chunks and the current chunk, such as rearranging the original positional information from $c_k=(n_1,n_2,...,n_k)$ to $(r_1,r_2,...,r_n,c_k)$, it can lead to catastrophic information collapse. Therefore, we simply set the positional information of retrieved chunks to 0. Improving positional encoding will be addressed in future work. At the model's upper layer, we modified the attention mechanism so that the current query can access the retrieved chunks. 
**Efficient Training**: During the fine-tuning phase, we only fine-tune the layers above the memory layer of the model. The advantage of this approach is that it significantly reduces the number of parameters to be fine-tuned compared to full fine-tuning, saving a lot of GPU memory. This is also why we only need a small amount of data.      

# Quick Start

## Environment

1. `conda create -n MemLong python=3.10`
2. `conda activate MemLong`
3. `pip install -r requirements.txt`

## Data Processing

In the paper, we used [slimpajama](https://huggingface.co/datasets/yaofu/slimpajama-per-source-length-upsample) as our training dataset. We strongly recommend that you download to the ./data folder. For Chinese users, we recommend using [HF-mirror](https://hf-mirror.com) for downloading. Here are some specific steps for downloading.

### Downloading Procedure (Recommend)

1. Only Need for Chinese User: `export HF_ENDPOINT=https://hf-mirror.com` 
2. `pip install -U huggingface_hub`
3. `wget https://hf-mirror.com/hfd/hfd.sh`
4. `chmod a+x hfd.sh`
5. `sudo apt-get install aria2c`
6. `cd ./data`
7. `./hfd.sh yaofu/slimpajama-per-source-length-upsample --dataset --tool aria2c -x 4`

### Pre-processing Dataset

```bash
cd ./data
bash process.sh
```

## Training

### Stage 1: Warm Up (Optional)

In order to make the model adapt to the MemLong in advance, we need to do a warm up. 

You can easily train a version of the LoRA model with no more than 20g of VRAM (single card 3090) and within 35 hours.

We provide the training script for the OpenLLaMA version, and you can easily train using the following command.

```
bash train_stage_1.sh
```

### Stage 2:  MemLong 

The biggest difference from the first step is that we have frozen the underlying parameters and introduced our core idea — the MemLong framework. In the default script we provide, we set 13 layers as memory layers, and define [13, 17, 21, 25] as retrieval layers.

Similarity,we provide the training script for the OpenLLaMA version, and you can easily train using the following command.

```bash
bash train_stage_2.sh
```



## Evaluation

We provide two types of evaluations, including language modeling evaluation and ICL (In Context Learning) evaluation, which you can perform specifically under the eval folder.

For language modeling tasks，you can first `cd eval/language_modeling` and then eval the model or method you want in `bash script/anything.sh` 



## Author

**😀 Weijie Liu**

* Github: [@Bui1dMySea](https://github.com/Bui1dMySea)

😀 **ZetangForward**

- Github: [@ZetangForward](https://github.com/ZetangForward)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!<br />Feel free to check [issues page](https://github.com/Bui1dMySea/MemLong/issues). 

## Show your support

Give a ⭐️ if this project helped you!