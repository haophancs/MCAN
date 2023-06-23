Deep Modular Co-Attention Network for VQA
----

This repository follows the paper [Deep Modular Co-Attention Networks for Visual Question Answering](https://arxiv.org/pdf/1906.10770.pdf) with modification to train on the [VQA dataset]() for VQA task in Vietnamese.

To reproduce the results on the VQA dataset, first you need to get the dataset as follows:
```
git clone https://github.com/haophancs/MCAN.git
cd MCAN && mkdir -p ../datasets ./saved_models
gdown 1Zvd8hi-RjE8Bdvd9kJi6JZ0yOK18fJsz
unzip -q viclevr.zip -d ../datasets/viclevr && rm viclevr.zip
python3 preprocess-images.py
```

Train the MCAN method with the following command:
```
python3 train.py
```

We especially design this method to train with Vietnamese pretrained word-embedding. To use pretrained word-embedding, open `config.py` then set word_embedding to the pretrained word-embedding you want:
```
"fasttext.vi.300d"
"phow2v.syllable.100d"
"phow2v.syllable.300d"
"phow2v.word.100d"
"phow2v.word.300d"
```