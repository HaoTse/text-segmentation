# Text Segmentation as a Supervised Learning Task (For Traditional Chinese)

This repository contains code and supplementary materials which are required to train and evaluate a model as described in the paper [Text Segmentation as a Supervised Learning Task](https://arxiv.org/abs/1803.09337)

## required resources

- English
    wiki-727K, wiki-50 datasets:
    >  https://www.dropbox.com/sh/k3jh0fjbyr0gw0a/AADzAd9SDTrBnvs1qLCJY5cza?dl=0

    word2vec:
    >  https://drive.google.com/a/audioburst.com/uc?export=download&confirm=zrin&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM

- Chinese
    need pre-traing word2vec yourself

Fill relevant paths in configgenerator.py, and execute the script (git repository includes Choi dataset)

## Creating an environment:

    conda create -n textseg python=2.7 numpy scipy gensim ipython 
    source activate textseg
    pip install http://download.pytorch.org/whl/cu80/torch-0.3.0-cp27-cp27mu-linux_x86_64.whl 
    pip install tqdm pathlib2 segeval tensorboard_logger flask flask_wtf nltk
    pip install pandas xlrd xlsxwriter termcolor
    pip install opencc-python-reimplemented

## How to run training process?

    python run.py --help

Example:

    python run.py --cuda --model max_sentence_embedding --wiki 

## How to evaluate trained model (on wiki-727/choi dataset)?

    python test_accuracy.py  --help

Example:

    python test_accuracy.py --cuda --model <path_to_model> --wiki



## How to create a new wikipedia dataset:
    python wiki_processor.py --input <input> --temp <temp_files_folder> --output <output_folder> --train <ratio> --test <ratio>

Input is the full path to the wikipedia dump, temp is the path to the temporary files folder, and output is the path to the newly generated wikipedia dataset.

Wikipedia dump can be downloaded from following url:
- English
> https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
- Chinese
> https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2

# Adaptation to Chinese
- Sentece tokenizer using regax `r'[。，；]|[!?！？]+'`
- Word tokenizer using jieba
> Can put yourown dictionary in `src/`