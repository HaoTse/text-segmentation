# coding=utf-8
import re
import os
import jieba
import numpy as np
import wiki_utils
import wiki_thresholds
import utils

missing_stop_words = set([u'的', u'個', u'了'])
# sentence split token
punkt_token = re.compile(ur'[。，；]|[!?！？]+')
# load jieba dictionary
if os.path.isfile('src/dict_t.txt'):
    jieba.load_userdict('src/dict_t.txt')

logger = utils.setup_logger(__name__, 'text_manipulation.log', True )

def chinese_punkt(text):
    return re.split(punkt_token, text)

def chinese_words_tokenizer(text):
    return jieba.lcut(text)

def split_sentence_with_list(sentence):

    list_pattern = "\n" + wiki_utils.get_list_token() + "."
    if sentence.endswith( list_pattern ):
        #splited_sentence = [str for str in sentence.encode('utf-8').split("\n" + wiki_utils.get_list_token() + ".") if len(str) > 0]
        splited_sentence = [str for str in sentence.split("\n" + wiki_utils.get_list_token() + ".") if
                            len(str) > 0]
        splited_sentence.append(wiki_utils.get_list_token() + ".")
        return splited_sentence
    else:
        return [sentence]

def split_sentece_colon_new_line(sentence):

    splited_sentence = sentence.split(":\n")
    if (len(splited_sentence) == 1):
        return splited_sentence
    new_sentences = []
    # -1 . not to add ":" to last sentence
    for i in range(len(splited_sentence) - 1):
        if (len(splited_sentence[i]) > 0):
            new_sentences.append(splited_sentence[i] + ":")
    if (len(splited_sentence[-1]) > 0):
        new_sentences.append(splited_sentence[-1])
    return new_sentences

def split_long_sentences_with_backslash_n(max_words_in_sentence,sentences, doc_id):
    new_sentences = []
    for sentence in sentences:
        sentence_words = extract_sentence_words(sentence)
        if len(sentence_words) > max_words_in_sentence:
            splitted_sentences = sentence.split('\n')
            if len(splitted_sentences) > 1:
                logger.info("Sentence with backslash was splitted. Doc Id: " + str(doc_id) +"   Sentence:  " + sentence)
            new_sentences.extend(splitted_sentences )
        else:
            if "\n" in sentence:
                logger.info("No split for sentence with backslash n. Doc Id: " + str(doc_id) +"   Sentence:  " + sentence)
            new_sentences.append(sentence)
    return new_sentences

def split_sentences(text, doc_id):
    # sentences = get_punkt().tokenize(text)
    sentences = chinese_punkt(text)
    senteces_list_fix = []
    for sentence in sentences:
        seplited_list_sentence = split_sentence_with_list(sentence)
        senteces_list_fix.extend(seplited_list_sentence)

    sentence_colon_fix = []
    for sentence in senteces_list_fix:
        splitted_colon_sentence =  split_sentece_colon_new_line(sentence)
        sentence_colon_fix.extend(splitted_colon_sentence)

    sentences_without_backslash_n = split_long_sentences_with_backslash_n(wiki_thresholds.max_words_in_sentence_with_backslash_n, sentence_colon_fix, doc_id)

    ret_sentences = []
    for sentence in sentences_without_backslash_n:
        ret_sentences.append(sentence.replace('\n',' '))


    return ret_sentences

def extract_sentence_words(sentence, remove_missing_emb_words = False,remove_special_tokens = False):
    if (remove_special_tokens):
        for token in wiki_utils.get_special_tokens():
            # Can't do on sentence words because tokenizer delete '***' of tokens.
            sentence = sentence.replace(token, "")
    # tokenizer = get_words_tokenizer()
    # sentence_words = tokenizer.tokenize(sentence)
    sentence_words = chinese_words_tokenizer(sentence)
    if remove_missing_emb_words:
        sentence_words = [w for w in sentence_words if w not in missing_stop_words]

    return sentence_words


def word_model(word, model):
    if model is None:
        return np.random.randn(1, 300)
    else:
        if word in model:
            return model[word].reshape(1, 300)
        else:
            #print ('Word missing w2v: ' + word)
            return model['UNK'].reshape(1, 300)

