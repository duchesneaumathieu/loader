# coding=utf-8

import os
import string
import json
import numpy as np
from collections import OrderedDict

def tokenize(raw_txt):
    u"""    
    Standard characters: <lowercase letters> <numbers> . ! ? … ( ) - , : ; ' " \n
    tokens: <sequence of lowercase letters> <sequence of number> . ! ? … ( ) - , : ; ' " \n
    steps:
        0) txt0: replace uppercase by lowercase
        1) txt1: replace remaining non-standard characters by space
        2) txt2: add space between tokens
        3) txt3: remove multiple spaces
        4) txt4: replace . . . by …
        5) txt5: replace ux by its ascii conterpart
        6) tokens: list of tokens
    """
    udouble_quotes = u'\u201c\u201d'
    usingle_quotes = u'\u0060\u00b4\u2018\u2019'
    uhyphens = u'\u058a\u05be\u2010\u2011\u2012\u2013\u2014\u2015\u207b\u208b\u2212\u2e3a\u2e3b\ufe58\ufe63\uff0d'
    other_chars = u'.!?…()-,:;\'"\n' + udouble_quotes + usingle_quotes + uhyphens
    valid_chars = string.lowercase + string.digits + other_chars
    valid_chars = set(map(lambda x: unicode(x), valid_chars))
    
    txt0 = raw_txt.lower()
    
    txt1 = list(txt0)
    for i,c in enumerate(txt1):
        if c not in valid_chars: txt1[i] = u' '
    txt1 = u''.join(txt1)
    
    txt2 = list(txt1)
    for i,c in enumerate(txt2):
        if c in other_chars: txt2[i] = u' {0} '.format(c)
    txt2 = u''.join(txt2)

    shift = 1
    for n, (c1, c2) in enumerate(zip(txt2[:-1], txt2[1:])):
        if (c1.isdigit() and c2.isalpha()) or (c1.isalpha() and c2.isdigit()):
            txt2 = txt2[:n+shift] + u' ' + txt2[n+shift:]
            shift += 1
    
    txt3 = txt2
    while '  ' in txt3:
        txt3 = txt3.replace(u'  ', u' ')
    while '\n \n' in txt3:
        txt3 = txt3.replace(u'\n \n', u'\n')
        
    txt4 = txt3.replace(u'. . .', u'…')
    
    txt5 = txt4
    for u in udouble_quotes:
        txt5 = txt5.replace(u, u'"')
    for u in usingle_quotes:
        txt5 = txt5.replace(u, u"'")
    for u in uhyphens:
        txt5 = txt5.replace(u, u'-')

    return [token for token in txt5.split(u' ') if token != u'']

#take all path to files
data_path = os.path.dirname(os.path.realpath(__file__))+'/'
txts_path = 'books_txt_full/'
paths = list()
genres = os.listdir(data_path+txts_path)
for genre in genres:
    files_name = os.listdir('{}{}{}/'.format(data_path, txts_path, genre))
    files_name = [file_name for file_name in files_name if '.txt' in file_name and '-all.txt' not in file_name]
    paths.extend(['{}{}{}/{}'.format(data_path, txts_path, genre, file_name) for file_name in files_name])
    
#remove corrupted
with open(data_path+'bad_files.txt', 'r') as f: bad_files = f.read().split()
paths = [path for path in paths if path not in bad_files]

#sort them (to make sure it is platform independent)
paths = sorted(paths)

#shuffle them
np.random.seed(0xcafe)
paths = np.random.permutation(paths).tolist()

#book_ids
book_ids = OrderedDict([(path, i) for i, path in enumerate(paths)])

#make 10 batches of paths of approximatly the same size (in term of bits)
n_partition = 20
partition_size = np.sum(map(lambda x: os.stat(x).st_size, paths))//n_partition + 1
size = 0
batches = [list() for i in range(n_partition)]
for n, path in enumerate(paths):
    size += os.stat(path).st_size
    n_batch = size//partition_size
    batches[n_batch].append(path)
    
folder_path = data_path + 'tokenized_books/'
for n, batch in enumerate(batches):
    partition = dict()
    for i, path in enumerate(batch):
        with open(path, 'r') as f: raw_txt = f.read().decode("utf-8-sig")
        
        infos = dict()
        infos['tokens'] = tokenize(raw_txt)
        spath = path.split('/')
        infos['path'] = '/'.join(spath[-3:])
        infos['name'] = spath[-1]
        infos['genre'] = spath[-2]
        infos['length'] = len(infos['tokens'])
        infos['id'] = book_ids[path]
        partition[infos['id']] = infos
        
        if i%50==0: print 'Partition {}/{}: {}/{} completed'.format(n, n_partition, i, len(batch))
        
    with open('{}tokenized_books_part_{}.json'.format(folder_path, n), 'w') as f:
        json.dump(partition, f)
