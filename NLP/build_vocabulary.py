import json
import sys

assert len(sys.argv) == 2 and sys.argv[1].isdigit(), 'Error: need word occurrence threshold.'

th = int(sys.argv[1])

n_files = 14
file_paths = ['tokenized_books/tokenized_books_part_{}.json'.format(i) for i in range(n_files)]
count = dict()
for n, file_path in enumerate(file_paths):
    print 'File {}/{} in progress...'.format(n+1, n_files)
    with open(file_path, 'r') as f: data = json.load(f)
    for books in data.values():
        for token in books['tokens']:
            if token in count: count[token] += 1
            else: count[token] = 1
                
with open('token_occurrence.txt', 'w') as f:
    for k, v in sorted(count.items(), key=lambda x: x[1], reverse=True):
        if k == '\n': k = '\\n'
        f.write(u'{} {}\n'.format(k, v).encode('utf8'))
            
vocabulary = sorted(filter(lambda x: count[x] >= th, count.keys()))
with open('vocabulary_{}.txt'.format(th), 'w') as f:
    f.write(u' '.join(vocabulary).encode('utf8'))

with open('vocabulary_{}_stats.txt'.format(th), 'w') as f:
    f.write('Word occurrence threshold: {}\n'.format(th))
    f.write('Vocabulary size: {}\n'.format(len(vocabulary)))
    f.write('Proportion of seen token that are in the vocabulary: {:.4f}\n'.format(len(vocabulary)/float(len(count))))
    n_words = sum(count.values())
    vocab_occ = sum([v for k, v in count.iteritems() if k in vocabulary])
    f.write('Proportion of word in the dataset that are in the vocabulary: {:.4f}\n\n'.format(vocab_occ/float(n_words)))

    th_occ = len(filter(lambda x: x==th-1, count.values()))
    f.write('Ratio of words with {} occurrences: {:.6f}\n'.format(th-1, th_occ/float(n_words)))

    th_occ = len(filter(lambda x: x==th, count.values()))
    f.write('Ratio of words with {} occurrences: {:.6f}\n'.format(th, th_occ/float(n_words)))
    
    th_occ = len(filter(lambda x: x==th+1, count.values()))
    f.write('Ratio of words with {} occurrences: {:.6f}\n'.format(th+1, th_occ/float(n_words)))
