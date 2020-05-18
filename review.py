from graphviz import Graph
from collections import Counter
from typing import Dict, List, Union
import itertools
import csv
import spacy


def read_reviews(review_file: str) -> Dict[str, Union[str, int]]:
    with open(review_file, 'rt') as fh:
        reader = csv.reader(fh, delimiter='\t', quotechar='\xe7')
        headers = next(reader)
        for row in reader:
            try:
                row_json = {header: row[hi] for hi, header in enumerate(headers)}
                yield row_json['review_text']
            except IndexError:
                print(headers)
                print(row)
                raise


def make_content_ngrams(doc, ngram_size: int, use_lemma: bool = False) -> List[str]:
    ngrams = []
    for sent in doc.sents:
        tokens = [token for token in sent]
        for ti, token in enumerate(tokens[:-ngram_size+1]):
            token_ngram = tokens[ti:ti+ngram_size]
            for token in token_ngram:
                if token.is_punct or token.is_stop or not token.is_alpha:
                    break
            else:
                if use_lemma:
                    ngrams.append(' '.join([token.lemma_.lower() for token in token_ngram]))
                else:
                    ngrams.append(' '.join([token.lower_ for token in token_ngram]))
    return ngrams


def count_content_ngrams(docs, ngram_size: int, use_lemma: bool = False):
    #bigram_count = Counter()
    ngram_count = Counter()
    for doc in docs:
        ngrams = make_content_ngrams(doc, ngram_size, use_lemma=use_lemma)
        ngram_count.update(ngrams)
    return ngram_count


def write_ngrams(review_file, ngram_word_file, ngram_lemma_file):
    fh_word = open(ngram_word_file, "wt")
    fh_lemma = open(ngram_lemma_file, "wt")
    word_writer = csv.writer(fh_word, delimiter='\t')
    lemma_writer = csv.writer(fh_lemma, delimiter='\t')
    for ri, review in enumerate(read_reviews(review_file)):
        doc = nlp(review)
        word_ngrams = make_content_ngrams(doc, ngram_size = 2, use_lemma=False)
        word_writer.writerow(word_ngrams)
        lemma_ngrams = make_content_ngrams(doc, ngram_size = 2, use_lemma=True)
        lemma_writer.writerow(lemma_ngrams)
        if (ri+1) % 1000 == 0:
            print(ri+1, 'ngram sets written')
    print(ri+1, 'ngram sets written')


def draw_cooccurrence_graph(cooc):
    dot = Graph(comment='Cooccurrence of common bi-grams')
    ngrams = list(set([ngram for pair, freq in cooc.most_common(50) for ngram in pair]))
    for index, ngram in enumerate(ngrams):
        dot.node(str(index), ngram)

    for pair, freq in cooc.most_common(50):
        ngram1, ngram2 = pair[0], pair[1]
        dot.edge(str(ngrams.index(ngram1)), str(ngrams.index(ngram2)), label="{}".format(freq))
    return dot


def get_ngram_cooccurrence(spacy_docs, ngram_size: int = 2, use_lemma: bool = False,
                           num_ngrams: int = 50, ignore_list: List[str] = []):
    ngram_sets = [list(set(make_content_ngrams(spacy_doc, ngram_size=ngram_size, use_lemma=use_lemma))) for spacy_doc in spacy_docs]
    ngram_freq = Counter([ngram for ngram_set in ngram_sets for ngram in ngram_set])
    common_ngrams = [ngram for ngram, _freq in ngram_freq.most_common(num_ngrams) if ngram not in ignore_list]
    ngram_cooc = get_cooccurrence(ngram_sets, common_ngrams)
    return ngram_cooc

def get_cooccurrence(ngram_sets, common_ngrams) -> Counter:
    cooc = Counter()
    for di, ngram_set in enumerate(ngram_sets):
        common = sorted([ngram for ngram in ngram_set if ngram in common_ngrams])
        if len(common) > 1:
            cooc.update([(ngram1, ngram2) for ngram1, ngram2 in itertools.combinations(common, 2) if ngram1 != ngram2])
    return cooc


review_files = [
    'reviews/reviews.Books.100-items.1000-reviews.100-characters.csv', # 100 books, 1000 reviews each
    'reviews/reviews.Books.10000-items.10-reviews.100-characters.csv', # 10000 books, 10 review each
    'reviews/reviews.Books.100000-items.1-reviews.100-characters.csv', # 100000 books, 1 review each
    'reviews/reviews_Gone_Girl.csv', # 19871 reviews of a single book
    'reviews/reviews_goodreads.books_with_100_reviews.csv'
]

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")


if __name__ == '__mainn__':
    for review_file in review_files:
        ngram_word_file = review_file.replace('csv', 'ngrams.word.csv')
        ngram_lemma_file = review_file.replace('csv', 'ngrams.lemma.csv')
        write_ngrams(review_file, ngram_word_file, ngram_lemma_file)



