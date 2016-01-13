__author__ = 'admin'


import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer

corpusPath = '/Users/admin/Desktop/Corpus CK 12 Textbooks/Cut/'

documents = []

for n in xrange(1, 4075):
    # Open corpus file
    path = corpusPath + str(n) + '.txt'
    text_file = open(path, "r")
    lines = text_file.readlines()
    lines = ' '.join(lines)
    documents.append(unicode(lines, 'utf-8'))

tfidftable = TfidfVectorizer(strip_accents='unicode', stop_words='english', sublinear_tf=True, norm='l2')

termDocumentMatrix = tfidftable.fit_transform(documents)


print termDocumentMatrix.get_shape()

