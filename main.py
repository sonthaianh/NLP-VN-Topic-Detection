import numpy as np
import pandas as pd
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import re

from underthesea import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



##### Lấy stop word trong file
def get_stopwords_list(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))
    
    
### gán các từ stop word vào biến stop_words
stop_words = get_stopwords_list('D:\\Tranning-NLP\\vn_stopwords.txt')


### chương trình xóa các từ stop word trong văn bản
def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stop_words:
            words.append(word)
    return ' '.join(words)


#print(stop_words)

bongda_path = "D:\\Tranning-NLP\\Dataset\\Bongda_*.txt"
giaoduc_path = "D:\\Tranning-NLP\\Dataset\\Giaoduc_*.txt"
phapluat_path = "D:\\Tranning-NLP\\Dataset\\Phapluat_*.txt"

bongda_files = glob.glob(bongda_path)
giaoduc_files = glob.glob(giaoduc_path)
phapluat_files = glob.glob(phapluat_path)

#print (bongda_files, giaoduc_files, phapluat_files)


### Chương trình xử lý văn bản bóng đá
bongda_content=[]
for bd_file in bongda_files:
    output = ''
    content = ''
    bdf = open(bd_file,'r',encoding='utf-8')
    while True:
        content=bdf.readline()
        if not content:
            break
        if content.strip()=='':
            continue
        # word segment
        output = word_tokenize(content, format="text")
        output = output.lower()
        
        # xóa các ký tự không cần thiết
        output = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',output)
        
        # xóa các khoảng trắng
        output = re.sub(r'\s+', ' ', output).strip()
        output = remove_stopwords(output)
        bongda_content.append(output)
    bdf.close()
#print(bongda_content)

### Chương trình xử lý văn bản giáo dục
giaoduc_content=[]
for bd_file in giaoduc_files:
    output = ''
    content=''
    bdf = open(bd_file,'r',encoding='utf-8')
    while True:
        content=bdf.readline()
        if not content:
            break
        if content.strip()=='':
            continue
         # word segment
        output = word_tokenize(content, format="text")
        output = output.lower()
        
        # xóa các ký tự không cần thiết
        output = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',output)
        
        # xóa các khoảng trắng
        output = re.sub(r'\s+', ' ', output).strip()
        output = remove_stopwords(output)
        giaoduc_content.append(output)
    bdf.close()
#print(giaoduc_content)


### Chương trình xử lý văn bản pháp luật
phapluat_content=[]
for bd_file in phapluat_files:
    output = ''
    content=''
    # with open(bd_file, "r", encoding="utf-8") as input:
    #     for string in input:
    bdf = open(bd_file,'r',encoding='utf-8')
    # print('bdf:',bdf)
    while True:
        content=bdf.readline()
        if not content:
            break
        if content.strip()=='':
            continue
        output = word_tokenize(content, format="text")
        output = output.lower()
        
        # xóa các ký tự không cần thiết
        output = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',output)
        
        # xóa các khoảng trắng
        output = re.sub(r'\s+', ' ', output).strip()
        output = remove_stopwords(output)
        phapluat_content.append(output)
    bdf.close()
#print(phapluat_content)

# print(len(bongda_content))
# print(len(giaoduc_content))
# print(len(phapluat_content))

### Ghép các file lại với nhau
#Merge files and add labels
corpus = bongda_content + giaoduc_content + phapluat_content
label = 31 * ["bongda"] + 48 * ["giaoduc"] + 94 * ["phapluat"]
#len(corpus)



#split data for train and test
corpus_train, corpus_test, label_train, label_test = train_test_split(corpus, label, test_size=0.2)
#len(corpus_train)


# building term matrices
vectorizer = CountVectorizer(tokenizer=str.split, stop_words=stop_words)
corpus_train_mat = vectorizer.fit_transform(corpus_train)
corpus_train_mat = corpus_train_mat.toarray()
corpus_test_mat = vectorizer.transform(corpus_test)
corpus_test_mat = corpus_test_mat.toarray()


# building Naive Bayes Classifier
def fit_NBclassifier(trainset, trainlabel):
    nbclassifier = MultinomialNB()
    nbclassifier.fit(trainset, trainlabel)
    return nbclassifier

NB_clf = fit_NBclassifier(corpus_train_mat, label_train) # train the classifier


# predict a label of the documents in the test set using the trained classifier
label_predicted = NB_clf.predict(corpus_test_mat)

#print (label_predicted)

accuracy = accuracy_score(label_test, label_predicted) # accuracy rate of the classifier 

#print (accuracy)

###################################### vẽ mô hình xem dạng biểu đồ #########################################
###### visualize a heat map of confusion matrix to evaluate the quality of the output of the classifier #### 

# conf_mat = confusion_matrix(label_test, label_predicted)
# labels = sorted(set(label_predicted))
# plt.figure()
# plt.title("Heat Map Confusion Matrix")
# plt.imshow(conf_mat, interpolation="nearest", cmap=plt.cm.Reds)
# plt.xticks(np.arange(len(labels)), labels)
# plt.yticks(np.arange(len(labels)), labels)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.colorbar()
# plt.show()



######################### TEST ##########################
test_file = "D:\\Tranning-NLP\\TestCorpus\\testfilebd.txt"
test_content=[]
output = ''
content=''
testf = open(test_file,'r',encoding='utf-8')
while True:
    content=testf.readline()
    if not content:
        break
    if content.strip()=='':
        continue
    output = word_tokenize(content, format="text")
    output = output.lower()
        
        # xóa các ký tự không cần thiết
    output = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',output)
        
        # xóa các khoảng trắng
    output = re.sub(r'\s+', ' ', output).strip()
    output = remove_stopwords(output)
    # print(f'\n out:{output[0]}')
    test_content.append(output)
testf.close()
#print(test_content)

test_mat = vectorizer.transform(test_content)
test_mat = test_mat.toarray()
label_predicted = NB_clf.predict(test_mat)
print(label_predicted)

#display value with highest frequency
values, counts = np.unique(label_predicted, return_counts=True)
values[counts.argmax()]
print(values)

