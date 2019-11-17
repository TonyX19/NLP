import re
import codecs
import jieba
import jieba.analyse
import numpy as np

fr1 = '/Users/hechen/Desktop/1.txt'
fr2 = '/Users/hechen/Desktop/2.txt'

class simhash:
    def __init__(self,content):
        self.simhash=self.simhash(content)

    def __str__(self):
        return str(self.simhash)

    def simhash(self,content):
        #seg = jieba.cut(content)
        #jieba.analyse.set_stop_words('stopword.txt')
        keyWord = jieba.analyse.extract_tags(
            '|'.join(content), topK=10, withWeight=True, allowPOS=())#在这里对jieba的tfidf.py进行了修改
        #将tags = sorted(freq.items(), key=itemgetter(1), reverse=True)修改成tags = sorted(freq.items(), key=itemgetter(1,0), reverse=True)
        #即先按照权重排序，再按照词排序
        keyList = []
        for feature, weight in keyWord:
            weight = int(weight * 10)
            feature = self.string_hash(feature)
            temp = []
            for i in feature:
                if(i == '1'):
                    temp.append(weight)
                else:
                    temp.append(-weight)
            # print(temp)
            keyList.append(temp)
        list1 = np.sum(np.array(keyList), axis=0)
        #print(list1)
        if(keyList==[]): #编码读不出来
            return '00'
        simhash = ''
        for i in list1:
            if(i > 0):
                simhash = simhash + '1'
            else:
                simhash = simhash + '0'
        return simhash

    def similarity(self, other):
        a = float(self.simhash)
        b = float(other.simhash)
        if a > b : return b / a
        else: return a / b

    def string_hash(self,source):
        if source == "":
            return 0
        else:
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            for c in source:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            x = bin(x).replace('0b', '').zfill(64)[-64:]
            #print(source,x)

            return str(x)

    def hammingDis(self, com):
        t1 = '0b' + self.simhash
        t2 = '0b' + com.simhash
        n = int(t1, 2) ^ int(t2, 2)
        i = 0
        while n:
            n &= (n - 1)
            i += 1
        return i


def get_line(fr1,fr2):
    punc = './ <>_ - - = ", 。，？！“”：‘’@#￥% … &×（）——+【】{};；● &～| \s:'
    stoplist = {'是'}
    with open(fr1, encoding='utf-8') as f:
        list1 = f.read()
        string = ''
        X, Y = ['\u4e00', '\u9fa5']
        text1 = re.sub(r'[^\w]+', '', list1)
        # print(text1)
        s = jieba.cut(text1)
        s = [i for i in s if len(i) > 1 and X <= i <= Y and i not in stoplist]
        string = string.join(s)
        line1 = re.sub(r"[{}]+".format(punc), "", string)

    with open(fr2, encoding='utf-8') as f:
        list2 = f.read()
        string = ''
        X, Y = ['\u4e00', '\u9fa5']
        text2 = re.sub(r'[^\w]+', '', list2)
        # print(text2)
        s = jieba.cut(text2)
        s = [i for i in s if len(i) > 1 and X <= i <= Y and i not in stoplist]
        string = string.join(s)
        line2 = re.sub(r"[{}]+".format(punc), "", string)
        hash1 = simhash(line1.split())
        hash2 = simhash(line2.split())
        print(hash1.hammingDis(hash2))
        if hash1.hammingDis(hash2) <= 18:
            print('文本相似')
        else:
            print('文本不相似')
        print(line1)
        print(line2)


if __name__ == '__main__':
    get_line(fr1, fr2)