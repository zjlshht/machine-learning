'''贝叶斯分类器'''
from typing import Set,NamedTuple,List,Tuple,Dict,Iterable
import re 
import math
from collections import defaultdict
def tokenize(text:str)->Set[str]: #提取单词，忽略汉字
    text = text.lower()                         #Convert to lowercase
    all_words=re.findall("[a-z0-9']+",text)     #extract the words,and
    return set(all_words)                       #remove duplicates

assert tokenize("Data Science is science")=={"data","science","is"}

class Message(NamedTuple):
    text:str
    is_spam:bool
    
class NaiveBayesClassifier:
    def __init__(self,k:float=0.5) ->None:
        self.k=k #smoothing factor
        
        self.tokens: Set[str]=set()
        self.token_spam_counts:Dict[str,int]=defaultdict(int)
        self.token_ham_counts:Dict[str,int]=defaultdict(int)
        self.spam_messages=self.ham_messages=0
        
'''Next,we'll give it a mothed to train it on a bunch of messages
First,we increment the spam_messages and ham_messages counts. Then we tokenize
each message text,and for each token we increment the token_spam_counts or 
token_ham_counts based on the message type:'''
def train(self,messages:Iterable[Message])->None:
    for message in messages:
        #Increment message counts
        if message.is_spam:
            self.spam_messages +=1
        else:
            self.ham_messages +=1
        #Increment word counts
        for token in tokenize(message.text):
            self.tokens.add(token)
            if message.is_spam:
                self.token_spam_counts[token] +=1
            else:
                self.token_ham_counts[token] +=1
'''Ultimately we'll want to predict P(spam|token).As we saw earlier,to apply
Bayes's theorem we need to know P(token|spam) and P(token|ham) for each token
in the vocabulary.So we'll create a "private" helper function to compute those:'''
def _probabilities(self,token:str)->Tuple[float,float]:
    """returns P(token|spam) and P(token|ham)"""
    spam=self.token_spam_counts[token]
    ham=self.token_ham_counts[token]
    
    p_token_spam=(spam +self.k)/(self.spam_messages + 2*self.k)
    p_token_ham=(ham+self.k)/(self.ham.messages+2*self.k)
    
    return p_token_spam,p_token_ham
'''Finally,we're ready to write our predict method.As mentioned earlier,rather
than multiplying together lots of small probabilities,we'll instead sum up the
log probabilities:'''
def predict(self,text:str)->float:
    text_tokens = tokenize(text)
    log_prob_if_spam=log_prob_if_ham=0.0
    #Iterate through each word in our vocabulart
    for token in self.tokens:
        prob_if_spam,prob_if_ham = self._probilities(token)
        #if *token* appears in the message,
        #add the log probability of seeing it
        if token in text_tokens:
            log_prob_if_spam +=math.log(prob_if_spam)
            log_prob_if_ham +=math.log(prob_if_ham)
            
        #Otherwise add the log probability of _not_ seeing it,
        # which is log(1-probability of seeing it)
    else:
        log_prob_if_spam +=math.log(1.0-prob_if_spam)
        log_prob_if_ham +=math.log(1.0-prob_if_ham)
        prob_if_spam =math.exp(log_prob_if_spam)
        prob_if_ham =math.exp(log_prob_if_ham)
        return prob_if_spam /(prob_if_spam + prob_if_ham)
    
#And now we have a classifier

'''
Testing our model
Testing our model
Testing our model
'''
#Let's make sure our model works by writing some unit tests for it

messages=[Message("spam rules",is_spam=True),
          Message("ham rules",is_spam=False),
          Message("hello ham",is_spam=False)]

model=NaiveBayesClassifier(k=0.5)
#model.train(messages) 这是书上写的 运行不了  包括后面的predict
train(model,messages)
#First,let's check that it got the counts right:
assert model.tokens=={"spam","ham","rules","hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam":1,"rules":1}
assert model.token_ham_counts =={"ham":2,"rules":2,"hello":1} 
'''Now let's make a prediction.We'll also(laboriously) go through our Naive Bayes
logic by hand,and make sure that we get the same result:'''
text = "hello spam"

probs_if_spam=[
    (1+0.5)/(1+2*0.5),#"spam"(present)
    1-(0+0.5)/(1+2*0.5),#"ham"(not present)
    1-(1+0.5)/(1+2*0.5),#"rules"(not present)
    (0+0.5)/(1+2*0.5)#"hello"(present)
    ]

probs_if_ham=[
    (0+0.5)/(2+2*0.5),#"spam"(present)
    1-(2+0.5)/(2+2*0.5),#"ham"(not present)
    1-(1+0.5)/(1+2*0.5),#"rules"(not present)
    (0+0.5)/(2+2*0.5),#"hello"(present)
    ]

p_if_spam=math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham=math.exp(sum(math.log(p) for p in probs_if_ham))

#Should be about 0.83
assert model.predict(text)==p_if_spam/(p_if_ham+p_if_spam)
    
'''
Using our model
Using our model
Using our model
'''

from io import BytesIO #So we can treat bytes as a file 
import requests #To download the files,which
import tarfile #are in .tar.bz format    

#尽管没有用了还是写完吧
BASE_URL="一个网站"
FILES=["20021010_easy_ham.tar.ba2",
       "20021010_hard_ham.tar.bz2",
       "20021010_spam.tar.bz2"]

# This is where the data will end up,
#in/spam,/easy_ham,and/hard_ham subdirectories
#Change this to where you want the data.
OUTPUT_DIR='spam_data'

for filename in FILES:
    #ues requests to get the file contents at each URL
    content = requests.get(f"{BASE_URL}/{filename}").content
    fin=BytesIO(content)
    with tarfile.open(fileobj=fin,mode='r:bz2') as tf:
        tf.extractall(OUTPUT_DIR)
        
    