#loading libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

# load the dataset
data = fetch_20newsgroups()
#print(data.target_names)

# prepare the dtaset

# define categories
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 
               'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
              'misc.forsale', 'rec.autos', 'rec.motorcycles',
             'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
            'sci.electronics', 'sci.med', 'sci.space',
           'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
          'talk.politics.misc', 'talk.religion.misc']

train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

#print(train.data[5])
#print(len(train.data))

# import model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# train the model
model.fit(train.data, train.target)

# predict
labels = model.predict(test.data)

# create confusion matrix 
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names,
            yticklabels=train.target_names)

plt.xlabel("True Label")
plt.ylabel("Predicted Label")
#plt.show()

# make custom prediction
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


my_custom_text_religion = "Followers of Islam are called Muslims. Muslims are monotheistic and worship one, all-knowing God, who in Arabic is known as Allah. Followers of Islam aim to live a life of complete submission to Allah. ... Islam teaches that Allah's word was revealed to the prophet Muhammad through the angel Gabriel."
my_custom_text_science = "Physics is the natural science that studies matter, its motion and behavior through space and time, and the related entities of energy and force. Physics is one of the most fundamental scientific disciplines, and its main goal is to understand how the universe behaves."


print(predict_category(my_custom_text_religion))
print(predict_category(my_custom_text_science))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# this is simple Naive Bayes Algorithm
# this algorithm intended for Text Classification task
