import pandas as pd
import random
import string
import re
import sys
import numpy as np
from math import exp
import matplotlib.pyplot as plt

# Reads the dataset into a pandas dataframe
print "Enter pathway of sms dataset file :"
p = raw_input()
df=pd.read_csv(p,sep='\t',names=('label','text'))

#shuffle data set
df=df.reindex(np.random.permutation(df.index))
df=df.reset_index(drop = True)

# Divide the dataset into 80-20 ratio where 80% constitute training set and 20% constitute testing set
train_data_df = df[:int((len(df)+1)*.80)]
test_data_df = df[int(len(df)*.80+1):]
test_data_df=test_data_df.reset_index(drop=True)

# initialize parameter,confusion_matrix,performance_metric DataFrame for both training and testing dataset
parameter=[['train',0,0,0,0],['test',0,0,0,0]]
confusion_matrix=[['train',0,0,0,0],['test',0,0,0,0]]
performance_metric=[['train',0,0,0,0,0],['test',0,0,0,0,0]]
df1=pd.DataFrame(parameter,columns=['category','total_testSpam','total_testHam','Post_spam','Post_ham'])
df2=pd.DataFrame(confusion_matrix,columns=['category','true_positive','true_negative','false_positive','false_negative'])
df3=pd.DataFrame(performance_metric,columns=['category','alpha','accuracy','f_score','precision','recall'])
ham_total_words,spam_total_words=0,0
Prior_Spam,Prior_ham,Prob_spam,Prob_ham=0,0,0,0
N,x_axis,y1,y2,y3,y4=20000,[],[],[],[],[]
spam_word,nspam_word={},{}

'''
 train() function calculates parameter viz total spam words ,total ham words ,total spam messages and so on for
 80% dataset and stores it in python dictionary
'''
def train():
	l_ham_total_words,l_spam_total_words,l_spam_total_msg,l_ham_total_msg=0,0,0,0
	s=''
	for i in range(0,len(train_data_df)):
		if train_data_df['label'][i]=='spam':
			l_spam_total_msg+=1
			s=" ".join(re.findall("[a-zA-Z']+",train_data_df['text'][i]))
			for word in s.split(' '):
				spam_word[word]=spam_word.get(word,0)+1
				l_spam_total_words+=1
		elif train_data_df['label'][i]=='ham':
			l_ham_total_msg+=1
			s=" ".join(re.findall("[a-zA-Z']+",train_data_df['text'][i]))
	    	for word in s.split(' '):
	    		nspam_word[word]=nspam_word.get(word,0)+1
	    		l_ham_total_words+=1
	global ham_total_words,spam_total_words
	ham_total_words,spam_total_words=l_ham_total_words,l_spam_total_words
	global Prior_Spam,Prior_ham
	Prior_Spam=float(l_spam_total_msg)/float(len(train_data_df))
	Prior_ham =float(l_ham_total_msg)/float(len(train_data_df))

'''
accuracy() takes in a dataset (training or testing) and calculates its accuracy and confusion matrix .
'''

def accuracy(data_set,alpha,category):
	l_total_testSpam,l_total_testHam,l_Post_spam,l_Post_ham=0,0,0,0
	true_positive,true_negative,false_positive,false_negative=0,0,0,0
	for i in range(0,len(data_set)):
		if train_data_df['label'][i]== 'spam':
			l_total_testSpam+=1
		elif train_data_df['label'][i]== 'ham':
			l_total_testHam+=1
		global Prior_Spam,Prior_ham
		l_Post_spam=Prior_Spam*Likelihood(data_set['text'][i],alpha,True)
		l_Post_ham=Prior_ham*Likelihood(data_set['text'][i],alpha,False)
		if (l_Post_spam > l_Post_ham):
			if data_set['label'][i] == 'spam':
				true_positive+=1
			else:
				false_positive+=1
		else:
			if data_set['label'][i] == 'ham':
				true_negative+=1
			else:
				false_negative+=1
    #update the confusion matrix for each category viz test and train
	df1.loc[df2['category']==category,'total_testSpam']=l_total_testSpam
	df1.loc[df2['category']==category,'total_testHam']=l_total_testHam
	df2.loc[df2['category']==category,'true_positive']=true_positive
	df2.loc[df2['category']==category,'true_negative']=true_negative
	df2.loc[df2['category']==category,'false_positive']=false_positive
	df2.loc[df2['category']==category,'false_negative']=false_negative
	df3.loc[df2['category']==category,'accuracy']= float(true_positive+true_negative)/float(l_total_testSpam+l_total_testHam)
	df3.loc[df2['category']==category,'precision']=float(true_positive)/float(true_positive+false_positive)
	precision=float(true_positive)/float(true_positive+false_positive)
	df3.loc[df2['category']==category,'recall']=float(true_positive)/float(true_positive+false_negative)
	recall=float(true_positive)/float(true_positive+false_negative)
	df3.loc[df2['category']==category,'f_score']=2*(precision*recall)/(precision+recall)
	df3.loc[df2['category']==category,'alpha']=alpha


'''
Likelihood() takes in single message and calculates the Likelihood of the message given spam or ham class.
'''
def Likelihood(message,alpha,label):
    prob_spam,prob_nspam=1,1
    s=" ".join(re.findall("[a-zA-Z]+",message))
    global spam_total_words,ham_total_words,N
    if label:
        for word in s.split(' '):
            prob_spam*=float(spam_word.get(word,0)+alpha)/float(spam_total_words+N*alpha)
        return prob_spam
    else:
		for word in s.split(' '):
			prob_nspam*=float(nspam_word.get(word,0)+alpha)/float(ham_total_words+N*alpha)
		return prob_nspam

# calculate the accuracy given alpha as 0.1 for both training and testing dataset
train()
accuracy(test_data_df,0.1,'test')
accuracy(train_data_df,0.1,'train')
print(df3.to_string(index=False))

'''
for loop calculates accuracy for value of alpha
accuracy and fscore of each value of alpha (both training and testing dataset) is stored
in a list which is used to plot y_axis of the graph.
'''
for alpha in list(map(lambda x:2**x,range(-5,1,1))):
	train()
	accuracy(test_data_df,alpha,'test')
	accuracy(train_data_df,alpha,'train')
	y1.append(df3['accuracy'][0])
	y2.append(df3['accuracy'][1])
	y3.append(df3['f_score'][0])
	y4.append(df3['f_score'][1])
for i in range(-5,1,1):x_axis.append(i)

# graph for accuracy versus different value of alpha
plt.title('Accuracy Measure')
plt.xlabel('Values of i',fontsize=15)
plt.ylabel('accuracy',fontsize=15)
plt.plot(x_axis, y1 ,label='Training_Accuracy')
plt.plot(x_axis, y2,label='Testing Accuracy' )
legend = plt.legend(loc='upper right', shadow=True)
plt.show()

# graph for F-score versus different value of alpha
plt.title('F-score')
plt.xlabel('Values of i',fontsize=15)
plt.ylabel('accuracy',fontsize=15)
plt.plot(x_axis, y3,label='Training f_score')
plt.plot(x_axis, y4,label='Testing f_score' )
legend = plt.legend(loc='upper right', shadow=True)
plt.show()
