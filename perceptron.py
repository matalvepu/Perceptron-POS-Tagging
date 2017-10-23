#Author: Md Kamrul Hasan
import numpy as np
import random
import sys

#This method generates all features map and associated weight map
#I have created feature only for emission and transition 
#Weights are initialized with uniform random value
def pre_process_data(f_train):
	phi={}
	weight={}
	train=[]
	states=set()
	with open(f_train) as f:
	    for line in f:
	    	line=line.strip()
	        line = line.split(' ')
	        line=line[1:]
	        train.append(line)
	        word=line[0::2]
	        tag=line[1::2]
	        prev_tag=None
	        for i in range(len(tag)):
	        	if tag[i] in phi:
	        		if word[i] not in phi[tag[i]]:
	        			phi[tag[i]][word[i]]=1
	        			weight[tag[i]][word[i]]=random.uniform(-1, 1)
	        	else:
	        		phi[tag[i]]={word[i]:1}
	        		weight[tag[i]]={word[i]:random.uniform(-1, 1)}

	        	if prev_tag:
	        		if prev_tag in phi:
	        			if tag[i] not in phi[prev_tag]:
	        				phi[prev_tag][tag[i]]=1
	        				weight[prev_tag][tag[i]]=random.uniform(-1, 1)
	        		else:
	        			phi[prev_tag]={tag[i]:1}
	        			weight[prev_tag]={tag[i]:random.uniform(-1, 1)}

	        	prev_tag=tag[i]
	        	states.add(tag[i])   	
	return phi,weight,list(states),train

#if this feature (x_i,y_i) is present in feature set then return associated weight
def get_weight_cross_phi(weight,phi,x_i,y_i):
	if y_i in weight:
		if x_i in weight[y_i]:
			return weight[y_i][x_i]
	return 0

#decode sentence and return tag sequence
#delta is N+1*T size table
#backpointer table for back track the tag sequence
def viterbi(phi,weight,states,x):
	delta=np.zeros((len(x)+1,len(states)))
	back_pointer=np.zeros((len(x)+1,len(states)))
	for i in range(1,len(x)+1):
		for t in range(len(states)):
			max_val=-np.inf
			max_back_pointer=None
			for t1 in range(len(states)):
				temp= delta[i-1][t1]+ get_weight_cross_phi(weight,phi,x[i-1],states[t])
				if i<len(x):
					temp+=get_weight_cross_phi(weight,phi,states[t],states[t1])
				if temp>max_val:
					max_val=temp
					max_back_pointer=t1

			delta[i][t]=max_val
			back_pointer[i][t]=max_back_pointer

	#this part backtrack and generate best tag sequence
	last_row=delta[len(x):,]
	s=np.argmax(last_row)
	hidden_seq=[]

	for i in range(len(x),0,-1):
		hidden_seq.append(states[s])
		s=int(back_pointer[i][s])

	return hidden_seq[::-1]	

#update weight 
#y=true tag sequences
#v=predicted tag sequences
def update_weight(weight,x,y,v):

	for i in range(len(x)):

		weight[y[i]][x[i]]+=1		
		if v[i] in weight:
			if x[i] in weight[v[i]]:
				weight[v[i]][x[i]]-=1

		if i==len(x)-1:
			break

		weight[y[i]][y[i+1]]+=1		
		if v[i] in weight:
			if v[i+1] in weight[v[i]]:
				weight[v[i]][v[i+1]]-=1

	return weight

#perceptron algorithm
def perceptron(phi,states,weight,train):
	n_itr=2
	for i in range(n_itr):
		for s in train:
			x=s[0::2]
			y=s[1::2]
			v=viterbi(phi,weight,states,x)
			if y!=v:
				weight=update_weight(weight,x,y,v)
			
	return weight

#calculate relative accuracy
def get_sentence_accuracy(y,v):
	c=0.
	for i in range(len(y)):
		if y[i]==v[i]:
			c+=1
	return c/(len(y))

#It read all test sentences and do decoding and generate accuracy
def test_accuracy(phi,weight,states,f_test):
	sen_count=0.
	accuracy_sum=0.
	with open(f_test) as f:
	    for line in f:
	    	if not line:
	    		continue
	    	line=line.strip()
	        line = line.split()
	        x=line[1::2]
	        y=line[2::2]
	        v=viterbi(phi,weight,states,x)
	        acc=get_sentence_accuracy(y,v)
	        print("acc:",acc)
	        accuracy_sum+=acc
	        sen_count+=1

	return accuracy_sum/sen_count

def main():
	
	f_train=sys.argv[1]
	f_test=sys.argv[2]

	phi,weight,states,train=pre_process_data(f_train)

	weight=perceptron(phi,states,weight,train)

	acc=test_accuracy(phi,weight,states,f_test)
	print("accuracy",acc)

if __name__ == '__main__':
    main()