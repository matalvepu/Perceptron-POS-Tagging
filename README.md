# Perceptron-for-POS-Tagging

Author: Md Kamrul Hasan
Email: mhasan8@cs.rochester.edu 
Date: 9/15/2017

===============================================================================================
Description: Implementation of Discriminative training approach (Perceptron) for POS tagging 

I did it as a part of homework problem in the Statistical Speech and Language Processing class 
taught by Prof Daniel Gildea (https://www.cs.rochester.edu/~gildea/) in Spring 2017.

===============================================================================================
Instruction to run:

	python perceptron_final.py train test

	You can change number of iteration in perceptron method by setting n_itr=n.



===============================================================================================

Preprocessing:
	I have preprocessed all training at the beginning to generate all features and associated 
	weights. As a feature, I only emissions and transitions. Then I run perceptron using the 
	viterbi. For every single training instances, I updated the weights acccording to the errors 
	that were genarted by viterbi sequence.  

===============================================================================================

Testing:

Perceptron return the updated weight. Using that weight and viterbi, I have calculated average 
accuarcy for the test set.

Accuracy: 91.75 (After four iteration)

Accuray: 90.2 % (Single iteration)

Accuracy: 91.09 % (Two iterations)

===============================================================================================

In homewrok of HMM Decoder, I got the accuray around 94% where I used smoothing techniques that 
improved my accuracy. I think if I add other features like Start , end states, caps for the 
start word it will improve the accuray more.
