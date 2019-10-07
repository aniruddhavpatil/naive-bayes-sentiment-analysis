import sys
import os
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt

# Make dataset paths
dataset = {}
dataset_folder = '.'
curr_dataset = None
text = []
folds = 10
classes = 2
dataset['amazon'] = os.path.join(dataset_folder, 'amazon_cells_labelled.txt')
dataset['imdb'] = os.path.join(dataset_folder, 'imdb_labelled.txt')
dataset['yelp'] = os.path.join(dataset_folder, 'yelp_labelled.txt')

# PUT THE FILENAME OF CUSTOM DATASET BELOW
CUSTOM_DATASET = None 

# UNCOMMENT THE LINE BELOW TO ALSO RUN ON CUSTOM DATASET
# dataset['custom'] = os.path.join(dataset_folder, CUSTOM_DATASET)

def prepare_data(curr_dataset):
	with open(dataset[curr_dataset]) as f:
		text = f.readlines()

	positive_samples = []
	negative_samples = []

	# Process the string to remove special characters
	strip_from_string = ',./;[]`-=<>?:"~!@#$%^&*()_+\'""'
	for i,sentence in enumerate(text):
		tokenized_sentence = sentence.split()
		tokenized_sentence = [y.strip(strip_from_string).lower() for y in tokenized_sentence]
		label = tokenized_sentence[-1]
		if label == '1':
			positive_samples.append(tokenized_sentence)
		else:
			negative_samples.append(tokenized_sentence)

	# Shuffle the samples in each class
	shuffle(positive_samples)
	shuffle(negative_samples)

	class_samples_per_fold = int((len(positive_samples) + len(negative_samples))/ (classes*folds))
	dataset_folds = []
	
	# Make the dataset folds
	for fold_number in range(folds):
		temp_fold = []
		for i in range(class_samples_per_fold):
			temp_fold.append(positive_samples.pop())
			temp_fold.append(negative_samples.pop())

		shuffle(temp_fold)
		dataset_folds.append(temp_fold)

	return dataset_folds


class NaiveBayes:

	def __init__(self, dataset_folds):
		self.dataset_folds = dataset_folds
		self.m = 0
		self.V = None
		self.test_fold = None
		self.accuracies = None
		self.curr_expt = 1


	def test(self):

		accuracies = []
		x = range(1,11)
		if self.curr_expt == 2:
			x = range(1,2)

		# Subset of data loop
		for i in x:

			fold_accuracy = []

			# Fold loop
			for test_fold_idx in range(folds):
				[test_fold] = self.dataset_folds[test_fold_idx:(test_fold_idx+1)]
				train_folds = self.dataset_folds[0:test_fold_idx] + self.dataset_folds[test_fold_idx+1:]
				train_fold = []
				for fold in train_folds:
					# print(fold)
					for x in fold:
						train_fold.append(x)

				train_folds = []

				V = [{}, {}]

				# Build vocabulary
				N = len(train_fold)/10
				sub_train_fold=train_fold[:int(i*N)]
				for item in sub_train_fold:
					words = item[:-1]
					label = int(item[-1])
					for word in words:
						if word not in V[label]:
							V[label][word] = 1

						else:
							V[label][word]+=1

				self.test_fold = test_fold

				self.V = V
				positive_vocab_count = len(self.V[1])
				negative_vocab_count = len(self.V[0])

				num_positive_count = sum(self.V[1].values())
				num_negative_count = sum(self.V[0].values())

				prior_positive = 0.5
				prior_negative  = 1 - prior_positive

				pos_denominator = num_positive_count + self.m * (positive_vocab_count + negative_vocab_count)
				neg_denominator = num_negative_count + self.m * (positive_vocab_count + negative_vocab_count)

				# pos_denominator = num_positive_count + self.m * positive_vocab_count
				# neg_denominator = num_negative_count + self.m * negative_vocab_count

				# Testing the classifier on the given fold setup
				num_correct = 0
				for item in self.test_fold:
					words = item[:-1]
					label = int(item[-1])
					prob_positive = np.log(prior_positive)
					prob_negative = np.log(prior_negative)

					prediction = 0
					for word in words:
						if word in self.V[1]:
							prob_positive+= np.log((self.V[1][word]+self.m)/pos_denominator)
						else:
							if self.m > 0:
								prob_positive+= np.log(self.m/pos_denominator)
							else:
								prob_positive-= np.log(pos_denominator)

						if word in self.V[0]:
							prob_negative+=np.log((self.V[0][word]+self.m)/neg_denominator)
						else:
							if self.m > 0:
								prob_negative+=np.log(self.m/neg_denominator)
							else:
								prob_negative-= np.log(neg_denominator)


					if prob_positive > prob_negative:
						prediction = 1

					if prediction == label:
						num_correct+=1

				fold_accuracy.append(float(num_correct)/len(self.test_fold))

			accuracies.append(fold_accuracy)	
		self.accuracies = accuracies


	# Conducts experiment 1
	def experiment1_results(self):
		plt.clf()
		self.curr_expt=1
		self.m = 0
		self.test()
		fig = plt.figure()
		mu = np.mean(self.accuracies, axis=1)
		s = np.std(self.accuracies, axis=1)
		x = [0.1*x for x in range(1,11)]
		plt.errorbar(x,mu, s, ecolor='red', elinewidth=0.5, capsize=3, label='m=0')

		self.m = 1
		self.test()
		mu = np.mean(self.accuracies, axis=1)
		s = np.std(self.accuracies, axis=1)

		plt.errorbar(x,mu, s, ecolor='green', elinewidth=0.5, capsize=3, label='m=1')
		plt.xticks([0.1*x for x in range(1,11)])
		plt.xlabel('Portion of training data used from the ' + curr_dataset[0].upper() + curr_dataset[1:] + ' dataset' )
		plt.ylabel('Mean of accuracies from 10-F SCV')
		plt.legend()
		plt.savefig('experiment1_'+curr_dataset+'.png')

	# Conducts experiment 2
	def experiment2_results(self):
		plt.clf()
		self.curr_expt=2
		set_of_m = [0.1*x for x in range(0,10)]
		set_of_m_int = [x for x in range(1,11)]
		set_of_m = set_of_m[:] + set_of_m_int[:]
		set_of_mu = []
		set_of_s = []


		for i in set_of_m:
			self.m = i
			self.test()
			set_of_mu.append(np.mean(self.accuracies))
			set_of_s.append(np.std(self.accuracies))

		plt.errorbar(set_of_m,set_of_mu, set_of_s, ecolor='red', elinewidth=0.5, capsize=3)
		plt.xlabel('Values of smoothing(m) parameter \n' + curr_dataset[0].upper() + curr_dataset[1:] + ' dataset')
		plt.ylabel('Mean of accuracies from 10-F SCV')
		plt.savefig('experiment2_'+curr_dataset+'.png')


# Loop through dataset
for ds in dataset:
	curr_dataset = ds
	dataset_folds = prepare_data(ds)

	NB = NaiveBayes(dataset_folds)
	print('Running experiment1 on the ' + curr_dataset[0].upper() + curr_dataset[1:] + ' dataset')
	NB.experiment1_results()
	print('Running experiment2 on the ' + curr_dataset[0].upper() + curr_dataset[1:] + ' dataset')
	NB.experiment2_results()
