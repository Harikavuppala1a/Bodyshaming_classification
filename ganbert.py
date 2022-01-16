# IMPORTS

#!pip install transformers==4.3.2
import torch
import io
import torch.nn.functional as F
import random
import numpy as np
import time
import math
import datetime
import torch.nn as nn
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, f1_score, precision_score
#!pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#!pip install sentencepiece

results = []
def get_qc_examples(input_file):
	examples = []

	with open(input_file, 'r') as f:
		contents = f.read()
		file_as_list = contents.splitlines()
		for line in file_as_list[1:]:
			split = line.split(" ")
			question = ' '.join(split[1:])
			text_a = question
			# inn_split = split[0].split(":")
			# label = inn_split[0] + "_" + inn_split[1]
			label = split[0]

			if label != '0' and label != '1':
				label = '1'
			examples.append((text_a, label))
	return examples

#Load the examples


def generate_data_loader(input_examples, label_masks, label_map, do_shuffle = False, balance_label_examples = False, length=64, batch_size = 64):
	examples = []

	num_labeled_examples = 0

	for label_mask in label_masks:
		if label_mask: 
			num_labeled_examples += 1
	label_mask_rate = num_labeled_examples/len(input_examples)


	for index, ex in enumerate(input_examples): 
		if label_mask_rate == 1 or not balance_label_examples:
			examples.append((ex, label_masks[index]))
		else:
			if label_masks[index]:
				balance = int(1/label_mask_rate)
				balance = int(math.log(balance,2))
				if balance < 1:
					balance = 1
				for b in range(0, int(balance)):
					examples.append((ex, label_masks[index]))
			else:
				examples.append((ex, label_masks[index]))

	input_ids = []
	input_mask_array = []
	label_mask_array = []
	label_id_array = []

	model_name = "bert-base-cased"
	max_seq_length = length
	print(length)
	batch_size = batch_size
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	for (text, label_mask) in examples:
		encoded_sent = tokenizer.encode(text[0], add_special_tokens=True, max_length=max_seq_length, padding="max_length", truncation=True)
		input_ids.append(encoded_sent)
		label_id_array.append(label_map[text[1]])
		label_mask_array.append(label_mask)
  
	for sent in input_ids:
		att_mask = [int(token_id > 0) for token_id in sent]                          
		input_mask_array.append(att_mask)


	input_ids = torch.tensor(input_ids) 
	input_mask_array = torch.tensor(input_mask_array)
	label_id_array = torch.tensor(label_id_array, dtype=torch.long)
	label_mask_array = torch.tensor(label_mask_array)
	print(input_ids.shape, input_mask_array.shape, label_id_array.shape, label_mask_array.shape)

	dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

	if do_shuffle:
		sampler = RandomSampler
	else:
		sampler = SequentialSampler

	return DataLoader(
	          dataset,  # The training samples.
	          sampler = sampler(dataset), 
	          batch_size = batch_size,
	          drop_last = True) # Trains with this batch size.

def format_time(elapsed):
	elapsed_rounded = int(round((elapsed)))
	return str(datetime.timedelta(seconds=elapsed_rounded))

class Generator(nn.Module):
	def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
		super(Generator, self).__init__()
		layers = []
		hidden_sizes = [noise_size] + hidden_sizes
		for i in range(len(hidden_sizes)-1):
		    layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

		layers.append(nn.Linear(hidden_sizes[-1],output_size))
		self.layers = nn.Sequential(*layers)

	def forward(self, noise):
		output_rep = self.layers(noise)
		return output_rep

class Discriminator(nn.Module):
	def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
		super(Discriminator, self).__init__()
		self.input_dropout = nn.Dropout(p=dropout_rate)
		layers = []
		hidden_sizes = [input_size] + hidden_sizes
		for i in range(len(hidden_sizes)-1):
		    layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

		self.layers = nn.Sequential(*layers) #per il flatten
		self.logit = nn.Linear(hidden_sizes[-1],num_labels+1) # +1 for the probability of this sample being fake/real.
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, input_rep):
		input_rep = self.input_dropout(input_rep)
		last_rep = self.layers(input_rep)
		logits = self.logit(last_rep)
		probs = self.softmax(logits)
		return last_rep, logits, probs

def main(length, seed, bat):

	global results
		##Set random values
	seed_val = seed
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed_val)


	# If there's a GPU available...
	if torch.cuda.is_available():    
	    # Tell PyTorch to use the GPU.    
	    device = torch.device("cuda")
	    print('There are %d GPU(s) available.' % torch.cuda.device_count())
	    print('We will use the GPU:', torch.cuda.get_device_name(0))
	# If not...
	else:
		print('No GPU available, using the CPU instead.')
		device = torch.device("cpu")



	max_seq_length = length
	batch_size = bat

	num_hidden_layers_g = 1; 
	num_hidden_layers_d = 1; 
	noise_size = 100
	out_dropout_rate = 0.2

	# Replicate labeled data to balance poorly represented datasets, 
	# e.g., less than 1% of labeled material
	apply_balance = True

	#--------------------------------
	#  Optimization parameters
	#--------------------------------
	learning_rate_discriminator = 5e-5
	learning_rate_generator = 5e-5
	epsilon = 1e-8
	num_train_epochs = 10
	multi_gpu = True
	# Scheduler
	apply_scheduler = False
	warmup_proportion = 0.1
	# Print
	print_each_n_step = 10


	model_name = "bert-base-cased"
	#model_name = "bert-base-uncased"
	# model_name = "roberta-base"
	#model_name = "albert-base-v2"
	#model_name = "xlm-roberta-base"
	#model_name = "amazon/bort"

	#  NOTE: in this setting 50 classes are involved
	labeled_file = "./ganbert/data/labeled.tsv"
	unlabeled_file = "./ganbert/data/unlabeled_full.tsv"
	test_filename = "./ganbert/data/valid.tsv"
	unseen_filename = "./ganbert/data/test.tsv"

	label_list = ["0", "1", "UNK"]

	fi = open("final_batch_result.csv", "a")

	fi.write("\nRunning for max seq len: " + str(length) + " for batch size: " + str(batch_size) + " for Seed value: " + str(seed) + "\n")

	transformer = AutoModel.from_pretrained(model_name)
	

	labeled_examples = get_qc_examples(labeled_file)
	unlabeled_examples = get_qc_examples(unlabeled_file)
	test_examples = get_qc_examples(test_filename)
	unseen_examples = get_qc_examples(unseen_filename)
	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i


	train_examples = labeled_examples
	train_label_masks = np.ones(len(labeled_examples), dtype=bool)
	if unlabeled_examples:
		train_examples = train_examples + unlabeled_examples
		tmp_masks = np.zeros(len(unlabeled_examples), dtype=bool)
		train_label_masks = np.concatenate([train_label_masks,tmp_masks])

	train_dataloader = generate_data_loader(train_examples, train_label_masks, label_map, do_shuffle = True, balance_label_examples = apply_balance, length = length, batch_size = batch_size)

	test_label_masks = np.ones(len(test_examples), dtype=bool)

	test_dataloader = generate_data_loader(test_examples, test_label_masks, label_map, do_shuffle = False, balance_label_examples = False, length = length, batch_size = batch_size)

	unseen_label_masks = np.ones(len(unseen_examples), dtype=bool)

	unseen_dataloader = generate_data_loader(unseen_examples, unseen_label_masks, label_map, do_shuffle = False, balance_label_examples = False, length = length, batch_size = batch_size)

	config = AutoConfig.from_pretrained(model_name)
	hidden_size = int(config.hidden_size)
	# Define the number and width of hidden layers
	hidden_levels_g = [hidden_size for i in range(0, num_hidden_layers_g)]
	hidden_levels_d = [hidden_size for i in range(0, num_hidden_layers_d)]


	generator = Generator(noise_size=noise_size, output_size=hidden_size, hidden_sizes=hidden_levels_g, dropout_rate=out_dropout_rate)
	discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_d, num_labels=len(label_list), dropout_rate=out_dropout_rate)

	# Put everything in the GPU if available
	if torch.cuda.is_available():    
	  generator.cuda()
	  discriminator.cuda()
	  transformer.cuda()
	  if multi_gpu:
	    transformer = torch.nn.DataParallel(transformer)


	training_stats = []

	# Measure the total training time for the whole run.
	total_t0 = time.time()

	#models parameters
	transformer_vars = [i for i in transformer.parameters()]
	d_vars = transformer_vars + [v for v in discriminator.parameters()]
	g_vars = [v for v in generator.parameters()]

	#optimizer
	dis_optimizer = torch.optim.AdamW(d_vars, lr=learning_rate_discriminator)
	gen_optimizer = torch.optim.AdamW(g_vars, lr=learning_rate_generator) 

#scheduler
	if apply_scheduler:
		num_train_examples = len(train_examples)
		num_train_steps = int(num_train_examples / batch_size * num_train_epochs)
		num_warmup_steps = int(num_train_steps * warmup_proportion)

		scheduler_d = get_constant_schedule_with_warmup(dis_optimizer, 
		                                       num_warmup_steps = num_warmup_steps)
		scheduler_g = get_constant_schedule_with_warmup(gen_optimizer, 
		                                       num_warmup_steps = num_warmup_steps)

	for epoch_i in range(0, num_train_epochs):

		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_train_epochs))
		print('Training...')

		fi.write('\n======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, num_train_epochs))
		# Measure how long the training epoch takes.
		t0 = time.time()

		# Reset the total loss for this epoch.
		tr_g_loss = 0
		tr_d_loss = 0

		# Put the model into training mode.
		transformer.train() 
		generator.train()
		discriminator.train()

    # For each batch of training data...
		for step, batch in enumerate(train_dataloader):

		# Progress update every print_each_n_step batches.
		# if step % print_each_n_step == 0 and not step == 0:
		#     # Calculate elapsed time in minutes.
		#     elapsed = format_time(time.time() - t0)
		    
		#     # Report progress.
		#     print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

		# Unpack this training batch from our dataloader. 
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)
			b_label_mask = batch[3].to(device)
			# print(b_input_ids.shape, b_input_mask.shape, b_labels.shape, b_label_mask.shape)

			# Encode real data in the Transformer
			model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
			hidden_states = model_outputs[-1]

			# Generate fake data that should have the same distribution of the ones
			# encoded by the transformer. 
			# First noisy input are used in input to the Generator
			noise = torch.zeros(b_input_ids.shape[0],noise_size, device=device).uniform_(0, 1)
			# Gnerate Fake data
			gen_rep = generator(noise)

			# Generate the output of the Discriminator for real and fake data.
			# First, we put together the output of the tranformer and the generator
			disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
			# Then, we select the output of the disciminator
			features, logits, probs = discriminator(disciminator_input)

			# Finally, we separate the discriminator's output for the real and fake
			# data
			features_list = torch.split(features, batch_size)
			D_real_features = features_list[0]
			D_fake_features = features_list[1]

			logits_list = torch.split(logits, batch_size)
			D_real_logits = logits_list[0]
			D_fake_logits = logits_list[1]

			probs_list = torch.split(probs, batch_size)
			D_real_probs = probs_list[0]
			D_fake_probs = probs_list[1]

			#---------------------------------
			#  LOSS evaluation
			#---------------------------------
			# Generator's LOSS estimation
			g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:,-1] + epsilon))
			g_feat_reg = torch.mean(torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
			g_loss = g_loss_d + g_feat_reg

			# Disciminator's LOSS estimation
			logits = D_real_logits[:,0:-1]
			log_probs = F.log_softmax(logits, dim=-1)
			# The discriminator provides an output for labeled and unlabeled real data
			# so the loss evaluated for unlabeled data is ignored (masked)
			label2one_hot = torch.nn.functional.one_hot(b_labels, len(label_list))
			# print(label2one_hot.data.cpu().numpy(), log_probs.data.cpu().numpy())
			# print(label2one_hot.shape, log_probs.shape)
			per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
			per_example_loss = torch.masked_select(per_example_loss, b_label_mask.to(device))
			labeled_example_count = per_example_loss.type(torch.float32).numel()
			
			if labeled_example_count == 0:
				D_L_Supervised = 0
			else:
				D_L_Supervised = torch.div(torch.sum(per_example_loss.to(device)), labeled_example_count)
	                 
			D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + epsilon))
			D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + epsilon))
			d_loss = D_L_Supervised + D_L_unsupervised1U + D_L_unsupervised2U

			#---------------------------------
			#  OPTIMIZATION
			#---------------------------------
			# Avoid gradient accumulation
			gen_optimizer.zero_grad()
			dis_optimizer.zero_grad()

			# Calculate weigth updates
			# retain_graph=True is required since the underlying graph will be deleted after backward
			g_loss.backward(retain_graph=True)
			d_loss.backward() 

			# Apply modifications
			gen_optimizer.step()
			dis_optimizer.step()

			# A detail log of the individual losses
			#print("{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}".
			#      format(D_L_Supervised, D_L_unsupervised1U, D_L_unsupervised2U,
			#             g_loss_d, g_feat_reg))

			# Save the losses to print them later
			tr_g_loss += g_loss.item()
			tr_d_loss += d_loss.item()

	        # Update the learning rate with the scheduler
			if apply_scheduler:
				scheduler_d.step()
				scheduler_g.step()

    # Calculate the average loss over all of the batches.
		avg_train_loss_g = tr_g_loss / len(train_dataloader)
		avg_train_loss_d = tr_d_loss / len(train_dataloader)             

		# Measure how long this epoch took.
		training_time = format_time(time.time() - t0)

		print("")
		print("  Average training loss generetor: {0:.3f}".format(avg_train_loss_g))
		print("  Average training loss discriminator: {0:.3f}".format(avg_train_loss_d))
		print("  Training epoch took: {:}".format(training_time))
		    
		# ========================================
		#     TEST ON THE EVALUATION DATASET
		# ========================================
		# After the completion of each training epoch, measure our performance on
		# our test set.
		print("")
		print("Running Validation...")

		t0 = time.time()

		# Put the model in evaluation mode--the dropout layers behave differently
		# during evaluation.
		transformer.eval() #maybe redundant
		discriminator.eval()
		generator.eval()

		# Tracking variables 
		total_test_accuracy = 0

		total_test_loss = 0
		nb_test_steps = 0

		all_preds = []
		all_labels_ids = []

		#loss
		nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

	    # Evaluate data for one epoch
		for batch in test_dataloader:
	        
			# Unpack this training batch from our dataloader. 
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)
	        
	        # Tell pytorch not to bother with constructing the compute graph during
	        # the forward pass, since this is only needed for backprop (training).
			with torch.no_grad():        
				model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
				hidden_states = model_outputs[-1]
				_, logits, probs = discriminator(hidden_states)
				###log_probs = F.log_softmax(probs[:,1:], dim=-1)
				filtered_logits = logits[:,0:-1]
				# Accumulate the test loss.
				total_test_loss += nll_loss(filtered_logits, b_labels)
	            
	        # Accumulate the predictions and the input labels
			_, preds = torch.max(filtered_logits, 1)
			all_preds += preds.detach().cpu()
			all_labels_ids += b_labels.detach().cpu()

	    # Report the final accuracy for this validation run.
		all_preds = torch.stack(all_preds).numpy()
		all_labels_ids = torch.stack(all_labels_ids).numpy()
		test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
		test_f1_macro = f1_score(all_labels_ids, all_preds, average='macro')
		test_f1_micro = f1_score(all_labels_ids, all_preds, average='micro')
		test_f1_weighted = f1_score(all_labels_ids, all_preds, average='weighted')
		test_prec_macro = precision_score(all_labels_ids, all_preds, average = 'macro')
		test_prec_0 = precision_score(all_labels_ids, all_preds, average = 'binary', pos_label=0)
		test_prec_1 = precision_score(all_labels_ids, all_preds, average = 'binary', pos_label=1)
		test_recall_macro = recall_score(all_labels_ids, all_preds, average='macro')
		test_recall_0 = recall_score(all_labels_ids, all_preds, average='binary', pos_label=0)
		test_recall_1 = recall_score(all_labels_ids, all_preds, average='macro', pos_label=1)

		print("  Accuracy: {0:.3f}".format(test_accuracy))

	    # for i, a in enumerate(zip(all_preds, all_labels_ids)):
	    #   if a[0] != a[1]:
	    #     print(sentence_store[tuple(all_labels_ids[i].tolist())])
	    # Calculate the average loss over all of the batches.
		avg_test_loss = total_test_loss / len(test_dataloader)
		avg_test_loss = avg_test_loss.item()

		# Measure how long the validation run took.
		test_time = format_time(time.time() - t0)

		print("  Validation Loss: {0:.3f}".format(avg_test_loss))
		print("  Validation took: {:}".format(test_time))
		print("classification report on validation data")
		print(classification_report(all_labels_ids, all_preds, digits = 4))

	    # Record all statistics from this epoch.
		training_stats.append(
		    {
		        'epoch': epoch_i + 1,
		        'Training Loss generator': avg_train_loss_g,
		        'Training Loss discriminator': avg_train_loss_d,
		        'Valid. Loss': avg_test_loss,
		        'Valid. Accur.': test_accuracy,
		        'Training Time': training_time,
		        'Valid f1 macro': test_f1_macro,
		        'Valid f1 micro': test_f1_macro,
		        'Valid f1 weighted': test_f1_weighted,
		        'Valid precision': test_prec_macro,
		        'Valid recall': test_recall_macro,
		        'Test Time': test_time
		    }
		)

		fi.write("Validation\t\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (test_accuracy, test_f1_macro, test_f1_micro, test_f1_weighted, test_prec_macro, test_prec_1, test_prec_0, test_recall_macro, test_recall_1, test_recall_0, avg_test_loss) )
		print("")
		print("Running on Unseen Data...")

		t0 = time.time()

		# Put the model in evaluation mode--the dropout layers behave differently
		# during evaluation.
		transformer.eval() #maybe redundant
		discriminator.eval()
		generator.eval()

		# Tracking variables 
		total_test_accuracy = 0

		total_test_loss = 0
		nb_test_steps = 0

		all_preds = []
		all_labels_ids = []

		#loss
		nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

	# Evaluate data for one epoch
		for batch in unseen_dataloader:
	    
		    # Unpack this training batch from our dataloader. 
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)

			# Tell pytorch not to bother with constructing the compute graph during
			# the forward pass, since this is only needed for backprop (training).
			with torch.no_grad():        
				model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
				hidden_states = model_outputs[-1]
				_, logits, probs = discriminator(hidden_states)
				###log_probs = F.log_softmax(probs[:,1:], dim=-1)
				filtered_logits = logits[:,0:-1]
				# Accumulate the test loss.
				total_test_loss += nll_loss(filtered_logits, b_labels)
	        
	    # Accumulate the predictions and the input labels
			_, preds = torch.max(filtered_logits, 1)
			all_preds += preds.detach().cpu()
			all_labels_ids += b_labels.detach().cpu()

	# Report the final accuracy for this validation run.
		all_preds = torch.stack(all_preds).numpy()
		all_labels_ids = torch.stack(all_labels_ids).numpy()
		test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
		test_f1_macro = f1_score(all_labels_ids, all_preds, average='macro')
		test_f1_micro = f1_score(all_labels_ids, all_preds, average='micro')
		test_f1_weighted = f1_score(all_labels_ids, all_preds, average='weighted')
		test_prec_macro = precision_score(all_labels_ids, all_preds, average = 'macro')
		test_prec_0 = precision_score(all_labels_ids, all_preds, average = 'binary', pos_label=0)
		test_prec_1 = precision_score(all_labels_ids, all_preds, average = 'binary', pos_label=1)
		test_recall_macro = recall_score(all_labels_ids, all_preds, average='macro')
		test_recall_0 = recall_score(all_labels_ids, all_preds, average='binary', pos_label=0)
		test_recall_1 = recall_score(all_labels_ids, all_preds, average='macro', pos_label=1)
		print("  Accuracy: {0:.3f}".format(test_accuracy))


		avg_test_loss = total_test_loss / len(test_dataloader)
		avg_test_loss = avg_test_loss.item()

		# Measure how long the validation run took.
		test_time = format_time(time.time() - t0)

		print("  Test Loss: {0:.3f}".format(avg_test_loss))
		print("  Test took: {:}".format(test_time))
		print(classification_report(all_labels_ids, all_preds, digits = 4))

		print("%.5f,\t\t%.5f,\t\t%.5f,\t\t%.5f,\t\t%.5f,\t\t%.5f" % (test_accuracy, test_f1_macro, test_f1_micro, test_f1_weighted, test_prec_macro, test_recall_macro) )
		fi.write("TestData\t\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (test_accuracy, test_f1_macro, test_f1_micro, test_f1_weighted, test_prec_macro, test_prec_1, test_prec_0, test_recall_macro, test_recall_1, test_recall_0, avg_test_loss) )


	for stat in training_stats:
		print(stat)  

		print("\nTraining complete!")

		print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


	print("Saving the models")

	torch.save(discriminator, "models/dicriminator"+str(length)+"_"+str(batch_size)+"_"+str(seed)+".pt")
	torch.save(generator, "models/generator"+str(length)+"_"+str(batch_size)+"_"+str(seed)+".pt")
	torch.save(transformer, "models/transformer"+str(length)+"_"+str(batch_size)+"_"+str(seed)+".pt")

	## INFERENCE

	print("")
	print("Running on Unseen Data...")

	t0 = time.time()

	# Put the model in evaluation mode--the dropout layers behave differently
	# during evaluation.
	transformer.eval() #maybe redundant
	discriminator.eval()
	generator.eval()

	# Tracking variables 
	total_test_accuracy = 0

	total_test_loss = 0
	nb_test_steps = 0

	all_preds = []
	all_labels_ids = []

	#loss
	nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

# Evaluate data for one epoch
	for batch in unseen_dataloader:
    
	    # Unpack this training batch from our dataloader. 
		b_input_ids = batch[0].to(device)
		b_input_mask = batch[1].to(device)
		b_labels = batch[2].to(device)

		# Tell pytorch not to bother with constructing the compute graph during
		# the forward pass, since this is only needed for backprop (training).
		with torch.no_grad():        
			model_outputs = transformer(b_input_ids, attention_mask=b_input_mask)
			hidden_states = model_outputs[-1]
			_, logits, probs = discriminator(hidden_states)
			###log_probs = F.log_softmax(probs[:,1:], dim=-1)
			filtered_logits = logits[:,0:-1]
			# Accumulate the test loss.
			total_test_loss += nll_loss(filtered_logits, b_labels)
        
    # Accumulate the predictions and the input labels
		_, preds = torch.max(filtered_logits, 1)
		all_preds += preds.detach().cpu()
		all_labels_ids += b_labels.detach().cpu()

# Report the final accuracy for this validation run.
	all_preds = torch.stack(all_preds).numpy()
	all_labels_ids = torch.stack(all_labels_ids).numpy()
	test_accuracy = np.sum(all_preds == all_labels_ids) / len(all_preds)
	test_f1_macro = f1_score(all_labels_ids, all_preds, average='macro')
	test_f1_micro = f1_score(all_labels_ids, all_preds, average='micro')
	test_f1_weighted = f1_score(all_labels_ids, all_preds, average='weighted')
	test_prec_macro = precision_score(all_labels_ids, all_preds, average = 'macro')
	test_prec_0 = precision_score(all_labels_ids, all_preds, average = 'binary', pos_label=0)
	test_prec_1 = precision_score(all_labels_ids, all_preds, average = 'binary', pos_label=1)
	test_recall_macro = recall_score(all_labels_ids, all_preds, average='macro')
	test_recall_0 = recall_score(all_labels_ids, all_preds, average='binary', pos_label=0)
	test_recall_1 = recall_score(all_labels_ids, all_preds, average='macro', pos_label=1)
	print("  Accuracy: {0:.3f}".format(test_accuracy))


	avg_test_loss = total_test_loss / len(test_dataloader)
	avg_test_loss = avg_test_loss.item()

	# Measure how long the validation run took.
	test_time = format_time(time.time() - t0)

	print("  Test Loss: {0:.3f}".format(avg_test_loss))
	print("  Test took: {:}".format(test_time))
	print(classification_report(all_labels_ids, all_preds, digits = 4))

	print("%.5f,\t\t%.5f,\t\t%.5f,\t\t%.5f,\t\t%.5f,\t\t%.5f" % (test_accuracy, test_f1_macro, test_f1_micro, test_f1_weighted, test_prec_macro, test_recall_macro) )

	results.append({
		"Test acc": test_accuracy,
		"Test F1": test_f1_macro,
		"Length": length,
		"valid acc": training_stats[len(training_stats)-1]["Valid. Accur."],
		"valid F1": training_stats[len(training_stats)-1]["Valid f1 macro"]
		})
	fi.write("\nTestFinal\t\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n" % (test_accuracy, test_f1_macro, test_f1_micro, test_f1_weighted, test_prec_macro, test_prec_1, test_prec_0, test_recall_macro, test_recall_1, test_recall_0, avg_test_loss) )

	fi.close()

length = [128, 64]
seeds = [5, 21, 42]
batches = [128, 100, 64, 32, 16, 8]

for seed in seeds:
	for batch in batches:
		for leng in length:
			if batch >= 100 and leng >= 100:
				continue
			main(leng, seed, batch) 

print(results)
