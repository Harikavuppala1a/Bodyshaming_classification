import os
import time
import sys
from sent_enc_embed import sent_enc_featurize
from word_embed import word_featurize
from neural_approaches import *

sys.setrecursionlimit(10000)
conf_dict_list, conf_dict_com = load_config(sys.argv[1])
# os.environ["CUDA_VISIBLE_DEVICES"] = conf_dict_com['GPU_ID']
# from keras.backend.tensorflow_backend import set_session
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

os.makedirs(conf_dict_com["output_folder_name"], exist_ok=True)
os.makedirs(conf_dict_com["save_folder_name"], exist_ok=True)

##path to save the results
tsv_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_tsv_filename"]
if os.path.isfile(tsv_path):
    f_tsv = open(tsv_path, 'a')
else:
    f_tsv = open(tsv_path, 'w')
    f_tsv.write("") 

data_dict = load_data(conf_dict_com["filename"], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com["test_mode"])
print("max # sentences: %d, max # words per sentence: %d, max # words per post: %d" % (data_dict['max_num_sent'], data_dict['max_words_sent'], data_dict['max_post_length']))

###get features and save those features

startTime = time.time()
##create fname_part which is unique filename of your model like below fname_part
# fname_part = ("%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s" % (model_type,word_feat_str,sent_enc_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,rnn_type,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag, conf_dict_com["test_mode"]))

# we need to reprt on average of three runs
for run_ind in range(conf_dict_com["num_runs"]):
    print('run: %s; %s\n' % (run_ind, info_str))
    if run_ind < len(mod_op_list_save_list):
        mod_op_list = mod_op_list_save_list[run_ind]   
    else:
        mod_op_list = []
        ### define a function which should save the model and return the model output
        mod_op = train_predict()
        mod_op_list.append(mod_op)
        mod_op_list_save_list.append(mod_op_list)
    ##evaluate model in this function
    evaluate_model()

## it should aggreate all three runs and print the results. 
aggregate_metr()
timeLapsed = int(time.time() - startTime + 0.5)
hrs = timeLapsed/3600.
t_str = "%.1f hours = %.1f minutes over %d hours\n" % (hrs, (timeLapsed % 3600)/60.0, int(hrs))
print(t_str)                

f_tsv.close()

##program to save the model
# if save_model:    
#             print("saving model o/p")
#             os.makedirs(save_folder_name + fname_part, exist_ok=True)
#             with open(fname_mod_op, 'wb') as f:
#                 pickle.dump(mod_op, f)


###program to load the saved model
# if use_saved_model and os.path.isfile(fname_mod_op):
#         print("loading model o/p")
#         with open(fname_mod_op, 'rb') as f:
#             mod_op = pickle.load(f)