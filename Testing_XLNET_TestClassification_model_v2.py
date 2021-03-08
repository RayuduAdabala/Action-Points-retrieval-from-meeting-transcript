#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install transformers')
# get_ipython().system('pip install torchvision')
# get_ipython().system('pip install SentencePiece ')
import os
import math

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
# import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# 

# In[2]:


# from google.colab import drive
# drive.mount('/content/drive')


# ## Tokenzation and Attention

# In[3]:


tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)


# In[4]:


def tokenize_inputs(text_list, tokenizer, num_embeddings=512):
    """
    Tokenizes the input text input into ids. Appends the appropriate special
    characters to the end of the text to denote end of sentence. Truncate or pad
    the appropriate sequence length.
    """
    # tokenize the text, then truncate sequence to the desired length minus 2 for
    # the 2 special characters
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings-2], text_list))
    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # append special token "<s>" and </s> to end of sentence
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # pad sequences
    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
    return input_ids

def create_attn_masks(input_ids):
    """
    Create attention masks to tell model whether attention should be applied to
    the input id tokens. Do not want to perform attention on padding tokens.
    """
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks


# ## Load fintuned model 

# In[5]:


def load_model(save_path):
    """
    Load the model from the path directory provided
    """
    checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
    model_state_dict = checkpoint['state_dict']
    model = XLNetForMultiLabelSequenceClassification(num_labels=model_state_dict["classifier.weight"].size()[0])
    model.load_state_dict(model_state_dict)

    epochs = checkpoint["epochs"]
    lowest_eval_loss = checkpoint["lowest_eval_loss"]
    train_loss_hist = checkpoint["train_loss_hist"]
    valid_loss_hist = checkpoint["valid_loss_hist"]
  
    return model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist


# In[6]:


# torch.cuda.empty_cache()


# In[1]:


config = XLNetConfig()
        
class XLNetForMultiLabelSequenceClassification(torch.nn.Module):
    def __init__(self, num_labels=2):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.classifier = torch.nn.Linear(768, num_labels)

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None,              attention_mask=None, labels=None):
    # last hidden layer
        last_hidden_state = self.xlnet(input_ids=input_ids,                                   attention_mask=attention_mask,                                   token_type_ids=token_type_ids)
    # pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)
        
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),                      labels.view(-1, self.num_labels))
            return loss
        else:
            return logits
    
    def freeze_xlnet_decoder(self):
        """
        Freeze XLNet weight parameters. They will not be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = False
    
    def unfreeze_xlnet_decoder(self):
        """
        Unfreeze XLNet weight parameters. They will be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = True
    
    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector 
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state
    
# model = XLNetForMultiLabelSequenceClassification(num_labels=len(Y_train[0]))
# model = torch.nn.DataParallel(model)
# model.cuda()


# In[8]:


cwd = os.getcwd()
model_save_path = output_model_file = os.path.join(cwd, "xlnet_toxic.bin")
model, start_epoch, lowest_eval_loss, train_loss_hist, valid_loss_hist = load_model(model_save_path)


# In[9]:


def generate_predictions(model, df, num_labels, device="cpu", batch_size=32):
    num_iter = math.ceil(df.shape[0]/batch_size)
  
    pred_probs = np.array([]).reshape(0, num_labels)
  
    model.to(device)
    model.eval()
  
    for i in range(num_iter):
        df_subset = df.iloc[i*batch_size:(i+1)*batch_size,:]
        X = df_subset["features"].values.tolist()
        masks = df_subset["masks"].values.tolist()
        X = torch.tensor(X)
        masks = torch.tensor(masks, dtype=torch.long)
        X = X.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            logits = model(input_ids=X, attention_mask=masks)
            logits = logits.sigmoid().detach().cpu().numpy()
            pred_probs = np.vstack([pred_probs, logits])
  
    return pred_probs


# ## Load Meeting Transcript

# In[10]:


import pandas as pd
df=pd.read_csv('transcript.txt',names=["sent"],sep='/:')
df.to_csv('someFileName.csv')


# In[11]:


df_list=df.values.tolist()
df.head()


# In[12]:


import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
new_list=[]
for i in df_list:
    j=str(i).strip("[]'")
    indx = j.find(":")#position of 'I'
    intro = j[indx+1:]
    new_list.append(intro)
# re.sub(r'[a-zA-Z]+[:]', 'I', stri)
# flat_list = [item for sublist in new_list for item in sublist]
new_list


# In[13]:


import numpy as np
test_np = np.array(new_list)
test_np


# In[14]:


# create input id tokens
test_np_input_ids = tokenize_inputs(test_np, tokenizer, num_embeddings=250)
test_np_input_ids


# In[15]:


# create attention masks
test_np_attention_masks = create_attn_masks(test_np_input_ids)
test_np_attention_masks


# In[16]:


import pandas as pd
dataset_unseen = pd.DataFrame()
dataset_unseen['Sent'] = test_np.tolist()
dataset_unseen.shape


# In[17]:


dataset_unseen["features"] = test_np_input_ids.tolist()
dataset_unseen["masks"] = test_np_attention_masks


# 

# In[18]:


dataset_unseen1=dataset_unseen[["Sent","features","masks"]]
dataset_unseen1.head()


# In[19]:


label_cols = ["label_ami_da_1","label_ami_da_11","label_ami_da_12","label_ami_da_13","label_ami_da_14","label_ami_da_15","label_ami_da_16","label_ami_da_2","label_ami_da_3","label_ami_da_4","label_ami_da_5","label_ami_da_6","label_ami_da_7","label_ami_da_8","label_ami_da_9"]
num_labels = len(label_cols)
pred_probs = generate_predictions(model, dataset_unseen1, num_labels, device="cpu", batch_size=32)
pred_probs


# In[20]:


# label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
label_cols = ["label_ami_da_1","label_ami_da_11","label_ami_da_12","label_ami_da_13","label_ami_da_14","label_ami_da_15","label_ami_da_16","label_ami_da_2","label_ami_da_3","label_ami_da_4","label_ami_da_5","label_ami_da_6","label_ami_da_7","label_ami_da_8","label_ami_da_9"]
dataset_unseen1["label_ami_da_1"] = pred_probs[:,0]
dataset_unseen1["label_ami_da_11"] = pred_probs[:,1]
dataset_unseen1["label_ami_da_12"] = pred_probs[:,2]
dataset_unseen1["label_ami_da_13"] = pred_probs[:,3]
dataset_unseen1["label_ami_da_14"] = pred_probs[:,4]
dataset_unseen1["label_ami_da_15"] = pred_probs[:,5]
dataset_unseen1["label_ami_da_16"] = pred_probs[:,6]
dataset_unseen1["label_ami_da_2"] = pred_probs[:,7]
dataset_unseen1["label_ami_da_3"] = pred_probs[:,8]
dataset_unseen1["label_ami_da_4"] = pred_probs[:,9]
dataset_unseen1["label_ami_da_5"] = pred_probs[:,10]
dataset_unseen1["label_ami_da_6"] = pred_probs[:,11]
dataset_unseen1["label_ami_da_7"] = pred_probs[:,12]
dataset_unseen1["label_ami_da_8"] = pred_probs[:,13]
dataset_unseen1["label_ami_da_9"] = pred_probs[:,14]


# In[21]:


dataset_unseen1['HighScore'] = dataset_unseen1.max(axis=1)


# In[22]:


dataset_unseen1


# In[23]:


Dict_label = {"label_ami_da_1":"Backchannel","label_ami_da_11":"Elicit-Assessment","label_ami_da_12":"Comment-About-Understanding","label_ami_da_13":"Elicit-Comment-Understanding","label_ami_da_14":"Be-Positive","label_ami_da_15":"Be-Negative","label_ami_da_16":"Other","label_ami_da_2":"Stall","label_ami_da_3":"Fragment","label_ami_da_4":"Inform","label_ami_da_5":"Elicit-Inform","label_ami_da_6":"Suggest","label_ami_da_7":"Offer","label_ami_da_8":"Elicit-Offer-Or-Suggestion","label_ami_da_9":"Assess"}
new_lst=[]
new_lst_label_name=[]
d=dataset_unseen1.columns[3:-1]
l=len(dataset_unseen1)
for i in range(l):
    for col in d:
        if dataset_unseen1[col][i] == dataset_unseen1["HighScore"][i]:
            new_lst.append(col)
            for k, v in Dict_label.items():
                if k == col:
                    new_lst_label_name.append(v)


      


# In[24]:


dataset_unseen1["Label"]=new_lst
dataset_unseen1["Label_name"]=new_lst_label_name
dataset_unseen1[["Sent","HighScore","Label","Label_name"]].to_csv("output_action_itmes")


# ## Trying to implement Abstractive Dialogue Summarization 

# In[40]:


# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# import torch
#
#
# model_name = 'google/pegasus-xsum'
# torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
# # batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest', return_tensors="pt").to(torch_device)
# # translated = model.generate(**batch)
# # tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
# # assert tgt_text[0] == "California's largest electricity provider has turned off power to hundreds of thousands of customers."
#
#
# # In[41]:
#
#
# src_text = [
#     """ we have completed four modules and started the fifth module and quality and testing also happening in prallel.
#      we have to give demo to the user on first week of next month, so be ready. please start the frontend work and complete it by end of the day. """
# ]
# batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest', return_tensors="pt").to(torch_device)
# translated = model.generate(**batch)
# tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
# tgt_text


# In[ ]:




