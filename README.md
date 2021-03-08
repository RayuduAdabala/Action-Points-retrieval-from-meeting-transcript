# Action-Points-retrieval-from-meeting-transcript
Classifying the dialogues speech from the meeting transcript using transformer based XLNet pretrained model fine tuning with AMI meeting corpus
Action points retrieval system

Use case: Action point’s retrieval from meeting transcript and extracted points will be summarized and sent to all the participants.

Implementation:
Data for training and testing the model is taken from AMI meeting corpus and data in the xml is prepared accordingly for training multi label text classification model.
For Multi label text classification we have used XLNet transformer based model from Huggingface using Pytorch. 

XLNet is an extension of the Transformer-XL model pre-trained using an autoregressive method to learn bidirectional contexts by maximizing the expected likelihood over all permutations of the input sequence factorization order.



#Finetuned model with AMI corpus can be downloaded from below given Gdrive link.
gdrive link : https://drive.google.com/file/d/1ckZ50JtitJ0qbWHbD9RfUqPSZ3lm8KQI/view?usp=sharing

Prepared AMI meeting corpus data is saved as sent_annot.pkl which can be directly used for training.

AMI corpus data document and annotations: 
https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml
https://groups.inf.ed.ac.uk/ami/corpus/Guidelines/dialogue_acts_manual_1.0.pdf
