# -*- coding: utf-8 -*-
"""model.ipynb

"""

!pip install transformers
!pip install rouge
!pip install sentencepiece

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import transformers
print(transformers.__version__)

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rouge import Rouge
#from rouge_metric import PyRouge
from tqdm import tqdm

# from transformers import AutoTokenizer, LEDConfig, LEDForConditionalGeneration, AdamW, LEDTokenizer
# from transformers import T5Tokenizer, T5ForConditionalGeneration

from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM, AdamW
from transformers import AlbertTokenizer, AutoTokenizer

"""# Load Suggested Model and Tokenizer"""

### Load Pretrained Model and Tokenizer

# pretrain = "allenai/led-base-16384"
# tokenizer = AutoTokenizer.from_pretrained(pretrain)
# config=LEDConfig.from_pretrained(pretrain)
# model = LEDForConditionalGeneration.from_pretrained(pretrain)


############### This from Hugging Face ######################
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/MultiIndicParaphraseGenerationSS", do_lower_case=False, use_fast=False, keep_accents=True)
# Or use tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/MultiIndicParaphraseGenerationSS", do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/MultiIndicParaphraseGenerationSS")
# Or use model = MBartForConditionalGeneration.from_pretrained("ai4bharat/MultiIndicParaphraseGenerationSS")

# Some initial mapping
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

model = torch.nn.DataParallel(model, device_ids=[0])
####################################
# inp = tokenizer("दिल्ली यूनिवर्सिटी देश की प्रसिद्ध यूनिवर्सिटी में से एक है. </s> <2hi>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"""# Data


"""

train_df = pd.read_csv("train_new.csv")  ################## Change File Path ###################
train_df.head()

test_df = pd.read_csv("test_new.csv")  ################# Change File Path ################
test_df.head()

"""# Dataloader"""

# Hyperparameters

lr = 3e-4
num_epochs = 5
batch_size = 16
input_max_length = 512
output_max_length = 70

##################################### Careful Check your hyperparameters cautiously  #############################

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.texts = self.data['body'].tolist()
        self.summaries = self.data['summary'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]
        return {'text':str(text), 'summary': str(summary)}

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    summaries = [item['summary'] for item in batch]
    encoded_text = tokenizer(texts, padding=True, truncation=True, max_length= input_max_length, return_tensors='pt')
    encoded_summary = tokenizer(summaries, padding=True, truncation=True, max_length= output_max_length, return_tensors='pt')
    return {'input_ids': encoded_text['input_ids'],
            'attention_mask': encoded_text['attention_mask'],
            'decoder_input_ids': encoded_summary['input_ids'][:, :-1],
            'decoder_attention_mask': encoded_summary['attention_mask'][:, :-1],
            'decoder_target_ids': encoded_summary['input_ids'][:, 1:],
            'decoder_target_attention_mask': encoded_summary['attention_mask'][:, 1:]}

# Load the dataset and create the DataLoader
train_dataset = CustomDataset('train_new.csv')   ##################### Change File Path #########################
train_dataloader = DataLoader(train_dataset, batch_size= batch_size, collate_fn=collate_fn)

test_dataset = CustomDataset('test_new.csv')    ##################### Change File Path #########################
test_dataloader = DataLoader(test_dataset, batch_size= batch_size, collate_fn=collate_fn)



"""# Training"""

# Loss Function and Optimizer

optimizer = AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training

model.to(device)
model.train()
for epoch in range(num_epochs):
    accumulating_loss = 0
    for batch in tqdm(train_dataloader):
        # Unpacking Elements
        input_ids = bat
        ch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        decoder_target_ids = batch["decoder_target_ids"].to(device)
        decoder_target_attention_mask = batch["decoder_target_attention_mask"].to(device)
        # Forward Pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask)
        loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)),decoder_target_ids.contiguous().view(-1))
        # print(loss)

        # Backward pass\
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accumulating_loss += loss.item()
    print(f"For Epoch {epoch} Loss is {accumulating_loss}")

outputs.keys()

outputs.encoder_last_hidden_state

outputs.encoder_last_hidden_state.shape

### Save your model in google drive

torch.save(model, "hindi_train_1.pt")

model=torch.load("hindi_train_1.pt")

"""###### Text Embedding Generation ########"""

model.to(device)
model.eval()

embeddings_stored = []

for batch in tqdm(train_dataloader):
    # Unpacking Elements
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    decoder_input_ids = batch["decoder_input_ids"].to(device)
    decoder_attention_mask = batch["decoder_attention_mask"].to(device)
    decoder_target_ids = batch["decoder_target_ids"].to(device)
    decoder_target_attention_mask = batch["decoder_target_attention_mask"].to(device)
    # Forward Pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask)
    hidden_state = outputs.encoder_last_hidden_state
    hidden_state = hidden_state.cpu().detach().numpy()
    embeddings_stored.append(hidden_state)

print(len(embeddings_stored))
print("Concatinating")
all_embeddings = np.concatenate(embeddings_stored, axis = 0)
print(all_embeddings.shape)

np.save("all_text_embeddings.npy", all_embeddings)

hidden_state.shape

hidden_state.to_numpy

"""# Evaluation"""

# Evaluation for Test

model.eval()
model.to("cuda")

# Define the ROUGE metric
rouge = Rouge()

# Initialize lists to store the ROUGE scores
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
# rougesu = []

predictions = []
# Loop through the test data and compute the ROUGE scores for each example
for i, row in test_df.iterrows():

    # Get the article text
    article = str(row["body"])   ############## Load Body Here ###################

    # Get the target summary
    summary = str(row["summary"])   ###################### Load Summary Here ####################

    # Encode the inputs and generate the predicted summary
    encoded_text = tokenizer(article, padding=True, truncation=True, max_length= input_max_length, return_tensors='pt')
    output_ids = model.module.generate(encoded_text['input_ids'].to('cuda'), max_length=output_max_length, num_beams=4)  ##### Change max_length according to hyperparameters ################
    predicted_summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    ############################ This is from Huggingface ###########################
    # inp = tokenizer("दिल्ली यूनिवर्सिटी देश की प्रसिद्ध यूनिवर्सिटी में से एक है. </s> <2hi>", add_special_tokens=False, return_tensors="pt", padding=True).input_ids

    # # For generation. Pardon the messiness. Note the decoder_start_token_id.

    # model_output=model.generate(inp, use_cache=True,no_repeat_ngram_size=3,encoder_no_repeat_ngram_size=3, num_beams=4, max_length=20, min_length=1, early_stopping=True, pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id, decoder_start_token_id=tokenizer._convert_token_to_id_with_added_voc("<2hi>"))

    # # Decode to get output strings
    #  decoded_output=tokenizer.decode(model_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    ############################################################################

    predicted_summary+=" "
    predictions.append(predicted_summary)
    # Compute the ROUGE scores for the predicted summary
    score = rouge.get_scores(predicted_summary, summary)
    # rougeSU4 = rouge_su_4(summary, predicted_summary)[-1]

    # Store the scores
    rouge1_scores.append(score[0]["rouge-1"]["f"])
    rouge2_scores.append(score[0]["rouge-2"]["f"])
    rougeL_scores.append(score[0]["rouge-l"]["f"])
    # rougesu.append(rougeSU4)

# Compute the average ROUGE scores
rouge1_avg = sum(rouge1_scores) / len(rouge1_scores)
rouge2_avg = sum(rouge2_scores) / len(rouge2_scores)
rougeL_avg = sum(rougeL_scores) / len(rougeL_scores)
# rougesu_avg = sum(rougesu)/len(rougesu)

# Print the results
print(f"Test Average ROUGE-1: {rouge1_avg:.3f}")
print(f"Test Average ROUGE-2: {rouge2_avg:.3f}")
print(f"Test Average ROUGE-L: {rougeL_avg:.3f}")
# print(f"Test Average ROUGE-SU4: {rougesu_avg:.3f}")

import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# Set up the reference summaries as a list of lists of words (tokenized)
references = [str(summary).split() for summary in test_df["summary"]]

# Initialize lists to store the BLEU scores
bleu_scores = []

predictions = []
# Loop through the test data and compute the BLEU scores for each example
for i, row in test_df.iterrows():

    # Get the article text
    article = str(row["body"])   ############## Load Body Here ###################

    # Get the target summary
    summary = str(row["summary"])   ###################### Load Summary Here ####################

    # Encode the inputs and generate the predicted summary
    encoded_text = tokenizer.encode(article, padding=True, truncation=True, max_length=input_max_length, return_tensors='pt', add_special_tokens=False)
    output_ids = model.module.generate(encoded_text.to('cuda'), max_length=output_max_length, num_beams=4)  ##### Change max_length according to hyperparameters ################
    predicted_summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    predicted_summary += " "
    predictions.append(predicted_summary)

    # Compute the BLEU score for the predicted summary
    bleu_score = sentence_bleu(references, predicted_summary.split())

    # Store the scores
    bleu_scores.append(bleu_score)

# Compute the average BLEU score
bleu_avg = sum(bleu_scores) / len(bleu_scores)

# Print the results
print(f"Test Average BLEU: {bleu_avg:.3f}")