# -*- coding: utf-8 -*-
"""baseline_with_imagepointer.ipynb

"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["TORCH_USE_CUDA_DSA"]="1"

import os
import numpy as np
import pandas as pd
import json
import warnings
import logging
import gc
import random
import math
import re
from PIL import Image
import ast
from tqdm import tqdm
from typing import Optional
from datetime import datetime
import torchvision.models as models
import os.path

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
# from rouge_score.rouge_scorer import RougeScorer
from torchvision import transforms
# from vit_pytorch import ViT
# from vit_pytorch.extractor import Extractor
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, classification_report
# from vit_pytorch.regionvit import RegionViT

from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pytorch_lightning as pl
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, TensorDataset
# from img_transformer import ImageTransformerEncoder


# import clip
torch.cuda.set_device(0)
# torch.cuda.empty_cache()
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

import pandas as pd

# Read the CSV file
data = pd.read_csv('/mnt/Data/yashv7523/SHUBHAM/pre4.csv')  # Replace with the actual path to your CSV

# Group the data by summary and 0.para columns
grouped = data.groupby(['summary', '0.para'])

# Filter groups with both label values 0 and 1
filtered_groups = []
for _, group in grouped:
    if group['label'].nunique() == 2:  # Check if there are both label values 0 and 1
        filtered_groups.append(group)

# Combine the filtered groups into a new DataFrame
filtered_data = pd.concat(filtered_groups, ignore_index=True)

# # Save the filtered data to a new CSV file
# filtered_data.to_csv('filtered_data.csv', index=False)

df1 = filtered_data[0:2000]

df1.to_csv('train1.csv')

df1.shape

df1.head()

df2 = filtered_data[2000:2100]

df2.to_csv('test1.csv')

df3 = filtered_data[2100:2200]

df3.to_csv('val1.csv')

MODEL_OUTPUT_DIR = r'/mnt/Data/yashv7523/SHUBHAM/MAFsaved/'
RESULT_OUTPUT_DIR = r'/mnt/Data/yashv7523/SHUBHAM/MAFsavedRes/'

path_to_images = r'/mnt/Data/yashv7523/yash/bbchindi/imagefolder/'

path_to_train = r'/mnt/Data/yashv7523/SHUBHAM/train1.csv'

path_to_val = r'/mnt/Data/yashv7523/SHUBHAM/val1.csv'

path_to_test = r'/mnt/Data/yashv7523/SHUBHAM/test1.csv'

LOWERCASE_UTTERANCES = False
UNFOLDED_DIALOGUE = True

if UNFOLDED_DIALOGUE:
    SOURCE_COLUMN = 'dialogue'
else:
    SOURCE_COLUMN_1 = 'target'
    SOURCE_COLUMN_2 = 'context'



SOURCE_MAX_LEN = 1024
TARGET_MAX_LEN = 50
MAX_UTTERANCES = 25

ACOUSTIC_DIM = 154
ACOUSTIC_MAX_LEN = 600

VISUAL_DIM = 2048
VISUAL_MAX_LEN = 49

BATCH_SIZE = 1
MAX_EPOCHS = 1

BASE_LEARNING_RATE = 3e-5
NEW_LEARNING_RATE = 3e-5
# BASE_LEARNING_RATE=7e-5
# NEW_LEARNING_RATE=7e-5
WEIGHT_DECAY = 1e-4

NUM_BEAMS = 4
EARLY_STOPPING = True
NO_REPEAT_NGRAM_SIZE = 3

EARLY_STOPPING_THRESHOLD = 5
best_as1=-1
best_as2=-1
best_as3=-1
best_as4=-1
best_as5=-1
best_as6=-1
best_acc_cmp=-1
best_acc_em = -1
best_acc_se =-1
best_acc_sev =-1
best_bleu=-1
best_iouf1 = -1
best_tokenf1 = -1
best_jacc = -1
best_indi1 = -1
best_indi2 = -1
best_indi3 = -1
best_indi4 = -1
best_comp = -1
best_sarc = -1
best_emo = -1
best_senti = -1
best_sev = -1
def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(42)

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from transformers.modeling_utils import PreTrainedModel, unwrap_model

from transformers import (
    BartTokenizerFast,
    AdamW
)

from transformers.models.bart.configuration_bart import BartConfig

from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    BartDecoder,
    BartLearnedPositionalEmbedding,
    BartEncoderLayer,
    shift_tokens_right,
    _make_causal_mask,
    _expand_mask
)


from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput
)


from transformer_encoder import TransformerEncoder

class MSEDataset(Dataset):
    def __init__(self, path_to_data_df, path_to_images, tokenizer, image_transform):
        self.data = pd.read_csv(path_to_data_df)
        # self.data = self.data.iloc[1:]

        self.path_to_images = path_to_images
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        # print('*(((((((((((((((((((((((((((')
        # print(row)
        # time.sleep(120)


        image_name = row['images']
        src_text = str(row['0.para'])
        # print(src_text)
        # print(type(src_text))
        target_text = str(row['summary'])

        max_length = 256
        encoded_dict = tokenizer.encode_plus(
            text=src_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            add_prefix_space = True
        )
        # print()
        src_ids = encoded_dict['input_ids'][0]
        src_mask = encoded_dict['attention_mask'][0]

        image_path = os.path.join(self.path_to_images, image_name)
        img = np.array(Image.open(image_path).convert('RGB'))
        img_inp = self.image_transform(img)


        encoded_dict = tokenizer(
          target_text,
          max_length=max_length,
          padding="max_length",
          truncation=True,
          return_tensors='pt',
          add_prefix_space = True
        )

        target_ids = encoded_dict['input_ids'][0]

        sample = {
            "input_ids": src_ids,
            "attention_mask": src_mask,
            "input_image": img_inp,
            "target_ids": target_ids,
        }
        return sample

    def __len__(self):
        return self.data.shape[0]

class MSEDataModule(pl.LightningDataModule):
    def __init__(self, path_to_train_df, path_to_val_df, path_to_test_df, path_to_images, tokenizer, image_transform, batch_size=1):
        super(MSEDataModule, self).__init__()
        self.path_to_train_df = path_to_train_df
        self.path_to_val_df = path_to_val_df
        self.path_to_test_df = path_to_test_df
        self.path_to_images = path_to_images
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def setup(self, stage=None):
        self.train_dataset = MSEDataset(self.path_to_train_df, self.path_to_images, self.tokenizer, self.image_transform)
        self.val_dataset = MSEDataset(self.path_to_val_df, self.path_to_images, self.tokenizer, self.image_transform)
        self.test_dataset = MSEDataset(self.path_to_test_df, self.path_to_images, self.tokenizer, self.image_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, sampler = RandomSampler(self.train_dataset), batch_size = self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = 1)


class ContextAwareAttention(nn.Module):

    def __init__(self,
                 dim_model: int,
                 dim_context: int,
                 dropout_rate: Optional[float]=0.0):
        super(ContextAwareAttention, self).__init__()

        self.dim_model = dim_model
        self.dim_context = dim_context
        self.dropout_rate = dropout_rate
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.dim_model,
                                                     num_heads=1,
                                                     dropout=self.dropout_rate,
                                                     bias=True,
                                                     add_zero_attn=False,
                                                     batch_first=True,
                                                     device=DEVICE)


        self.u_k = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_k = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_k = nn.Linear(self.dim_model, 1, bias=False)

        self.u_v = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_v = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_v = nn.Linear(self.dim_model, 1, bias=False)





    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                context: Optional[torch.Tensor]=None):

        key_context = self.u_k(context)
        value_context = self.u_v(context)

        lambda_k = F.sigmoid(self.w1_k(k) + self.w2_k(key_context))
        lambda_v = F.sigmoid(self.w1_v(v) + self.w2_v(value_context))

        k_cap = (1 - lambda_k) * k + lambda_k * key_context
        v_cap = (1 - lambda_v) * v + lambda_v * value_context

        attention_output, _ = self.attention_layer(query=q,
                                                   key=k_cap,
                                                   value=v_cap)
        return attention_output



class MAF(nn.Module):

    def __init__(self,
                 dim_model: int,
                 dropout_rate: int):
        super(MAF, self).__init__()
        self.dropout_rate = dropout_rate

        # self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias=False)
        self.visual_context_transform = nn.Linear(49, SOURCE_MAX_LEN, bias=False)

        # self.acoustic_context_attention = ContextAwareAttention(dim_model=dim_model,
        #                                                         dim_context=ACOUSTIC_DIM,
        #                                                         dropout_rate=dropout_rate)
        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=512,
                                                              dropout_rate=dropout_rate)
        # self.acoustic_gate = nn.Linear(2*dim_model, dim_model)
        self.visual_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)





    def forward(self,
                text_input: torch.Tensor,
                visual_context: Optional[torch.Tensor]=None):

      

        # Video as Context for Attention
        visual_context = visual_context.permute(0, 2, 1)
        # print(visual_context.shape)
        visual_context = self.visual_context_transform(visual_context)
        visual_context = visual_context.permute(0, 2, 1)

        video_out = self.visual_context_attention(q=text_input,
                                                  k=text_input,
                                                  v=text_input,
                                                  context=visual_context)

        weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))

        output = self.final_layer_norm(text_input +
                                       # weight_a * audio_out +
                                    weight_v * video_out)

        return output


list13 = []
class MultimodalBartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()
        self.gradient_checkpointing = False

        # ================================ Modifications ================================ #
        self.fusion_at_layer = [5]
        # 7
        # self.clipmodel, self.clippreprocess = clip.load("ViT-B/32", device=DEVICE)
        self.vgg=models.vgg19_bn(pretrained=True)
        self.image_encoder = list(self.vgg.children())[0]

        self.feat11_lrproj = nn.Linear(512, 256)
        self.feat12_lrproj = nn.Linear(512, 256)
        self.feat21_lrproj = nn.Linear(768, 256)
        self.feat22_lrproj = nn.Linear(224, 256)
        self.tanh = torch.nn.Tanh()
        self.output_net2 = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 768)
        )

        self.visual_transformer = TransformerEncoder(d_model=VISUAL_DIM,
                                                     n_layers=4,
                                                     n_heads=8,
                                                     d_ff=VISUAL_DIM)
   
        self.MAF_layer = MAF(dim_model=embed_dim,
                             dropout_rate=0.2)
    def MLB2(self, feat1, feat2):

        z=self.MAF_layer(feat1, feat2)
        z_out = self.tanh(z)
        # print(z_out.shape)
        mm_feat = self.output_net2(z_out)
        return mm_feat
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # acoustic_input=None,      # New addition of acoustic_input
        visual_input=None,      # New addition of visual_input
        visual_inf=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        image_len=None
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # embed_pos = self.embed_positions(input_shape)
        embed_pos = self.embed_positions(input_ids)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):

            # ================================ Modifications ================================ #
            if idx in self.fusion_at_layer:
                vgg_image_features = self.image_encoder(visual_inf)


                vgg_image_features = vgg_image_features.permute(0, 2, 3, 1)
                vgg_image_features = vgg_image_features.reshape(
                    -1,
                    vgg_image_features.size()[1]*vgg_image_features.size()[2],
                    512
                    )




                trys=self.MLB2(hidden_states, vgg_image_features)
                                   visual_context=vgg_image_features)
                hidden_states=trys
                tures = self.climodel.encode_image(image_features)

                tensor_cpu = hidden_states.cpu()
                hide = tensor_cpu.detach().numpy()
                list13.append(hide)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]


            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class MultimodalBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = MultimodalBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # acoustic_input=None,      # New addition of acoustic_input
        visual_input=None,      # New addition of
        visual_inf=None,
        # image_features=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        image_len=None
    ):


        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # acoustic_input=acoustic_input,      # New addition of acoustic_input
                visual_input=visual_input,      # New addition of visual_input
                visual_inf=visual_inf,
                # image_features=image_features,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                image_len=image_len
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class MultimodalBartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = MultimodalBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # acoustic_input=None,      # New addition of acoustic_input
        visual_input=None,      # New addition of visual_input
        visual_inf=None,
        # image_features=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        image_len=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            # acoustic_input=acoustic_input,      # New addition of acoustic_input
            visual_input=visual_input,      # New addition of visual_input
            visual_inf=visual_inf,
            # image_features=image_features,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_len=image_len
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
def read_json_data(path):
    f = open(path)
    data = json.load(f)
    f.close()
    del f
    gc.collect()
    return data


# src=[]
# label=[]
# visual_feat=[]

def prepare_dataset(text_path,visual_path,image_transform,trainornot):
    data = pd.read_csv(text_path)
    path_to_images = visual_path
    src_text=data['0.para'].tolist()
    labels=data['summary'].tolist()
    # pid=data['Image_urls'].tolist()
    image_name =[]
    for i in data["images"]:
      if(str(i)!='nan'):
        i =i.replace(" ", "")
        i =i.replace("'", "")
        i = (i.split(";")[0])
        image_name.append(i.split("/")[-1])
    # print(pid)
    visual_feat=[]
    for p in image_name:
        image_path = os.path.join(path_to_images, p)
        if os.path.isfile(image_path)==True:
            # print(image_path)
            img = np.array(Image.open(image_path).convert('RGB'))
            # print(img.shape)
            img_inp=img
            img_inp = image_transform(img)
            # print(img_inp.shape)
            visual_feat.append(img_inp)
        else:
            print(image_path)
            img_inp=torch.zeros(3, 224,224)
            visual_feat.append(img_inp)






    print(visual_feat[0].shape)
    df =  pd.DataFrame(list(zip(src_text, labels,visual_feat)),columns=['src_text', 'labels','visual_feat'])
    df = df.dropna()
    return df

def pad_seq(tensor: torch.tensor,
            dim: int,
            max_len: int):
    if max_len > tensor.shape[0]:
        return torch.cat([tensor, torch.zeros(max_len - tensor.shape[0], dim)])
    else:
        return tensor[:max_len]



def preprocess_dataset(dataset):
    source=[SOURCE_PREFIX + s for s in dataset['src_text'].values.tolist()]
    model_inputs = TOKENIZER(source,
                             max_length=SOURCE_MAX_LEN,
                             padding='max_length',
                             truncation=True)


    target = [TARGET_PREFIX + t for t in dataset['labels'].values.tolist()]
    with TOKENIZER.as_target_tokenizer():
        labels = TOKENIZER(target,
                           max_length=TARGET_MAX_LEN,
                           padding='max_length',
                           truncation=True)
	    # IMP:
	    # Replace all tokenizer.pad_token_id in the labels by -100 to ignore padding tokens in the loss.
        labels['input_ids'] = [[(l if l != TOKENIZER.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs['input_ids'] = torch.tensor([i for i in model_inputs['input_ids']], dtype=torch.long, device=DEVICE)
    model_inputs['attention_mask'] = torch.tensor([a for a in model_inputs['attention_mask']], dtype=torch.long, device=DEVICE)


    model_inputs['visual_inf']=torch.stack(dataset['visual_feat'].values.tolist())
    # model_inputs['image_features']=torch.stack(dataset['image_features'].values.tolist())
    print(type(model_inputs['visual_inf']))
    for l in model_inputs['visual_inf']:
    	print(l.shape)

    model_inputs['visual_input'] = torch.stack([pad_seq(torch.tensor(vf[0], dtype=torch.float),
                                                        dim=VISUAL_DIM,
                                                        max_len=VISUAL_MAX_LEN)
                                                for vf in dataset['visual_feat'].values.tolist()], 0).to(DEVICE)
    print(type(model_inputs['visual_input']))

    model_inputs['labels'] = torch.tensor([l for l in labels['input_ids']], dtype=torch.long, device=DEVICE)


    del target
    del labels
    gc.collect()
    return model_inputs

def set_up_data_loader(text_path: str,
                       visual_path: str,image_transform,
                       trainornot):
    dataset = preprocess_dataset(prepare_dataset(text_path,
                                                 visual_path,image_transform,trainornot))
    print(dataset.keys())
    dataset = TensorDataset(dataset['input_ids'],
                            dataset['attention_mask'],
                            dataset['visual_input'],
                            dataset['labels'],
                            dataset['visual_inf'])
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )



from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

def get_scores(reference_list: list, hypothesis_list: list):
    count = 0
    met = 0
    bleu_1 = 0
    bleu_2 = 0
    bleu_3 = 0
    bleu_4 = 0
    rouge1 = 0
    rouge2 = 0
    rougel = 0
    weights_1 = (1.0,)
    weights_2 = (0.5, 0.5)
    weights_3 = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    weights_4 = (0.25, 0.25, 0.25, 0.25)

    from rouge import Rouge
    ROUGE = Rouge()

    for reference, hypothesis in zip([reference_list], [hypothesis_list]):
        print(reference)
        print(hypothesis)
        scores = ROUGE.get_scores(reference, hypothesis)
        rouge1 += scores[0]['rouge-1']['f']
        rouge2 += scores[0]['rouge-2']['f']
        rougel += scores[0]['rouge-l']['f']
        scores = ROUGE.get_scores(reference, hypothesis)
        rouge1 += scores[0]['rouge-1']['f']
        rouge2 += scores[0]['rouge-2']['f']
        rougel += scores[0]['rouge-l']['f']
        # met += meteor_score([reference],hypothesis)
        meteor_scores = meteor.compute(predictions=hypothesis, references=reference)
        print(meteor_scores)
        met += meteor_scores['meteor']

        reference = ' '.join(reference).split()
        hypothesis = ' '.join(hypothesis).split()

        bleu_1 += sentence_bleu([reference], hypothesis, weights_1)
        bleu_2 += sentence_bleu([reference], hypothesis, weights_2)
        bleu_3 += sentence_bleu([reference], hypothesis, weights_3)
        bleu_4 += sentence_bleu([reference], hypothesis, weights_4)
        count += 1

    return {
        "rouge_1": rouge1 * 100 / count,
        "rouge_2": rouge2 * 100 / count,
        "rouge_L": rougel * 100 / count,
        "bleu_1": bleu_1 * 100 / count,
        "bleu_2": bleu_2 * 100 / count,
        "bleu_3": bleu_3 * 100 / count,
        "bleu_4": bleu_4 * 100 / count,
        "meteor": met * 100 / count,
    }

def _save(model,
          output_dir: str,
          tokenizer=None,
          state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model, PreTrainedModel):
            if isinstance(unwrap_model(model), PreTrainedModel):
                if state_dict is None:
                    state_dict = model.state_dict()
                unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(output_dir, state_dict=state_dict)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)


def save_model(model,
               output_dir: str,
               tokenizer=None,
               state_dict=None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.
        Will only save from the main process.
        """
        _save(model,output_dir, tokenizer=tokenizer, state_dict=state_dict)

def train_epoch(model,
                data_loader,
                optimizer):
    model.train()
    epoch_train_loss = 0.0
    for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, attention_mask, visual_input, labels,visual_inf = batch
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        # acoustic_input=acoustic_input,
                        visual_input=visual_input,
                        visual_inf=visual_inf,
                        labels=labels)
        loss = outputs['loss']
        epoch_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    del batch
    del input_ids
    del attention_mask
    # del acoustic_input
    del visual_input
    del labels
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_train_loss/ step




def val_epoch(model,
              data_loader,
              optimizer):
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Validation Loss Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, visual_input, labels,visual_inf = batch

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            # acoustic_input=acoustic_input,
                            visual_input=visual_input,
                            visual_inf=visual_inf,
                            labels=labels)
            loss = outputs['loss']
            epoch_val_loss += loss.item()

    del batch
    del input_ids
    del attention_mask
    # del acoustic_input
    del visual_input
    del labels
    del outputs
    del loss
    gc.collect()
    torch.cuda.empty_cache()

    return epoch_val_loss/ step




def test_epoch(model,
               tokenizer,
               data_loader,
               desc,
               **gen_kwargs):
    model.eval()
    predictions = []
    gold = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc=desc)):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, visual_input, labels,visual_inf = batch

            generated_ids = model.generate(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           # acoustic_input=acoustic_input,
                                           visual_input=visual_input,
                                           visual_inf=visual_inf,
                                           **gen_kwargs)

            generated_ids = generated_ids.detach().cpu().numpy()
            generated_ids = np.where(generated_ids != -100, generated_ids, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            labels = labels.detach().cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            gold.extend(decoded_labels)
        print(len(predictions))
        print(len(gold))

    del batch
    del input_ids
    del attention_mask
    # del acoustic_input
    del visual_input
    del labels
    del generated_ids
    del decoded_preds
    del decoded_labels
    gc.collect()
    torch.cuda.empty_cache()

    return predictions, gold




def get_val_scores(model,
                   tokenizer,
                   data_loader,
                   desc,
                   epoch,
                   **gen_kwargs):
    predictions, gold = test_epoch(model,
                                   tokenizer,
                                   data_loader,
                                   desc=desc,
                                   **gen_kwargs)
    result = get_scores(predictions, gold)
    print(result)
    return result

    del predictions
    del gold
    gc.collect()
    torch.cuda.empty_cache()

	# return result

import nltk
nltk.download('wordnet')

import evaluate
meteor = evaluate.load('meteor')

def prepare_for_training(model,
                         base_learning_rate: float,
                         new_learning_rate: float,
                         weight_decay: float):
    base_params_list = []
    new_params_list = []
    for name, param in model.named_parameters():
        if "acoustic_transformer" or "visual_transformer" or "MAF_layer" in name:
            new_params_list.append(param)
        else:
            base_params_list.append(param)

    optimizer = AdamW(
        [
            {'params': base_params_list,'lr': base_learning_rate, 'weight_decay': weight_decay},
            {'params': new_params_list,'lr': new_learning_rate, 'weight_decay': weight_decay}
        ],
        lr=base_learning_rate,
        weight_decay=weight_decay
    )

    del base_params_list
    del new_params_list
    gc.collect()
    torch.cuda.empty_cache()

    return optimizer




def train(model,
          tokenizer,
          train_data_loader,
          val_data_loader,
          test_data_loader,
          base_learning_rate,
          new_learning_rate,
          weight_decay,
          **gen_kwargs):

    optimizer = prepare_for_training(model=model,
                                     base_learning_rate=base_learning_rate,
                                     new_learning_rate=new_learning_rate,
                                     weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    val_rouge_2 = []
    patience = 1

    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model,
                                 train_data_loader,
                                 optimizer)
        train_losses.append(train_loss)

        val_loss = val_epoch(model,
                             val_data_loader,
                             optimizer)
        val_losses.append(val_loss)



        get_val_scores(model,tokenizer,test_data_loader,desc="Test Generation Iteration",epoch=epoch,**gen_kwargs)


        path = MODEL_OUTPUT_DIR + "MAF_TAV_BART_epoch__epoch_" + str(epoch+1) + "_" + datetime.now().strftime('%d-%m-%Y-%H:%M')
        print(path)
        save_model(model,
                   path,
                   tokenizer)
        print("Model saved at path: ", path)


        del train_loss
        del val_loss
        del path
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    TOKENIZER = BartTokenizerFast.from_pretrained('facebook/bart-base')
    print("Tokenizer loaded...\n")
    MODEL = MultimodalBartForConditionalGeneration.from_pretrained('facebook/bart-base')
    print("Model loaded...\n")
    for name, param in MODEL.state_dict().items():
        print(name, param.size())

    MODEL.to(DEVICE)





    SOURCE_PREFIX = ''
    TARGET_PREFIX = ''



    gc.collect()

    pytorch_total_params = sum(p.numel() for p in MODEL.parameters())
    print("Total parameters: ", pytorch_total_params)
    pytorch_total_train_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    print("Total trainable parameters: ", pytorch_total_train_params)
    # input('ENter')

    for name, param in MODEL.named_parameters():
        if "acoustic_transformer" or "visual_transformer" or "MAF_layer" in name:
            print(name)



    image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

    mse_data = MSEDataModule(path_to_train, path_to_val,
                         path_to_test, path_to_images,
                         TOKENIZER, image_transform, batch_size=1)
    print('DONE===>')

    train_dataset = set_up_data_loader(path_to_train,
                                       path_to_images,image_transform,trainornot=True)
    print("\nTraining Data Loaded...")

    val_dataset = set_up_data_loader(path_to_val,path_to_images,image_transform,trainornot=False)
    print("\nValidation Data Loaded...")

    test_dataset = set_up_data_loader(path_to_test,path_to_images,image_transform,trainornot=False)
    print("\nTest Data Loaded...")
    gc.collect()

    # ------------------------------ TRAINING SETUP ------------------------------ #

    gen_kwargs = {
        'num_beams': NUM_BEAMS,
        'max_length': TARGET_MAX_LEN,
        'early_stopping': EARLY_STOPPING,
        'no_repeat_ngram_size': NO_REPEAT_NGRAM_SIZE
    }

    train(model=MODEL,
          tokenizer=TOKENIZER,
          train_data_loader=train_dataset,
          val_data_loader=val_dataset,
          test_data_loader=test_dataset,
          base_learning_rate=BASE_LEARNING_RATE,
          new_learning_rate=NEW_LEARNING_RATE,
          weight_decay=WEIGHT_DECAY,
          **gen_kwargs)

    print("Model Trained!")

len(list13)

import numpy as np

# Assuming you have a list named list13 with length 239
# Each element has a shape of (1, 1024, 768)

# Initialize an empty list to store the flattened arrays
flattened_list = []

# Loop through each element in list13 and flatten it
for item in list13:
    flattened_array = item.flatten()
    flattened_list.append(flattened_array)

# Convert the flattened_list to a NumPy array if needed
flattened_array = np.array(flattened_list)

print(flattened_array.shape)  # Output: (239, 1024 * 768)

import pandas as pd
import numpy as np

# Assuming you have the flattened_array with shape (239, 1024 * 768)

# Create a DataFrame
d = pd.DataFrame()

# Add the flattened_array as a new column named 'embedding'
d['embedding'] = list(flattened_array)

# Display the updated DataFrame
print(d)

d['embedding'][0]

df = filtered_data

df.head()

df = df.drop(['Unnamed: 0','summary','0.para','images'],axis=1)

df.head()

df = df[0:2200]

df.shape

d.shape

concatenated_df = pd.concat([d, df], axis=1)

# Display the concatenated DataFrame
print(concatenated_df)

concatenated_df['embedding'].dtype

concatenated_df['embedding'][0].dtype

concatenated_df['label'].dtype

import torch
import torch.nn as nn

class ModifiedLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ModifiedLSTMModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x.to(torch.float32))
        output, _ = self.gru1(x)
        output, _ = self.gru2(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc4(output)
        return torch.sigmoid(output)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

class MyDataset(Dataset):
    def __init__(self, embed, label):
        self.embeddings = embed.tolist()
        self.labels = label.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        embedding = torch.tensor(self.embeddings[index])
        label = torch.tensor(self.labels[index])
        return embedding, label

def create_data_loaders(dataframe, batch_size, test_size=0.4):
    train_df, test_df = train_test_split(dataframe, test_size=test_size, shuffle=False)
    X_train, y_train = train_df['embedding'], train_df['label']
    X_test, y_test = test_df['embedding'], test_df['label']

    oversampler = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train.values.reshape(-1, 1), y_train)

    train_dataset = MyDataset(X_train_resampled, y_train_resampled)
    test_dataset = MyDataset(X_test, y_test)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_data_loader, test_data_loader

# Assuming you have your concatenated_df DataFrame defined
train_data_loader, test_data_loader = create_data_loaders(data_no_duplicates, 32)

from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

num_epochs = 5

# Create the LSTM model
model = ModifiedLSTMModel(786432, 256, 1)

# Define the loss function
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Prepare your training data (train_data_loader)

# Train the model
model.train()
for epoch in range(num_epochs):
    for inputs, targets in tqdm(train_data_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()

import torch
import torch.nn as nn
import numpy as np

def evaluate(model, data_loader, device):
    model.eval()
    total_samples = 0
    correct_predictions = 0
    predicted_labels_list = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            predicted_labels = (torch.sigmoid(outputs) > 0.5).squeeze().long()

            total_samples += targets.size(0)
            correct_predictions += (predicted_labels == targets).sum().item()

            predicted_labels_list.append(predicted_labels.cpu().numpy())

    accuracy = correct_predictions / total_samples
    predicted_labels_array = np.concatenate(predicted_labels_list)
    return accuracy, predicted_labels_array

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Evaluate the model on the test data and get the predicted labels
accuracy, predicted_labels = evaluate(model, train_data_loader, device)
print(f"Accuracy: {accuracy:.4f}")

from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

num_epochs = 19

# Create the LSTM model
model = ModifiedLSTMModel(786432, 256, 1)

# Define the loss function
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Prepare your training data (train_data_loader)

# Train the model
model.train()
for epoch in range(num_epochs):
    for inputs, targets in tqdm(train_data_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()

X = np.array(concatenated_df['embedding'].tolist())

X

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


data_df = concatenated_df

# Splitting data into features (embeddings) and labels
X = np.array(data_df['embedding'].tolist())
y = np.array(data_df['label'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize a Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear', random_state=42)

# Train the classifier on the scaled training data
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your DataFrame with embeddings and labels
# Assuming your DataFrame is named 'data_df' with columns 'embedding' and 'label'
# For example:
data_df = concatenated_df

# Splitting data into features (embeddings) and labels
X = np.array(data_df['embedding'].tolist())
y = np.array(data_df['label'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize a Random Forest classifier
# You can adjust the n_estimators, max_depth, and other hyperparameters as needed
classifier = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

# Train the classifier on the scaled training data
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your DataFrame with embeddings and labels
# Assuming your DataFrame is named 'data_df' with columns 'embedding' and 'label'
# For example:
data_df = concatenated_df

# Splitting data into features (embeddings) and labels
X = np.array(data_df['embedding'].tolist())
y = np.array(data_df['label'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize an MLP classifier
# You can adjust the hidden_layer_sizes, activation, solver, and other hyperparameters as needed
classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)

# Train the classifier on the scaled training data
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

y_pred

