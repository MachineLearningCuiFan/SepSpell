import warnings

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput, \
    QuestionAnsweringModelOutput, TokenClassifierOutput

from transformers import BertConfig
from models.fusion_embedding_own import FusionBertEmbeddings
from models.classifier import BertMLP
from models.modeling_glycebert import GlyceBertModel


class GlyceBertModelOwn(BertModel):
    def __init__(self, config):
        super(GlyceBertModelOwn, self).__init__(config)
        self.config = config
        self.embeddings = FusionBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(
        self,
        input_ids_mask=None,
        input_ids=None,
        pinyin_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the models is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the models is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids_mask=input_ids_mask,input_ids=input_ids, pinyin_ids=pinyin_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



##########  Correction_model  #############
class GlyceBertForCsc(BertPreTrainedModel):
    def __init__(self, config):
        super(GlyceBertForCsc, self).__init__(config)
        self.bert = GlyceBertModelOwn(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = BertOnlyMLMHead(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.classifier.predictions.decoder

    def forward(
        self,
        input_ids_mask=None,
        input_ids=None,
        pinyin_ids=None,
        attention_mask=None,
        label_ids=None,
        loss_mask=None,
        ):
        hidden_states = self.bert(input_ids_mask=input_ids_mask,
                                  input_ids=input_ids,
                                  pinyin_ids=pinyin_ids,
                                  attention_mask=attention_mask)
        sequence_output = hidden_states[0]
        bert_sequence_output = self.dropout(sequence_output)
        logits = self.classifier(bert_sequence_output)
        output = (logits,)  # add hidden states and attention if they are here

        loss = None
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if loss_mask is not None:
                active_loss = loss_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.vocab_size)[active_loss]
                active_labels = label_ids.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config.num_labels), label_ids.view(-1))

        return ((loss,) + output) if loss is not None else output



######  Detection_model  ##############
class GlyceBertDetectionForCsc(nn.Module):
    def __init__(self,args):
        super(GlyceBertDetectionForCsc, self).__init__()
        self.args = args
        if args.model_name_or_path:
            self.glyceBert = GlyceBertModel.from_pretrained(args.model_name_or_path)
            self.config = BertConfig.from_pretrained(args.model_name_or_path)
        else:
            self.config = BertConfig.from_pretrained(args.model_name_or_path)
            self.glyceBert = GlyceBertModel(config=self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.dense_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        # gru 配置
        # self.gru = nn.GRU(input_size=self.config.hidden_size,hidden_size=self.config.hidden_size//2,bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        self.activation = nn.Tanh()

    def forward(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            label_ids=None
    ):
        hidden_states = self.glyceBert(
                                  input_ids=input_ids,
                                  pinyin_ids=pinyin_ids,
                                  attention_mask=attention_mask)
        sequence_output = hidden_states[0]
        sequence_output = self.dropout(sequence_output)
        # gru_out, _ = self.gru(sequence_output)
        # gru_out = self.dropout(gru_out)
        sequence_output = self.dense_layer(sequence_output)
        sequence_output = self.activation(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0,5.0],device=logits.device))
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, 2)[active_loss]
                active_labels = label_ids.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))

        output = (logits,) + hidden_states[1:]
        return ((loss,) + output) if loss is not None else output