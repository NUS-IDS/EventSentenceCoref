import torch
import logging
import torch.nn as nn
from transformers import BertModel, BertConfig, AlbertConfig, AlbertModel
from sklearn.metrics import adjusted_rand_score


class BERTRepresenter(nn.Module):
    def __init__(self, bert_model='bert-base-cased'):
        super(BERTRepresenter, self).__init__()
        if 'albert' in bert_model.lower():
            config = AlbertConfig()
            self.bert = AlbertModel(config).from_pretrained(bert_model)
        else:
            config = BertConfig()
            # config = BertConfig(vocab_size=24000, hidden_size=264)
            self.bert = BertModel(config).from_pretrained(bert_model)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logging.debug(f'outputs:{outputs}')
        return outputs


class LSTMRepresenter(nn.Module):
    def __init__(self, embedding_dim=768, lstm_out_dim=64, use_pos=False, dropout=0.1, device='cpu'):
        super(LSTMRepresenter, self).__init__()
        self.lstm = nn.LSTM(
            embedding_dim+61 if use_pos else embedding_dim,
            lstm_out_dim//2,
            bidirectional=True,
            dropout=dropout
        )
        self.device = device
        self.use_pos = use_pos
        self.flatten = nn.Flatten()

    def forward(self, output, pair_pos):
        logging.debug(f'bert output.shape:{output.shape}') # bert output.shape:torch.Size([16, 183, 768])
        # bert last_hidden_state gives (batch_size, seq_len, hidden_size), where seq_len is num tokens from BertTokenizer
        # lstm expects inputs of (seq_len, batch_size, features)
        if self.use_pos:
            output = torch.cat([
                output.type(torch.FloatTensor), 
                pair_pos.type(torch.FloatTensor)
                ], dim=2).to(self.device)
            logging.debug(f'cat output.shape:{output.shape}') # bert permute output.shape:torch.Size([183, 16, 768])    
        output = output.permute(1, 0, 2)
        logging.debug(f'bert permute output.shape:{output.shape}')
        output, (ht, ct) = self.lstm(output)
        logging.debug(f'lstm output.shape:{output.shape}')
        output = self.flatten(output.permute(1, 0, 2))
        logging.debug(f'flatten output.shape:{output.shape}')
        return output


class BiLSTMPairClassifier(nn.Module):

    def __init__(
            self, seq_len=500, n_classes=2, lstm_out_dim=64, 
            use_bilstm=True, use_pos=True, use_info=True, use_suj=False,
            bert_model='bert-base-cased', dropout_rate=0.3, 
            hidden_dim=200, device='cpu'):
        super(BiLSTMPairClassifier, self).__init__()
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.lstm_out_dim = lstm_out_dim
        self.use_bilstm = use_bilstm
        self.use_info = use_info
        self.use_pos = use_pos
        self.use_suj = use_suj
        self.device = device
        self.bert_model = bert_model
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

        self.bert = BERTRepresenter(bert_model=bert_model)
        self.embedding_dim = self.bert.bert.config.hidden_size

        
        if use_bilstm:
            self.lstm = LSTMRepresenter(
                lstm_out_dim=lstm_out_dim, 
                embedding_dim=self.embedding_dim,
                use_pos=self.use_pos,
                dropout=dropout_rate,
                device=self.device
                )
            self.linear = nn.Linear(lstm_out_dim*seq_len, hidden_dim)
        else:
            self.linear = nn.Linear(self.embedding_dim, hidden_dim)
        
        if use_info:
            hidden_dim += 6
        if use_suj:
            hidden_dim += 27
        self.out = nn.Linear(hidden_dim, n_classes)
        
        self.drop = nn.Dropout(p=dropout_rate)


    def forward(self, input_ids, pair_infos=None, pair_pos=None, attention_mask=None, token_type_ids=None):
        """
        input_ids.shape: torch.Size([16, 200]) 
        --> (batch_size, max_len)
        """
        logging.debug(f'input_ids.shape: {input_ids.shape}')
        batch_size, max_len = input_ids.shape
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        if self.use_bilstm:
            output = self.lstm(
                output=output['last_hidden_state'], 
                pair_pos=pair_pos)
        else:
            output = output['pooler_output']
            logging.debug(f'bert output.shape:{output.shape}')
        output = self.drop(output)
        logging.debug(f'dropout output.shape:{output.shape}')
        output = self.linear(output)
        logging.debug(f'linear output.shape:{output.shape}')
        if self.use_info:
            output = torch.cat([
                pair_infos.type(torch.FloatTensor), 
                output.type(torch.FloatTensor)
                ], dim=1).to(self.device)
            logging.debug(output)
            logging.debug(f'cat output.shape:{output.shape}')
        output = self.out(output)
        logging.debug(f'linear output.shape:{output.shape}')
        if batch_size==1:
            output = output.view(1,-1)
            logging.debug(f'single bs output.shape:{output.shape}')
        return output
