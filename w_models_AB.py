import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from w_models_upAB import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=outputs
        prob=F.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob


class PredictionClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args, input_size=None):
        super().__init__()
        # self.dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        if input_size is None:
            input_size = args.hidden_size_ta
        self.dense = nn.Linear(input_size, args.hidden_size_ta)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(args.hidden_size_ta, args.num_classes)

    def forward(self, features):  #
        x = features
        x = self.dropout(x)
        x = self.dense(x.float())
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class GNNGVD(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(GNNGVD, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.tokenizer = tokenizer
        if args.gnn == "ReGGNN":
            self.gnn = ReGGNN(feature_dim_size=args.feature_dim_size,
                                hidden_size=args.hidden_size_ta,
                                num_GNN_layers=args.num_GNN_layers,
                                dropout=config.hidden_dropout_prob,
                                residual=not args.remove_residual,
                                att_op=args.att_op)
        else:
            self.gnn = ReGCN(feature_dim_size=args.feature_dim_size,
                               hidden_size=args.hidden_size_ta,
                               num_GNN_layers=args.num_GNN_layers,
                               dropout=config.hidden_dropout_prob,
                               residual=not args.remove_residual,
                               att_op=args.att_op)
        gnn_out_dim = self.gnn.out_dim
        self.classifier = PredictionClassification(config, args, input_size=gnn_out_dim)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids=None, att_masks=None, labels=None):
        # construct graph
        if self.args.format == "uni":
            adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings, window_size=self.args.window_size)
        else:
            adj, x_feature = build_graph_text(input_ids.cpu().detach().numpy(), self.w_embeddings, window_size=self.args.window_size)
        # initilizatioin
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)
        # run over GNNs
        outputs = self.gnn(adj_feature.to(device).double(), adj.to(device).double(), adj_mask.to(device).double())
        logits = self.classifier(outputs)
        """
        prob = F.sigmoid(logits)
        """
        prob = F.softmax(logits)
        if labels is not None:
            """
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            """
            loss = self.loss_fct(logits, labels)
            loss = loss.mean()
            return loss, prob
        else:
            return prob


class TextCNNVed(nn.Module):
    def __init__(self, encoder, static=False, dropout=0.1, kernel_num=100, kernel_sizes=[3,4,5], embed_dim=768, class_num=2):
        super(TextCNNVed, self).__init__()
        self.dropout = dropout
        self.encoder = encoder
        self.static = static
        self.dropout = dropout
        Co = kernel_num
        Ks = kernel_sizes
        
        D = embed_dim
        Ci = 1
        C = class_num

        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, input_ids, attention_masks=None, labels=None, build_loss_fn=None):
        x = self.encoder.roberta.embeddings(input_ids)  # (N, W, D)
        x = x * attention_masks.unsqueeze(-1)
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logits = self.fc1(x)  # (N, C)

        if labels is not None:
            if build_loss_fn is not None:
                loss = build_loss_fn(logits, labels)
            else:
                loss = self.loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


# Model with classifier layers on top of RoBERTa
class ROBERTAClassifier(torch.nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ROBERTA(torch.nn.Module):
    def __init__(self, model, num_classes=2):
        super(ROBERTA, self).__init__()
        self.model = model
        self.classes = num_classes
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.classifier_bert = ROBERTAClassifier(num_classes, input_size=768)

    def forward(self, input_ids, attention_masks, labels=None):
        outputs = self.model.roberta(input_ids, attention_mask=attention_masks)[0][:,0,:]
        logits = self.classifier_bert(outputs)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.classes), labels.long().view(-1))
            return loss, logits
        else:
            return logits


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
