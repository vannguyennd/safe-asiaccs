import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from modelGNN_updates import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = outputs
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class ModelCheck(nn.Module):
    def __init__(self, encoder, config, tokenizer, args, teacherA, teacherB):
        super(ModelCheck, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.dense_project = nn.ReLU(nn.Linear(self.args.roberta_embedding, self.args.hidden_size_projection))
        self.classifier_roberta = ROBERTAClassifierS(args, input_size=self.args.roberta_embedding)
        self.classifier_dia = ROBERTAClassifier(args, input_size=self.args.roberta_embedding)
        self.classifier_dib = ROBERTAClassifier(args, input_size=self.args.roberta_embedding)
        self.teacherA = teacherA
        self.teacherB = teacherB
        self.distillation_loss = torch.nn.KLDivLoss()

    def cl_dot(self, outputs, targets):
        # cl on outputs_roberta (optional)
        selected_rs = outputs
        selected_rst = torch.t(selected_rs)

        selected_encodes_uv = torch.matmul(selected_rs, selected_rst)
        matrix_dot_lts = torch.divide(selected_encodes_uv, self.args.tau_tempt)

        y_train_c = torch.reshape(targets, [targets.shape[0], 1])
        y_train_ct = torch.tensor(torch.t(y_train_c))
        mask_p = torch.eq(y_train_c, y_train_ct)
        mask_pf = mask_p.to(torch.float)

        cardinality_p = torch.sum(mask_pf, dim=-1, keepdim=True)

        mask_a = torch.eye(y_train_c.shape[0]).to(torch.float)
        mask_a = 1.0 - mask_a

        max_matrix_dot_lts_dim = torch.max(matrix_dot_lts, dim=-1, keepdim=True)
        matrix_dot_lts_update = matrix_dot_lts - max_matrix_dot_lts_dim[0]

        exp_matrix_dot = torch.exp(matrix_dot_lts_update)
        matrix_a = torch.mul(exp_matrix_dot.to(device), mask_a.to(device))
        matrix_a_sum = torch.sum(matrix_a, dim=-1, keepdim=True)
        log_prob = torch.log(matrix_dot_lts + 1e-10) - (max_matrix_dot_lts_dim[0] + torch.log(matrix_a_sum + 1e-10))

        modified_cl = torch.sum(torch.mul(log_prob, mask_pf), dim=-1, keepdim=True) * y_train_c
        mean_log_prob_p = torch.mean(modified_cl / cardinality_p)
        additional_loss = -self.args.trade_off_cl * mean_log_prob_p

        return additional_loss

    def forward(self, input_ids_sb=None, masks_sb=None, labels=None, input_ids_a=None, masks_a=None):
        outputs_roberta = self.encoder.roberta(input_ids_sb, attention_mask=input_ids_sb.ne(1))[0]
        outputs_roberta_cls = outputs_roberta[:, 0, :]
        outputs_roberta_cls = self.dense_project(outputs_roberta_cls)

        self.classifier_roberta.load_state_dict(self.encoder.classifier.state_dict())
        logits = self.classifier_roberta(outputs_roberta_cls)
        prob = F.sigmoid(logits)

        """
        get outputs from teacher A and B
        """
        teacherA_logits = self.teacherA(input_ids_a, masks_a)
        teacherB_logits = self.teacherB(input_ids_sb, masks_sb)

        locs_a = (input_ids_sb==self.tokenizer.dia_token_id).nonzero(as_tuple=True)
        locs_a = locs_a[1].tolist()
        outputs_disa = []
        for i in range(len(locs_a)):
            outputs_disa.append(outputs_roberta[i, locs_a[i], :].tolist())
        outputs_disa = torch.tensor(outputs_disa).to(device)
        outputs_disa = self.classifier_dia(outputs_disa)

        locs_b = (input_ids_sb==self.tokenizer.dib_token_id).nonzero(as_tuple=True)
        locs_b = locs_b[1].tolist()
        outputs_disb = []
        for i in range(len(locs_b)):
            outputs_disb.append(outputs_roberta[i, locs_b[i], :].tolist())
        outputs_disb = torch.tensor(outputs_disb).to(device)
        outputs_disb = self.classifier_dib(outputs_disb)

        distillation_loss_ta = self.distillation_loss(F.softmax(teacherA_logits / self.args.distill_tempt, dim=1), F.softmax(outputs_disa / self.args.distill_tempt, dim=1),) * \
                        self.args.distill_tempt**2
        distillation_loss_tb = self.distillation_loss(F.softmax(teacherB_logits / self.args.distill_tempt, dim=1), F.softmax(outputs_disb / self.args.distill_tempt, dim=1),) * \
                        self.args.distill_tempt**2

        if labels is not None:
            labels = labels.float()
            loss_cls = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss_cls = -loss_cls.mean()
            if self.args.trade_off_cl != -1.0:
                loss_cl = self.cl_dot(outputs_roberta_cls, labels)
            else:
                loss_cl = 0.0
            
            tradeoff_teacheara = (1 - self.args.student_alpha)*self.args.teacher_beta
            tradeoff_teacherb = 1 - (self.args.student_alpha + tradeoff_teacheara)
            loss = self.args.student_alpha*(loss_cls + loss_cl) + tradeoff_teacheara*distillation_loss_ta + tradeoff_teacherb*distillation_loss_tb

            return loss, prob, F.softmax(outputs_disa), F.softmax(outputs_disb)
        else:
            return prob, F.softmax(outputs_disa), F.softmax(outputs_disb)


class ROBERTAClassifierS(torch.nn.Module):
    def __init__(self, args, dropout_rate=0.1, input_size=None):
        super(ROBERTAClassifierS, self).__init__()

        self.dense = torch.nn.Linear(input_size, input_size, bias=True)
        self.dropout = torch.nn.Dropout(dropout_rate, inplace=False)
        self.out_proj = torch.nn.Linear(input_size, args.num_classes_s, bias=True)

    def forward(self, features):
        x = features
        x = self.dense(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class ROBERTAClassifier(torch.nn.Module):
    def __init__(self, args, dropout_rate=0.1, input_size=None):
        super(ROBERTAClassifier, self).__init__()

        self.dense = torch.nn.Linear(input_size, input_size, bias=True)
        self.dropout = torch.nn.Dropout(dropout_rate, inplace=False)
        self.out_proj = torch.nn.Linear(input_size, args.num_classes, bias=True)

    def forward(self, features):
        x = features
        x = self.dense(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x
