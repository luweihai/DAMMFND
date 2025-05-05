import os
import tqdm
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D
from transformers import BertModel
import torch.nn as nn
import models_mae
from utils.utils import data2gpu, Averager, metrics, Recorder, clipdata2gpu
from utils.utils import metricsTrueFalse
from .layers import *
from .pivot import *
from timm.models.vision_transformer import Block
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models




class DomainAwareTransformer(nn.Module):
    def __init__(self, dim=512, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.global_proj = nn.Linear(dim, dim)
        self.weight_proj = nn.Linear(dim, 3)
        
        # Add & Norm layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, modality_reps, domain_rep):
        batch_size = domain_rep.shape[0]
        
        # global features
        global_rep = torch.mean(torch.stack(modality_reps, dim=1), dim=1)
        global_rep = self.global_proj(global_rep)
        
        #  domain features plus global features
        q = self.q_proj(domain_rep + global_rep).view(batch_size, self.num_heads, self.head_dim)
        
        # 
        k = torch.stack([self.k_proj(rep) for rep in modality_reps], dim=1)
        v = torch.stack([self.v_proj(rep) for rep in modality_reps], dim=1)
        
        k = k.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        
        # multihead
        attn = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).squeeze(2)
        out = out.reshape(batch_size, -1)

        # Add & Norm 
        domain_rep = domain_rep + self.dropout(out)
        domain_rep = self.norm1(domain_rep)
        
        # FFN
        ffn_output = self.ffn(domain_rep)
        
        # Add & Norm
        domain_rep = domain_rep + self.dropout(ffn_output)
        domain_rep = self.norm2(domain_rep)
        
        # final weights
        weights = self.weight_proj(domain_rep)
        weights = F.softmax(weights, dim=-1)
        
        return weights


class DAMMFNDMODEL(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout):
        super(DAMMFNDMODEL, self).__init__()
        self.num_expert = 6
        self.task_num = 2
        # self.domain_num = 9
        self.domain_num = self.task_num
        self.gate_num = 3
        self.num_share = 1
        self.unified_dim, self.text_dim = emb_dim, 768
        self.image_dim = 768
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.text_token_len = 197
        self.image_token_len = 197

        text_expert_list = []
        for i in range(self.domain_num):
            text_expert = []
            for j in range(self.num_expert):
                text_expert.append(cnn_extractor(emb_dim, feature_kernel))

            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
        self.text_experts = nn.ModuleList(text_expert_list)

        image_expert_list = []
        for i in range(self.domain_num):
            image_expert = []
            for j in range(self.num_expert):
                image_expert.append(cnn_extractor(self.image_dim, feature_kernel))
            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)
        self.image_experts = nn.ModuleList(image_expert_list)

        fusion_expert_list = []
        for i in range(self.domain_num):
            fusion_expert = []
            for j in range(self.num_expert):
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       nn.Linear(320, 320),
                                       )
                fusion_expert.append(expert)
            fusion_expert = nn.ModuleList(fusion_expert)
            fusion_expert_list.append(fusion_expert)
        self.fusion_experts = nn.ModuleList(fusion_expert_list)

        final_expert_list = []
        for i in range(self.domain_num):
            final_expert = []
            for j in range(self.num_expert):
                final_expert.append(Block(dim=320, num_heads=8))
            final_expert = nn.ModuleList(final_expert)
            final_expert_list.append(final_expert)
        self.final_experts = nn.ModuleList(final_expert_list)

        text_share_expert, image_share_expert, fusion_share_expert, final_share_expert = [], [], [], []
        for i in range(self.num_share):
            text_share = []
            image_share = []
            fusion_share = []
            final_share = []
            for j in range(self.num_expert * 2):
                text_share.append(cnn_extractor(emb_dim, feature_kernel))
                image_share.append(cnn_extractor(self.image_dim, feature_kernel))
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       nn.Linear(320, 320),
                                       )
                fusion_share.append(expert)
                final_share.append(Block(dim=320, num_heads=8))
            text_share = nn.ModuleList(text_share)
            text_share_expert.append(text_share)
            image_share = nn.ModuleList(image_share)
            image_share_expert.append(image_share)
            fusion_share = nn.ModuleList(fusion_share)
            fusion_share_expert.append(fusion_share)
            final_share = nn.ModuleList(final_share)
            final_share_expert.append(final_share)
        self.text_share_expert = nn.ModuleList(text_share_expert)
        self.image_share_expert = nn.ModuleList(image_share_expert)
        self.fusion_share_expert = nn.ModuleList(fusion_share_expert)
        self.final_share_expert = nn.ModuleList(final_share_expert)

        image_gate_list, text_gate_list, fusion_gate_list, fusion_gate_list0, final_gate_list = [], [], [], [], []
        for i in range(self.domain_num):
            image_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.unified_dim, self.num_expert * 3),
                                       nn.Dropout(0.1),
                                       nn.Softmax(dim=1)
                                       )
            text_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.unified_dim, self.num_expert * 3),
                                      nn.Dropout(0.1),
                                      nn.Softmax(dim=1)
                                      )
            fusion_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                        nn.SiLU(),
                                        nn.Linear(self.unified_dim, self.num_expert * 4),
                                        nn.Dropout(0.1),
                                        nn.Softmax(dim=1)
                                        )
            fusion_gate0 = nn.Sequential(nn.Linear(320, 160),
                                         nn.SiLU(),
                                         nn.Linear(160, self.num_expert * 3),
                                         nn.Dropout(0.1),
                                         nn.Softmax(dim=1)
                                         )
            final_gate = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       nn.Linear(320, 160),
                                       nn.SiLU(),
                                       nn.Linear(160, self.num_expert * 3),
                                       nn.Dropout(0.1),
                                       nn.Softmax(dim=1)
                                       )
            image_gate_list.append(image_gate)
            text_gate_list.append(text_gate)
            fusion_gate_list.append(fusion_gate)
            fusion_gate_list0.append(fusion_gate0)
            final_gate_list.append(final_gate)
        self.image_gate_list = nn.ModuleList(image_gate_list)
        self.text_gate_list = nn.ModuleList(text_gate_list)
        self.fusion_gate_list = nn.ModuleList(fusion_gate_list)
        self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0)
        self.final_gate_list = nn.ModuleList(final_gate_list)

        self.text_attention = MaskAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.fusion_attention = TokenAttention(self.unified_dim * 2)
        self.final_attention = TokenAttention(320)

        self.text_classifier = MLP(320, mlp_dims, dropout)
        self.text_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)
        self.image_classifier = MLP(320, mlp_dims, dropout)
        self.image_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)
        self.fusion_classifier = MLP(320, mlp_dims, dropout)
        self.fusion_classifier_Mu = MLP_Mu(320, mlp_dims, dropout)

        self.max_classifier = MLP(320 * 1, mlp_dims, dropout)


        self.domain_aware_text_classifier = MLP(320 * 1, mlp_dims, dropout)
        self.domain_aware_image_classifier = MLP(320 * 1, mlp_dims, dropout)
        self.domain_aware_fusion_classifier = MLP(320 * 1, mlp_dims, dropout)

        self.domain_aware_total_classifier = MLP(320 * 1, mlp_dims, dropout)


        share_classifier_list = []

        for i in range(self.domain_num):
            share_classifier = MLP(320, mlp_dims, dropout)
            share_classifier_list.append(share_classifier)
        self.share_classifier_list = nn.ModuleList(share_classifier_list)

        dom_classifier_list = []

        for i in range(self.domain_num):
            dom_classifier = MLP(320, mlp_dims, dropout)
            dom_classifier_list.append(dom_classifier)
        self.dom_classifier_list = nn.ModuleList(dom_classifier_list)

        final_classifier_list = []

        for i in range(self.domain_num):
            final_classifier = MLP(320, mlp_dims, dropout)
            final_classifier_list.append(final_classifier)
        self.final_classifier_list = nn.ModuleList(final_classifier_list)

        self.MLP_fusion = MLP_fusion(960, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(320, 320, [348], 0.1)
        self.MLP_fusion0 = MLP_fusion(768 * 2, 768, [348], 0.1)
        self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)
        self.att_mlp_text = MLP_fusion(320, 2, [174], 0.1)
        self.att_mlp_img = MLP_fusion(320, 2, [174], 0.1)
        self.att_mlp_mm = MLP_fusion(320, 2, [174], 0.1)



        self.model_size = "base"
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        self.image_model.cuda()
        checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)
        for param in self.image_model.parameters():
            param.requires_grad = False


        self.ClipModel, _ = load_from_name("ViT-B-16", device="cuda", download_root='./')

        self.fake_news_layernorm = LayerNorm(320 * 3, eps=1e-12)
        self.domain_classification_layernorm = LayerNorm(320 * 1, eps=1e-12)
        self.gate_trans = nn.Sequential(
            nn.Linear(320 * 1, 3 * 320, bias=False),
            nn.GELU(),
            nn.Linear(3 * 320, 320 * 1, bias=False),
            nn.GELU(),
        )

        self.query_text = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 1, bias=False)
        )
        self.query_image = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 1, bias=False)
        )
        self.query_fusion = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 320),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 320),
            nn.Sigmoid()
        )

        self.gate_fusion_prefer = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 320),
            nn.Sigmoid()
        )

        self.attention = DomainAwareTransformer(dim=320, num_heads=8)


        self.tau = 0.5

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        text_feature = self.bert(inputs, attention_mask=masks)[0]  # ([64, 197, 768])
        image = kwargs['image']
        image_feature = self.image_model.forward_ying(image)  # ([64, 197, 768])
        clip_image = kwargs['clip_image']
        clip_text = kwargs['clip_text']
        with torch.no_grad():
            clip_image_feature = self.ClipModel.encode_image(clip_image)  # ([64, 512])
            clip_text_feature = self.ClipModel.encode_text(clip_text)  # ([64, 512])
            clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)
            clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True)
        clip_fusion_feature = torch.cat((clip_image_feature, clip_text_feature), dim=-1)  # torch.Size([64, 1024])
        clip_fusion_feature = self.clip_fusion(clip_fusion_feature.float())  # torch.Size([64, 320])

        text_atn_feature = self.text_attention(text_feature, masks)
        image_atn_feature, _ = self.image_attention(image_feature)
        fusion_feature = torch.cat((image_feature, text_feature), dim=-1)
        fusion_atn_feature, _ = self.fusion_attention(fusion_feature)  # ([64, 1536])
        fusion_atn_feature = self.MLP_fusion0(fusion_atn_feature)

        text_gate_input = text_atn_feature  # ([64, 1536])
        image_gate_input = image_atn_feature
        fusion_gate_input = fusion_atn_feature


        # Multi-view Features Extraction and Aggregation

        text_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.text_gate_list[i](text_gate_input)
            text_gate_out_list.append(gate_out)
        self.text_gate_out_list = text_gate_out_list

        image_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.image_gate_list[i](image_gate_input)
            image_gate_out_list.append(gate_out)
        self.image_gate_out_list = image_gate_out_list

        fusion_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.fusion_gate_list[i](fusion_gate_input)
            fusion_gate_out_list.append(gate_out)
        self.fusion_gate_out_list = fusion_gate_out_list

        text_gate_expert_value = []
        text_experts_feature = 0
        text_gate_share_expert_value = []
        for i in range(1):
            gate_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.text_experts[i][j](text_feature)  # ([64, 320])
                gate_expert += (tmp_expert * text_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert * 2):
                tmp_expert = self.text_share_expert[0][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
            text_experts_feature = gate_expert
            text_gate_share_expert_value.append(gate_share_expert)

        att = F.softmax(self.att_mlp_text(text_experts_feature), dim=-1)
        text_experts_feature0 = att[:, 0].view(-1, 1)*text_experts_feature
        text_experts_feature1 = att[:, 1].view(-1, 1)*text_experts_feature
        text_gate_expert_value.append(text_experts_feature0)
        text_gate_expert_value.append(text_experts_feature1)


        image_gate_expert_value = []
        image_experts_feature = 0
        image_gate_share_expert_value = []
        for i in range(1):
            gate_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.image_experts[i][j](image_feature)  # ([64, 320])
                gate_expert += (tmp_expert * image_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert * 2):
                tmp_expert = self.image_share_expert[0][j](image_feature)
                gate_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
            image_experts_feature = gate_expert
            image_gate_share_expert_value.append(gate_share_expert)

        att = F.softmax(self.att_mlp_img(image_experts_feature), dim=-1)
        image_experts_feature0 = att[:, 0].view(-1, 1)*image_experts_feature
        image_experts_feature1 = att[:, 1].view(-1, 1)*image_experts_feature
        image_gate_expert_value.append(image_experts_feature0)
        image_gate_expert_value.append(image_experts_feature1)


        # clip_fusion_feature
        # fusion

        text = text_gate_share_expert_value[0]
        image = image_gate_share_expert_value[0]
        fusion_share_feature = torch.cat((clip_fusion_feature, text, image), dim=-1)

        fusion_share_feature = self.MLP_fusion(fusion_share_feature)
        fusion_gate_input0 = self.domain_fusion(fusion_share_feature)
        fusion_gate_out_list0 = []
        for k in range(self.domain_num):
            gate_out = self.fusion_gate_list0[k](fusion_gate_input0)
            fusion_gate_out_list0.append(gate_out)
        self.fusion_gate_out_list0 = fusion_gate_out_list0

        fusion_gate_expert_value0 = []
        fusion_experts_feature = 0
        fusion_gate_share_expert_value0 = []
        for m in range(1):
            share_gate_expert0 = 0
            gate_spacial_expert = 0
            gate_share_expert = 0
            for n in range(self.num_expert):
                fusion_tmp_expert0 = self.fusion_experts[m][n](fusion_share_feature)
                share_gate_expert0 += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, n].unsqueeze(1))
            for n in range(self.num_expert * 2):
                fusion_tmp_expert0 = self.fusion_share_expert[0][n](fusion_share_feature)
                share_gate_expert0 += (
                            fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
                gate_share_expert += (
                            fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
            #fusion_gate_expert_value0.append(share_gate_expert0)
            fusion_gate_share_expert_value0.append(gate_share_expert)
            fusion_experts_feature = fusion_tmp_expert0

        att = F.softmax(self.att_mlp_mm(fusion_experts_feature), dim=-1)
        fusion_experts_feature0 = att[:, 0].view(-1, 1)*fusion_experts_feature
        fusion_experts_feature1 = att[:, 1].view(-1, 1)*fusion_experts_feature
        fusion_gate_expert_value0.append(fusion_experts_feature0)
        fusion_gate_expert_value0.append(fusion_experts_feature1)


        # text
        text_two_task = []
        image_two_task = []
        fusion_two_task = []
        text_two_task.append(self.text_classifier(text_gate_expert_value[0]).squeeze(1))
        text_two_task.append(self.text_classifier_Mu(text_gate_expert_value[1]).squeeze(1))
        image_two_task.append(self.image_classifier(image_gate_expert_value[0]).squeeze(1))
        image_two_task.append(self.image_classifier_Mu(image_gate_expert_value[1]).squeeze(1))
        fusion_two_task.append(self.fusion_classifier(fusion_gate_expert_value0[0]).squeeze(1))
        fusion_two_task.append(self.fusion_classifier_Mu(fusion_gate_expert_value0[1]).squeeze(1))

        
        
        # Domain Disentanglement
        text_fake_news = torch.softmax(text_two_task[0],-1)
        image_fake_news = torch.softmax(image_two_task[0],-1)
        fusion_fake_news = torch.softmax(fusion_two_task[0],-1)


        text_multi_domain = torch.softmax(text_two_task[1], -1) # [64, 9]
        image_multi_domain = torch.softmax(image_two_task[1], -1)
        fusion_multi_domain = torch.softmax(fusion_two_task[1], -1)


        multi_label_feature = text_gate_expert_value[0] + image_gate_expert_value[0] + fusion_gate_expert_value0[0]
        fake_news_feature = text_gate_expert_value[1] + image_gate_expert_value[1] + fusion_gate_expert_value0[1]


        # Domain-Aware Multi-View Discriminator
        text_domain_features = text_gate_expert_value[0]
        image_domain_features = image_gate_expert_value[0]
        fusion_domain_features = fusion_gate_expert_value0[0]

        text_domain_features = self.gate_text_prefer(fake_news_feature) * text_domain_features
        image_domain_features = self.gate_image_prefer(fake_news_feature) * image_domain_features
        fusion_domain_features = self.gate_fusion_prefer(fake_news_feature) * fusion_domain_features

        domain_aware_text_view = torch.sigmoid(self.domain_aware_text_classifier(text_gate_expert_value[0] + text_domain_features).squeeze())
        domain_aware_image_view = torch.sigmoid(self.domain_aware_image_classifier(image_gate_expert_value[0] + image_domain_features).squeeze())
        domain_aware_fusion_view = torch.sigmoid(self.domain_aware_fusion_classifier(fusion_gate_expert_value0[0] + fusion_domain_features).squeeze())

        # Domain-Enhanced Multi-view Decision Layer 
        weight_common = self.attention([text_gate_expert_value[0], image_gate_expert_value[0], fusion_gate_expert_value0[0]], multi_label_feature)

        fake_news_sigmoid = weight_common[:, 0].squeeze() * domain_aware_text_view + \
                            weight_common[:, 1].squeeze() * domain_aware_image_view + \
                            weight_common[:, 2].squeeze() * domain_aware_fusion_view 

        fake_news_sigmoid = torch.clamp(fake_news_sigmoid, min=0.0, max=1.0)

        return fake_news_sigmoid, text_fake_news, text_multi_domain, image_fake_news, image_multi_domain, fusion_fake_news, fusion_multi_domain, domain_aware_text_view, domain_aware_image_view, domain_aware_fusion_view



class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 bert,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 loss_weight=[1, 0.006, 0.009, 5e-5],
                 early_stop=5,
                 epoches=100
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        if not os.path.exists(save_param_dir):
            self.save_param_dir = os.makedirs(save_param_dir)
        else:
            self.save_param_dir = save_param_dir

    def train(self):
        self.model = DAMMFNDMODEL(self.emb_dim, self.mlp_dims, self.bert, 320, self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                batch_data = clipdata2gpu(batch)
                label = batch_data['label']
                category = batch_data['multi_category']
                labels_domain = category

                label0, text_fake_news, text_multi_domain, image_fake_news, image_multi_domain, fusion_fake_news, fusion_multi_domain, domain_aware_text_view, domain_aware_image_view, domain_aware_fusion_view = self.model(
                    **batch_data)
                loss0 = loss_fn(label0, label.float())


                loss11 = torch.nn.functional.binary_cross_entropy_with_logits(text_multi_domain, labels_domain.float())
                loss21 = torch.nn.functional.binary_cross_entropy_with_logits(image_multi_domain, labels_domain.float())
                loss31 = torch.nn.functional.binary_cross_entropy_with_logits(fusion_multi_domain,
                                                                              labels_domain.float())


                loss12_aux = loss_fn(domain_aware_text_view.squeeze(), label.float())
                loss22_aux = loss_fn(domain_aware_image_view.squeeze(), label.float())
                loss32_auc = loss_fn(domain_aware_fusion_view.squeeze(), label.float())

                uniform_target = torch.ones_like(text_fake_news, dtype=torch.float).cuda() / 9
                loss12 = F.kl_div(text_fake_news, uniform_target.float())
                loss22 = F.kl_div(image_fake_news, uniform_target.float())
                loss32 = F.kl_div(fusion_fake_news, uniform_target.float())

                loss = loss0 + (loss11 + loss12 + loss21 + loss22 + loss31 + loss32) / 6 + (loss12_aux + loss22_aux + loss32_auc) / 3.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))
            print("----- self.save_param_dir", self.save_param_dir)
            results0 = self.test(self.val_loader)
            mark = recorder.add(results0)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter_dammfnd.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_dammfnd.pkl')))
        print("开始进行最后的测试")
        results0 = self.test(self.test_loader)
        print("final: ", results0)

        return results0, os.path.join(self.save_param_dir, 'parameter_dammfnd.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = clipdata2gpu(batch)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred, _, _, _, _, _, _, _, _, _ = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        metric_res = metricsTrueFalse(label, pred, category, self.category_dict)
        return metric_res