# Adapted from https://github.com/psinger/kaggle-curriculum-solution/blob/master/models/feedback_metric_model.py

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel

# -------


class ArcMarginProduct(nn.Module):
    """Arc margin product for head of ArcFace Loss"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=2):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features*k, in_features))

        self.reset_parameters()

        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine = cosine.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine, dim=2)
        return cosine


class ArcFaceLoss(nn.modules.Module):
    """Calculate ArcFace Loss"""

    def __init__(self, scale, margin, embedding_size, num_classes):
        super().__init__()

        s = scale
        m = margin

        in_features = embedding_size
        out_features = num_classes

        # self.head = ArcMarginProduct_subcenter(in_features, out_features)
        self.head = ArcMarginProduct(in_features, out_features)

        self.crit = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
        self.init(s, m)

    def init(self, s, m):
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        logits = self.head(embeddings)
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)

        output = labels2 * phi
        output = output + ((1.0 - labels2) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)

        return loss


class OnmiNERModel(nn.Module):
    """
    OmniNERModel with arcface loss
    """

    def __init__(self, cfg):
        super(OnmiNERModel, self).__init__()
        self.cfg = deepcopy(cfg)

        # --- Backbone -------------------------------------------------------------------#
        backbone_config = AutoConfig.from_pretrained(cfg.model.backbone_path)
        backbone_config.update({"use_cache": False})

        if cfg.model.skip_dropout:
            backbone_config.update(
                {
                    "hidden_dropout_prob": 0.,
                    "attention_probs_dropout_prob": 0.,
                }
            )

        self.backbone = AutoModel.from_pretrained(
            self.cfg.model.backbone_path,
            config=backbone_config
        )

        # enable gradient checkpointing
        if cfg.model.use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # --- Loss function ------------------------------------------------------------#
        self.layer_norm = nn.LayerNorm(self.backbone.config.hidden_size, 1e-7)

        self.loss_fn = ArcFaceLoss(
            scale=cfg.model.arcface.scale,
            margin=cfg.model.arcface.margin,
            embedding_size=self.backbone.config.hidden_size,
            num_classes=cfg.model.arcface.n_groups,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        span_head_idxs,
        span_tail_idxs,
        labels=None,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )

        encoder_layer = outputs.last_hidden_state

        # --
        feature_vector = []
        bs = encoder_layer.shape[0]

        for i in range(bs):
            span_vec_i = []
            for head, tail in zip(span_head_idxs[i], span_tail_idxs[i]):
                tmp = torch.mean(encoder_layer[i, head:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_spans, h)
            feature_vector.append(span_vec_i)

        embeddings = torch.stack(feature_vector)  # (bs, num_spans, h)
        # print(embeddings.shape)
        embeddings = self.layer_norm(embeddings)  # (bs, num_spans, h)
        print(embeddings[0])

        loss = None
        if labels is not None:
            labels = labels.long()  # (bs, num_spans)
            # reshape embeddings and labels
            embeddings = embeddings.reshape(-1, embeddings.size(-1))
            labels = labels.reshape(-1)
            print(f"embeddings shape: {embeddings.shape}, labels shape: {labels.shape}")
            loss = self.loss_fn(embeddings, labels)

        return loss
