# 面向癫痫发作预测的多层次可解释模型：结构图说明与 PyTorch 伪代码框架

下面内容可直接用于开题报告、论文方法章节或项目设计文档，包含两部分：

1. **模型结构图说明**
2. **PyTorch 伪代码框架**

---

# 一、模型结构图说明

## 1. 总体思路

模型采用 **“双分支共享主干 + 四层解释头 + 风险聚合告警层”** 的结构：

- **原始信号分支**：保证预测性能
- **可解释特征分支**：提供特征级解释锚点
- **共享融合层**：形成统一时空表征
- **四层解释头**：
  - 特征解释
  - 通道解释
  - 时序解释
  - 病例解释
- **告警层**：把窗级风险转成连续预警

---

## 2. 可直接放论文里的结构图

```text
┌──────────────────────────────────────────────────────────────────────┐
│                           输入 EEG 序列                              │
│                 X ∈ R[B, N, C, T]                                   │
│   B: batch size, N: 连续窗口数, C: 通道数, T: 每窗时间点数            │
└──────────────────────────────────────────────────────────────────────┘
                                │
               ┌────────────────┴────────────────┐
               │                                 │
               │                                 │
┌──────────────────────────────┐   ┌──────────────────────────────────┐
│       原始信号编码分支        │   │       可解释特征提取分支          │
│  每窗 EEG: [C, T]            │   │  每窗提取人工特征: [C, K]         │
│  1D CNN / TCN / STFT-CNN     │   │  band power / entropy / PLV 等    │
│  输出: Z_raw ∈ R[B,N,D1]     │   │  MLP/Concept Encoder              │
│                              │   │  输出: Z_feat ∈ R[B,N,D2]         │
└──────────────────────────────┘   └──────────────────────────────────┘
               │                                 │
               └────────────────┬────────────────┘
                                │
                    ┌────────────────────────┐
                    │      融合共享表征层     │
                    │ concat + fusion MLP    │
                    │ Z ∈ R[B,N,D]           │
                    └────────────────────────┘
                                │
      ┌───────────────┬───────────────┬───────────────┬───────────────┐
      │               │               │               │
      │               │               │               │
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────────┐
│ 特征解释头    │ │ 通道解释头    │ │ 时序解释头    │ │ 病例解释头      │
│ Feature Head │ │ Channel Head │ │ Temporal Head│ │ Prototype Head │
│ 输出 α_f      │ │ 输出 α_c      │ │ 输出 α_t      │ │ 检索相似原型 p  │
│ [B,N,K’]     │ │ [B,N,C]       │ │ [B,N]         │ │ Top-k prototypes│
└──────────────┘ └──────────────┘ └──────────────┘ └────────────────┘
      │               │               │               │
      └───────────────┴───────┬───────┴───────────────┘
                              │
                  ┌────────────────────────────┐
                  │      窗级风险预测头         │
                  │  Risk Head → r_t           │
                  │  r ∈ R[B,N,1]              │
                  └────────────────────────────┘
                              │
                  ┌────────────────────────────┐
                  │   风险聚合 / 告警决策层      │
                  │ EWMA / firing power        │
                  │ threshold + refractory     │
                  │ 输出: seizure warning      │
                  └────────────────────────────┘
                              │
                  ┌────────────────────────────┐
                  │         最终输出            │
                  │ 1. 风险分数                 │
                  │ 2. 特征解释                 │
                  │ 3. 通道解释                 │
                  │ 4. 时序解释                 │
                  │ 5. 病例相似性解释           │
                  └────────────────────────────┘
```

---

## 3. 每个模块的作用说明

### （1）输入层

输入不是单窗，而是**连续多个窗口**：

\[
X \in \mathbb{R}^{B \times N \times C \times T}
\]

- \(B\)：batch size
- \(N\)：连续时间窗数量
- \(C\)：通道数
- \(T\)：每个时间窗的采样点数

这样做是为了支持**时序解释**。

---

### （2）原始信号编码分支

作用：从原始 EEG 中提取高维判别表征，保证预测性能。

可选：
- 1D CNN
- TCN
- CNN + Transformer
- STFT 后 2D CNN

输出：
\[
Z_{raw} \in \mathbb{R}^{B \times N \times D_1}
\]

---

### （3）可解释特征分支

作用：把 EEG 转成有生理意义的中间表示。

每窗每通道提取：
- 频带功率
- 频带比
- sample entropy
- fractal dimension
- coherence / PLV
- 图论指标

形成特征矩阵：
\[
F \in \mathbb{R}^{B \times N \times C \times K}
\]

再通过 MLP / concept encoder 压缩成：
\[
Z_{feat} \in \mathbb{R}^{B \times N \times D_2}
\]

---

### （4）融合共享表征层

把原始表征和概念表征融合：

\[
Z = Fusion([Z_{raw}; Z_{feat}])
\]

输出：
\[
Z \in \mathbb{R}^{B \times N \times D}
\]

这是后续四个解释头和风险头共享的主干表示。

---

### （5）特征解释头

输入：概念特征或融合表征  
输出：特征权重

\[
\alpha_f \in \mathbb{R}^{B \times N \times K'}
\]

含义：
- 哪类特征推动了风险上升
- 例如 gamma 功率、熵、连通性等的重要性

---

### （6）通道解释头

输入：通道级中间表示  
输出：每个通道权重

\[
\alpha_c \in \mathbb{R}^{B \times N \times C}
\]

含义：
- 哪些通道 / 脑区最关键
- 可以映射到左颞区、右额区等空间解释

---

### （7）时序解释头

输入：连续窗口表征 \(Z\)  
输出：时间权重

\[
\alpha_t \in \mathbb{R}^{B \times N}
\]

含义：
- 哪些时间窗推动了最终预警
- 风险是持续上升还是瞬时波动

---

### （8）病例解释头

核心思想：建立**原型记忆库**。

- 保存典型 preictal / interictal 原型 embedding
- 当前样本与原型做相似度匹配

输出：
- Top-k 相似历史片段
- 相似度分数
- 原型标签（preictal / interictal）

---

### （9）风险头与告警层

每个时间窗输出风险：

\[
r_t = RiskHead(Z_t)
\]

再进行连续风险聚合：

\[
R_t = \beta R_{t-1} + (1-\beta) r_t
\]

最后用阈值 + 冷却时间输出告警。

---

## 4. 一张更适合 PPT/论文画图软件的简化版

```text
EEG序列输入
   │
   ├── 原始信号分支（CNN/TCN）
   │        └── 时空深层表征
   │
   ├── 可解释特征分支（频谱/熵/连通性）
   │        └── 概念表征
   │
   └── 融合共享表征
            │
            ├── 特征解释头：哪些特征重要
            ├── 通道解释头：哪些脑区/电极重要
            ├── 时序解释头：哪些时间窗重要
            ├── 病例解释头：最像哪些历史发作前片段
            │
            └── 风险预测头
                    └── 风险聚合与告警输出
```

---

# 二、PyTorch 伪代码框架

下面这份是**研究原型级别**的伪代码，结构完整，但省略了部分底层细节，适合后续展开成正式实现。

---

## 1. 模型整体框架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1. 原始信号编码分支
# 输入: x [B, N, C, T]
# 输出:
#   z_raw [B, N, D_raw]
#   ch_embed [B, N, C, D_ch]   # 给通道解释头用
# =========================
class RawSignalEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, ch_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # 每个通道的独立嵌入，可用于通道解释
        self.channel_proj = nn.Linear(hidden_dim, ch_dim)

        # 全局池化得到每窗表示
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, N, C, T]
        B, N, C, T = x.shape
        x = x.reshape(B * N, C, T)

        h = F.relu(self.bn1(self.conv1(x)))   # [B*N, 32, T]
        h = F.relu(self.bn2(self.conv2(h)))   # [B*N, 64, T]
        h = F.relu(self.bn3(self.conv3(h)))   # [B*N, D_raw, T]

        # 窗级 embedding
        z_raw = self.global_pool(h).squeeze(-1)   # [B*N, D_raw]
        z_raw = z_raw.reshape(B, N, -1)           # [B, N, D_raw]

        # 通道级 embedding（简化示意）
        # 正式实现建议改成 per-channel encoder
        x_mean = x.mean(dim=-1)                   # [B*N, C]
        ch_embed = x_mean.unsqueeze(-1).repeat(1, 1, self.channel_proj.out_features)
        ch_embed = ch_embed.reshape(B, N, C, -1) # [B, N, C, D_ch]

        return z_raw, ch_embed
```

---

## 2. 可解释特征分支

这里默认你已经有一个 `extract_handcrafted_features()`，它把 EEG 每窗转成人工特征。

```python
# =========================
# 2. 可解释特征编码分支
# 输入:
#   feat [B, N, C, K]
# 输出:
#   z_feat [B, N, D_feat]
#   feat_concept [B, N, Kc]
# =========================
class FeatureConceptEncoder(nn.Module):
    def __init__(self, in_feat_dim: int, hidden_dim: int, concept_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.concept_head = nn.Linear(hidden_dim, concept_dim)

    def forward(self, feat):
        # feat: [B, N, C, K]
        B, N, C, K = feat.shape

        # 先在通道维做聚合，也可以换成 attention pooling
        feat_agg = feat.mean(dim=2)              # [B, N, K]
        h = self.mlp(feat_agg)                   # [B, N, hidden_dim]
        feat_concept = self.concept_head(h)      # [B, N, concept_dim]

        return h, feat_concept
```

---

## 3. 融合层

```python
# =========================
# 3. 融合共享表征层
# 输入:
#   z_raw [B, N, D1]
#   z_feat [B, N, D2]
# 输出:
#   z [B, N, D]
# =========================
class FusionLayer(nn.Module):
    def __init__(self, d_raw: int, d_feat: int, d_model: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(d_raw + d_feat, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model)
        )

    def forward(self, z_raw, z_feat):
        z = torch.cat([z_raw, z_feat], dim=-1)
        z = self.fusion(z)
        return z
```

---

## 4. 特征解释头

```python
# =========================
# 4. 特征解释头
# 输入:
#   feat_concept [B, N, Kc]
# 输出:
#   feat_importance [B, N, Kc]
# =========================
class FeatureExplanationHead(nn.Module):
    def __init__(self, concept_dim: int):
        super().__init__()
        self.gate = nn.Linear(concept_dim, concept_dim)

    def forward(self, feat_concept):
        scores = self.gate(feat_concept)                     # [B, N, Kc]
        feat_importance = torch.softmax(scores, dim=-1)
        return feat_importance
```

---

## 5. 通道解释头

```python
# =========================
# 5. 通道解释头
# 输入:
#   ch_embed [B, N, C, D_ch]
# 输出:
#   ch_importance [B, N, C]
# =========================
class ChannelExplanationHead(nn.Module):
    def __init__(self, ch_dim: int):
        super().__init__()
        self.scorer = nn.Linear(ch_dim, 1)

    def forward(self, ch_embed):
        scores = self.scorer(ch_embed).squeeze(-1)          # [B, N, C]
        ch_importance = torch.softmax(scores, dim=-1)
        return ch_importance
```

---

## 6. 时序解释头

```python
# =========================
# 6. 时序解释头
# 输入:
#   z [B, N, D]
# 输出:
#   temp_importance [B, N]
#   z_context [B, D]
# =========================
class TemporalExplanationHead(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.attn = nn.Linear(d_model, 1)

    def forward(self, z):
        # z: [B, N, D]
        h, _ = self.gru(z)                                  # [B, N, D]
        scores = self.attn(h).squeeze(-1)                   # [B, N]
        temp_importance = torch.softmax(scores, dim=-1)     # [B, N]

        # 注意力加权上下文
        z_context = torch.sum(h * temp_importance.unsqueeze(-1), dim=1)  # [B, D]
        return temp_importance, z_context
```

---

## 7. 病例解释头（原型记忆）

```python
# =========================
# 7. 病例解释头 / 原型记忆库
# 输入:
#   z_context [B, D]
# 输出:
#   proto_sim [B, P]
#   topk_idx, topk_val
# =========================
class PrototypeExplanationHead(nn.Module):
    def __init__(self, d_model: int, num_prototypes: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))
        self.prototype_labels = None   # 可在外部维护元信息

    def forward(self, z_context, topk: int = 3):
        # z_context: [B, D]
        z_norm = F.normalize(z_context, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)

        sim = torch.matmul(z_norm, p_norm.t())              # [B, P]
        topk_val, topk_idx = torch.topk(sim, k=topk, dim=-1)

        return sim, topk_idx, topk_val
```

---

## 8. 风险预测头与告警层

```python
# =========================
# 8. 风险头
# 输入:
#   z_context [B, D]
# 输出:
#   risk_logit [B, 1]
# =========================
class RiskHead(nn.Module):
    def __init__(self, d_model: int, concept_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model + concept_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )

    def forward(self, z_context, feat_summary):
        x = torch.cat([z_context, feat_summary], dim=-1)
        risk_logit = self.fc(x)
        return risk_logit


# =========================
# 9. EWMA 风险聚合器（推理时用）
# 输入:
#   risk_seq [N]
# 输出:
#   agg_risk [N]
# =========================
def ewma_aggregate(risk_seq, beta=0.8):
    agg = []
    prev = 0.0
    for r in risk_seq:
        prev = beta * prev + (1 - beta) * float(r)
        agg.append(prev)
    return agg
```

---

## 9. 总模型封装

```python
class InterpretableSeizurePredictor(nn.Module):
    def __init__(
        self,
        num_channels: int,
        raw_dim: int,
        ch_dim: int,
        feat_dim: int,
        feat_hidden_dim: int,
        concept_dim: int,
        d_model: int,
        num_prototypes: int
    ):
        super().__init__()

        self.raw_encoder = RawSignalEncoder(
            in_channels=num_channels,
            hidden_dim=raw_dim,
            ch_dim=ch_dim
        )

        self.feat_encoder = FeatureConceptEncoder(
            in_feat_dim=feat_dim,
            hidden_dim=feat_hidden_dim,
            concept_dim=concept_dim
        )

        self.fusion = FusionLayer(
            d_raw=raw_dim,
            d_feat=feat_hidden_dim,
            d_model=d_model
        )

        self.feature_head = FeatureExplanationHead(concept_dim=concept_dim)
        self.channel_head = ChannelExplanationHead(ch_dim=ch_dim)
        self.temporal_head = TemporalExplanationHead(d_model=d_model)
        self.prototype_head = PrototypeExplanationHead(
            d_model=d_model,
            num_prototypes=num_prototypes
        )
        self.risk_head = RiskHead(d_model=d_model, concept_dim=concept_dim)

    def forward(self, x, feat):
        """
        x:    [B, N, C, T]
        feat: [B, N, C, K]
        """
        z_raw, ch_embed = self.raw_encoder(x)                   # [B,N,D_raw], [B,N,C,D_ch]
        z_feat, feat_concept = self.feat_encoder(feat)          # [B,N,D_feat], [B,N,Kc]
        z = self.fusion(z_raw, z_feat)                          # [B,N,D]

        feat_importance = self.feature_head(feat_concept)       # [B,N,Kc]
        ch_importance = self.channel_head(ch_embed)             # [B,N,C]
        temp_importance, z_context = self.temporal_head(z)      # [B,N], [B,D]

        proto_sim, topk_idx, topk_val = self.prototype_head(z_context)

        # 对概念层做时间聚合，得到最终风险头输入
        feat_summary = torch.sum(
            feat_concept * feat_importance,
            dim=1
        )                                                       # [B,Kc]

        risk_logit = self.risk_head(z_context, feat_summary)    # [B,1]
        risk_prob = torch.sigmoid(risk_logit)

        output = {
            "risk_logit": risk_logit,
            "risk_prob": risk_prob,
            "feat_importance": feat_importance,
            "channel_importance": ch_importance,
            "temporal_importance": temp_importance,
            "prototype_similarity": proto_sim,
            "prototype_topk_idx": topk_idx,
            "prototype_topk_val": topk_val,
            "z_context": z_context,
        }
        return output
```

---

## 10. 多任务损失函数

```python
def compute_loss(outputs, labels,
                 lambda_feat=1e-3,
                 lambda_ch=1e-3,
                 lambda_temp=1e-3,
                 lambda_proto=1e-3):
    """
    labels: [B,1]
    """
    risk_logit = outputs["risk_logit"]
    feat_importance = outputs["feat_importance"]
    ch_importance = outputs["channel_importance"]
    temp_importance = outputs["temporal_importance"]
    proto_sim = outputs["prototype_similarity"]

    # 1. 主预测损失
    pred_loss = F.binary_cross_entropy_with_logits(risk_logit, labels.float())

    # 2. 特征稀疏正则：鼓励少数概念主导
    feat_sparsity = feat_importance.abs().mean()

    # 3. 通道稀疏正则
    ch_sparsity = ch_importance.abs().mean()

    # 4. 时序平滑/低熵约束：让重要时间窗更集中
    temp_entropy = -(temp_importance * (temp_importance + 1e-8).log()).sum(dim=-1).mean()

    # 5. 原型分离项（示意）
    proto_conf = -proto_sim.max(dim=-1).values.mean()

    loss = (
        pred_loss
        + lambda_feat * feat_sparsity
        + lambda_ch * ch_sparsity
        + lambda_temp * temp_entropy
        + lambda_proto * proto_conf
    )

    loss_dict = {
        "total_loss": loss,
        "pred_loss": pred_loss,
        "feat_sparsity": feat_sparsity,
        "ch_sparsity": ch_sparsity,
        "temp_entropy": temp_entropy,
        "proto_conf": proto_conf,
    }
    return loss_dict
```

---

## 11. 训练循环伪代码

```python
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        x = batch["eeg"].to(device)          # [B,N,C,T]
        feat = batch["feat"].to(device)      # [B,N,C,K]
        y = batch["label"].to(device)        # [B,1]

        outputs = model(x, feat)
        loss_dict = compute_loss(outputs, y)

        optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        optimizer.step()

        total_loss += float(loss_dict["total_loss"].item())

    return total_loss / len(dataloader)
```

---

# 三、推理阶段的解释输出模板

你后续做系统界面时，可以把模型输出组织成下面这种格式：

```python
def build_explanation_report(outputs, feature_names, channel_names, prototype_meta, topk=3):
    feat_imp = outputs["feat_importance"][0].mean(dim=0)      # [Kc]
    ch_imp = outputs["channel_importance"][0].mean(dim=0)     # [C]
    temp_imp = outputs["temporal_importance"][0]              # [N]
    proto_idx = outputs["prototype_topk_idx"][0]              # [topk]
    proto_val = outputs["prototype_topk_val"][0]              # [topk]

    feat_rank = torch.argsort(feat_imp, descending=True)
    ch_rank = torch.argsort(ch_imp, descending=True)
    time_rank = torch.argsort(temp_imp, descending=True)

    report = {
        "top_features": [(feature_names[i], float(feat_imp[i])) for i in feat_rank[:topk]],
        "top_channels": [(channel_names[i], float(ch_imp[i])) for i in ch_rank[:topk]],
        "top_time_windows": [(int(i), float(temp_imp[i])) for i in time_rank[:topk]],
        "top_prototypes": [
            {
                "prototype_id": int(idx),
                "similarity": float(val),
                "meta": prototype_meta[int(idx)]
            }
            for idx, val in zip(proto_idx, proto_val)
        ]
    }
    return report
```

输出结果会像：

- **特征解释**：gamma power、sample entropy、PLV 最重要
- **通道解释**：F7、T3、T5 最重要
- **时序解释**：最近第 7、8、9 个窗口贡献最高
- **病例解释**：最相似的是历史第 2 次发作前 18 分钟片段

---

# 四、正式实现时还需要补的 5 个关键点

## 1. 人工特征提取函数
需要单独实现：
- 频带功率
- 熵
- 分形维数
- PLV/coherence
- 图论指标

## 2. 更真实的通道编码
上面通道分支里用了简化写法。正式版建议做：
- per-channel CNN
- graph encoder
- spatial transformer

## 3. 原型库元信息管理
原型不仅存向量，还要存：
- 来源患者
- 来源发作编号
- 距离发作时间
- 类别标签

## 4. 时序风险聚合与告警逻辑
真正部署时建议加入：
- EWMA
- 连续超过阈值 m 次才告警
- refractory period

## 5. 解释稳定性验证
论文里最好单独做：
- 同一患者不同发作是否解释一致
- 不同折训练是否解释稳定
- 与病灶侧/临床标注是否一致

---

# 五、可直接放论文的方法总结

**本文提出一种面向癫痫发作预测的多层次可解释模型。模型以连续 EEG 窗口序列为输入，采用原始信号分支提取深层时序表征，采用可解释特征分支提取频谱、复杂度及脑网络概念表征，并通过共享融合层获得统一表示。在此基础上，分别构建特征解释头、通道解释头、时序解释头和病例解释头，实现对“模型依赖何种生理特征、哪些脑区最关键、哪些时间段驱动风险上升、当前状态与哪些历史发作前模式相似”的多层次解释。最后，通过风险预测头输出发作风险，并经风险聚合与阈值决策实现连续预警。**
