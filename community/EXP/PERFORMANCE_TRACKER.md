# Performance Tracker — HCMUS TheFangs | NeurIPS 2026 Unlearning

> **Cách dùng**: Sau mỗi lần chạy EXP trên Kaggle, copy số từ JSON/ASCII table vào đây.  
> Ghi insight ngay khi còn nhớ. Format: fill vào bảng → viết phân tích bên dưới.

---

## Metric Glossary

| Metric | Ý nghĩa | Hướng tốt |
|--------|----------|-----------|
| `model_utility` | Khả năng trả lời câu hỏi retain set đúng | ↑ cao hơn tốt (max ~1.0) |
| `forget_quality` | Mức độ "quên" forget set (MIA-based) | ↑ cao hơn tốt (max ~1.0) |
| `privleak` | Privacy leakage score | ↓ gần 0 tốt (thường âm lớn = tệ) |
| `extraction_strength` | Khả năng extract nội dung từ forget | ↓ thấp tốt |
| **Composite** | `0.4·utility + 0.4·forget_quality - 0.1·\|privleak\| - 0.1·extraction` | ↑ cao hơn tốt |

---

## Composite Score Formula

$$\text{Composite} = 0.4 \cdot \text{utility} + 0.4 \cdot \text{forget\_quality} - 0.1 \cdot |\text{privleak}| - 0.1 \cdot \text{extraction}$$

---

## 🎯 SOTA Targets — Beat These for NeurIPS 2026

> **Mục tiêu**: Vượt qua **SimNPO** (SOTA hiện tại) trên TOFU forget10, Llama-3.2-1B-Instruct.  
> Nguồn: *OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics*, NeurIPS 2025, Table 3 & Table 6.  
> Setup chuẩn: single A100, BF16, batch=32, paged AdamW, 27 hyperparameter trials/method.

---

### 📊 Table 3 — Full Composite (Agg = HM of Mem + Priv + Utility) — PRIMARY TARGET

> Đây là bảng **chính thức** của paper. Metric gộp = Harmonic Mean của 3 chiều.  
> **Nhắm tới**: Agg > 0.53 (SimNPO), đồng thời Utility ≥ 1.00, Priv ≥ 0.63.

| Method | Agg ↑ | Mem ↑ | Priv ↑ | Utility ↑ | 🎯 Beat? |
|--------|:-----:|:-----:|:------:|:---------:|:--------:|
| **Retain** *(gold standard)* | **0.58** | **0.31** | **1.00** | **0.99** | — |
| **SimNPO** *(#1 SOTA)* | **0.53** | 0.32 | 0.63 | 1.00 | ⬜ |
| RMU *(#2)* | 0.52 | 0.47 | 0.50 | 0.61 | ⬜ |
| UNDIAL *(#3)* | 0.42 | 0.27 | 0.48 | 0.78 | ⬜ |
| AltPO | 0.15 | 0.63 | 0.06 | 0.95 | ✅ trivial |
| IdkNLL | 0.15 | 0.08 | 0.17 | 0.93 | ✅ trivial |
| NPO | 0.15 | 0.52 | 0.06 | 0.99 | ✅ trivial |
| IdkDPO | 0.14 | 0.56 | 0.06 | 0.95 | ✅ trivial |
| GradDiff | 9e-3 | 0.97 | 3e-3 | 0.79 | ✅ trivial |
| Init. finetuned *(worst)* | 0.00 | 0.00 | 0.10 | 1.00 | ✅ trivial |
| **ERBWP (EXP_04)** | — | — | — | — | ⬜ |
| **CSAES (EXP_05)** | — | — | — | — | ⬜ |
| **MGFAA (EXP_06)** | — | — | — | — | ⬜ |

**Metric definitions** (paper formulas):
- **Mem** = HM(1−ES, 1−EM, 1−Para.Prob, 1−Truth Ratio) — higher = ít knowledge còn lại hơn = tốt
- **Priv** = HM(sLOSS, sZLib, sMin-k, sMink++) — higher = gần retain model hơn
- **Utility** = HM(Model_Utility_TOFU, Forget_Fluency) — higher = tốt

---

### 📊 Table 6 — Mem+Utility Only Composite (không có Privacy trong Agg)

> Paper dùng bảng này khi **không có oracle retain model**. Relevant cho Kaggle setup.  
> **Nhắm tới**: Agg > 0.87 (GradDiff không khó), quan trọng hơn là beat NPO (0.69) và RMU (0.53) mà **vẫn có Priv tốt**.

| Method | Agg (Mem+Util) ↑ | Mem ↑ | Priv (ref) | Utility ↑ | 🎯 Beat? |
|--------|:---------------:|:-----:|:----------:|:---------:|:--------:|
| **Retain** *(gold)* | **0.58** | **0.31** | **1.00** | **0.99** | — |
| GradDiff *(#1 w/o priv)* | 0.87 | 0.97 | 3.3e-3 | 0.79 | ⬜ |
| AltPO | 0.76 | 0.63 | 0.06 | 0.95 | ⬜ |
| IdkDPO | 0.71 | 0.56 | 0.06 | 0.95 | ⬜ |
| NPO | 0.69 | 0.52 | 0.06 | 0.99 | ⬜ |
| RMU | 0.53 | 0.47 | 0.50 | 0.61 | ⬜ |
| **SimNPO** | 0.49 | 0.32 | 0.63 | 1.00 | ✅ trivial |
| UNDIAL | 0.40 | 0.27 | 0.48 | 0.78 | ✅ trivial |
| IdkNLL | 0.14 | 0.08 | 0.17 | 0.93 | ✅ trivial |
| **ERBWP (EXP_04)** | — | — | — | — | ⬜ |
| **CSAES (EXP_05)** | — | — | — | — | ⬜ |
| **MGFAA (EXP_06)** | — | — | — | — | ⬜ |

---

### 📊 Table 2 — Metric Reliability Meta-Evaluation

> Biết metric nào đáng tin để tối ưu khi tuning. Paper dùng **Extraction Strength** và **Exact Mem** là reliable nhất.

| Metric | Overall Agg ↑ | Faithfulness ↑ | Robustness ↑ | Quantiz. ↑ | Relearn ↑ |
|--------|:-------------:|:--------------:|:------------:|:----------:|:---------:|
| **Extraction Strength** *(best)* | **0.85** | 0.92 | 0.79 | 0.95 | 0.68 |
| Exact Memorization | 0.80 | 0.90 | 0.72 | 0.92 | 0.59 |
| Truth Ratio | 0.73 | **0.95** | 0.59 | 0.92 | 0.43 |
| Para. Prob | 0.73 | 0.71 | 0.75 | 0.60 | **0.98** |
| Para. ROUGE | 0.72 | 0.89 | 0.61 | 0.93 | 0.45 |
| Probability | 0.72 | 0.82 | 0.65 | 0.60 | 0.70 |
| ROUGE | 0.70 | 0.79 | 0.64 | 0.93 | 0.48 |
| Jailbreak ROUGE | 0.69 | 0.83 | 0.59 | 0.85 | 0.45 |
| MIA - ZLib | 0.71 | 0.92 | 0.57 | 0.56 | 0.59 |
| MIA - MinK | 0.67 | 0.93 | 0.52 | 0.48 | 0.57 |
| MIA - LOSS | 0.66 | 0.93 | 0.52 | 0.48 | 0.57 |
| MIA - MinK++ | 0.61 | 0.81 | 0.48 | 0.61 | 0.40 |

**Takeaway**: Khi tuning hyperparameters, ưu tiên tối ưu **ES** và **EM** (reliable nhất). MIA metrics có faithfulness cao nhưng robustness kém — không dùng để **select** model.

---

### 🏹 Explicit NeurIPS 2026 Targets

| Dimension | Paper SOTA | **Our Target** | Notes |
|-----------|:----------:|:--------------:|-------|
| **Agg (Table 3)** | 0.53 (SimNPO) | **≥ 0.56** | Vượt SimNPO ít nhất +0.03 |
| **Agg vs Retain** | 0.58 (Retain) | **≥ 0.57** | Sát Retain là đủ cho paper |
| **Mem** | 0.32 (SimNPO) | **≥ 0.38** | Cải thiện memorization removal |
| **Priv** | 0.63 (SimNPO) | **≥ 0.65** | Giữ privacy tốt hơn |
| **Utility** | 1.00 (SimNPO) | **≥ 0.98** | Không drop utility quá 2% |
| **Robustness (relearn)** | 0.68 (ES best) | **≥ 0.70** | Harder to relearn = better |
| **Consistency** | 1 method | **≥ 2/3 methods beat SOTA** | Cần ERBWP hoặc CSAES hoặc MGFAA beat |

**Chiến lược**:
1. 🎯 **Primary**: ERBWP (entropy-reservoir) — tackle over/under-unlearning balance → target Priv + Mem both improve
2. 🎯 **Secondary**: CSAES (contrastive suppression) — maintain utility while forgetting → target Utility stay high
3. 🎯 **Ablation**: MGFAA (gradient-free) — gradient-free steers → interesting if matches or beats gradient-based
4. 📝 **Paper angle**: Show that all 3 novel methods beat NPO/UNDIAL, at least 1 beats SimNPO

---

## EXP_00 — GradAscent Baseline (forget01 / forget05 / forget10)

**Config**: `trainer=GradAscent`, `max_steps=200`, `lr=1e-5`, `Llama-3.2-1B-Instruct`  
**Run date**: ___  
**Kaggle notebook**: ___

| Split | model_utility | forget_quality | privleak | extraction_strength | Composite |
|-------|:---:|:---:|:---:|:---:|:---:|
| forget01 | — | — | — | — | — |
| forget05 | — | — | — | — | — |
| forget10 | — | — | — | — | — |

**Insight**:
- [ ] utility drop từ forget01 → forget10: ___
- [ ] forget_quality trend: ___
- [ ] Có bị representation collapse không? (utility < 0.3 = collapse): ___
- [ ] Ghi chú: ___

---

## EXP_01 — Standard Methods Benchmark (forget10)

**Config**: GradDiff / NPO / SimNPO / RMU, `forget10`, `max_steps=200`  
**Run date**: ___  
**Kaggle notebook**: ___

| Method | model_utility | forget_quality | privleak | extraction_strength | Composite | Paper ref |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| GradDiff | — | — | — | — | — | Agg≈0.009 |
| NPO | — | — | — | — | — | — |
| SimNPO | — | — | — | — | — | Agg≈0.53 |
| RMU | — | — | — | — | — | Agg≈0.52 |

**So sánh với paper (Table 3, Llama-3.2-1B, forget10)**:

| Method | Paper utility | Ours utility | Paper forget_q | Ours forget_q |
|--------|:---:|:---:|:---:|:---:|
| SimNPO | 1.00 | — | 0.53 | — |
| RMU | 0.61 | — | 0.52 | — |
| GradDiff | 0.79 | — | ~0.009 | — |

**Insight**:
- [ ] Gap với paper số lớn nhất ở method nào? ___
- [ ] SimNPO có reproduce được utility=1.0 không? ___
- [ ] RMU memorization removal quality (forget_quality): ___
- [ ] Ghi chú về learning rate sensitivity: ___

---

## EXP_02 — Sequential Unlearning Chain (NPO)

**Config**: NPO sequential forget01 → forget05 → forget10  
**Run date**: ___  
**Kaggle notebook**: ___

| Step | Forget Split | model_utility | forget_quality | privleak | extraction_strength | Composite |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Step 1 | forget01 | — | — | — | — | — |
| Step 2 | forget05 | — | — | — | — | — |
| Step 3 | forget10 | — | — | — | — | — |

**So sánh với EXP_01 NPO direct (forget10)**:

| | model_utility | forget_quality | privleak |
|---|:---:|:---:|:---:|
| NPO direct (EXP_01) | — | — | — |
| NPO sequential step3 | — | — | — |
| **Delta (seq - direct)** | — | — | — |

**Insight**:
- [ ] Error accumulation có xảy ra không? (utility_seq < utility_direct): ___
- [ ] Forgetting depth: forget_quality_seq vs direct: ___
- [ ] Đây là evidence cho "non-geodesic drift" của PreRound_1: ___
- [ ] Ghi chú: ___

---

## EXP_03 — Geometry-Aware Community Methods (WGA / UNDIAL / DPO)

**Config**: WGA, UNDIAL, DPO trên forget10  
**Run date**: ___  
**Kaggle notebook**: ___

| Method | model_utility | forget_quality | privleak | extraction_strength | Composite |
|--------|:---:|:---:|:---:|:---:|:---:|
| WGA | — | — | — | — | — |
| UNDIAL | — | — | — | — | — |
| DPO | — | — | — | — | — |

**So sánh với EXP_01 baselines**:

| | Best EXP_01 (SimNPO) | WGA | UNDIAL | DPO |
|---|:---:|:---:|:---:|:---:|
| utility | — | — | — | — |
| forget_quality | — | — | — | — |
| Composite | — | — | — | — |

**Insight**:
- [ ] WGA geometry-awareness có giúp utility không? ___
- [ ] UNDIAL surgical forgetting: forget_quality cao không? ___
- [ ] DPO preference-based vs NPO: ___
- [ ] Method nào geometry-aware nhất (least utility drop)? ___
- [ ] Ghi chú: ___

---

## EXP_04 — ER-BWP Novel Method (Entropy-Reservoir BW Projection)

**Config**: ERBWP lambda sweep + PDU/SatImp comparison  
**Run date**: ___  
**Kaggle notebook**: ___

### Part A: Lambda Sweep

| Config | λ_max | λ_min | decay | model_utility | forget_quality | privleak | extraction_strength | Composite |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| lmax0.1_d1.0 | 0.1 | 0.001 | 1.0 | — | — | — | — | — |
| lmax0.5_d1.0 | 0.5 | 0.001 | 1.0 | — | — | — | — | — |
| lmax1.0_d1.0 | 1.0 | 0.001 | 1.0 | — | — | — | — | — |
| lmax0.5_d0.5 | 0.5 | 0.001 | 0.5 | — | — | — | — | — |

### Part B: Comparison vs PDU/SatImp

| Method | model_utility | forget_quality | privleak | extraction_strength | Composite |
|--------|:---:|:---:|:---:|:---:|:---:|
| PDU | — | — | — | — | — |
| SatImp | — | — | — | — | — |
| **ERBWP_best** | — | — | — | — | **—** |

**Best λ config**: ___

**So sánh với EXP_01 NPO (same family)**:

| | NPO (EXP_01) | ERBWP_best | Delta |
|---|:---:|:---:|:---:|
| utility | — | — | — |
| forget_quality | — | — | — |
| privleak | — | — | — |

**Insight**:
- [ ] λ_max tối ưu là bao nhiêu? ___
- [ ] Entropy coupling có giúp utility không? (lmax0.5 vs lmax0.1): ___
- [ ] Quá nhiều entropy (lmax1.0) → over-regularize? ___
- [ ] Fast decay (d0.5) vs slow (d1.0): ___
- [ ] ERBWP có vượt SimNPO về composite không? ___
- [ ] **Key finding cho paper**: ___

---

## EXP_05 — C-SAES Novel Method (Contrastive Sparse Autoencoder Suppression)

**Config**: CSAES sweep target_layer × proj_coeff  
**Run date**: ___  
**Kaggle notebook**: ___

| Config | layer | proj_coeff | contrast_coeff | model_utility | forget_quality | privleak | extraction_strength | Composite |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| L-4_p0.5 | -4 | 0.5 | 0.3 | — | — | — | — | — |
| L-8_p1.0 | -8 | 1.0 | 0.5 | — | — | — | — | — |
| L-12_p1.0 | -12 | 1.0 | 0.5 | — | — | — | — | — |
| L-8_p2.0 | -8 | 2.0 | 0.5 | — | — | — | — | — |
| L-8_p0.5 | -8 | 0.5 | 0.5 | — | — | — | — | — |

**Best config**: ___

**So sánh với EXP_01 + ERBWP**:

| | SimNPO | ERBWP_best | **CSAES_best** |
|---|:---:|:---:|:---:|
| utility | — | — | — |
| forget_quality | — | — | — |
| Composite | — | — | — |

**Insight**:
- [ ] Layer tối ưu (-4 / -8 / -12): ___
- [ ] Late layer (-4) có aggressive không? ___
- [ ] proj_coeff=2.0 vs 1.0 vs 0.5: over-suppress / balanced / under-suppress: ___
- [ ] Forget direction cosine similarity (forget vs retain): ___
- [ ] C-SAES có surgical hơn GradAscent không? (higher utility): ___
- [ ] **Key finding cho paper**: ___

---

## EXP_06 — MGFAA Novel Method (Mechanistic Gradient-Free Activation Ablation)

**Config**: MGFAA sweep steer_layer × steer_alpha (GRADIENT FREE)  
**Run date**: ___  
**Kaggle notebook**: ___

> **Note**: MGFAA không update weights — calibration ~50 forward passes, cực nhanh

| Config | layer | alpha | normalize | model_utility | forget_quality | privleak | extraction_strength | Composite |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| L-4_a5 | -4 | 5 | yes | — | — | — | — | — |
| L-4_a15 | -4 | 15 | yes | — | — | — | — | — |
| L-8_a5 | -8 | 5 | yes | — | — | — | — | — |
| L-8_a15 | -8 | 15 | yes | — | — | — | — | — |
| L-8_a30 | -8 | 30 | yes | — | — | — | — | — |
| L-12_a15 | -12 | 15 | yes | — | — | — | — | — |
| L-16_a15 | -16 | 15 | yes | — | — | — | — | — |
| L-8_a15_nonorm | -8 | 15 | **no** | — | — | — | — | — |

### Heatmap (Layer × Alpha — Composite Score)

|  | α=5 | α=15 | α=30 |
|---|:---:|:---:|:---:|
| layer=-4 | — | — | — |
| layer=-8 | — | — | **← best?** |
| layer=-12 | — | — | — |
| layer=-16 | — | — | — |

**Best config**: ___

**So sánh toàn bộ methods (summary)**:

| | SimNPO | ERBWP_best | CSAES_best | **MGFAA_best** |
|---|:---:|:---:|:---:|:---:|
| utility | — | — | — | — |
| forget_quality | — | — | — | — |
| Composite | — | — | — | — |
| Training time | full | full | full | **~2min** |

**Insight**:
- [ ] Optimal layer: ___
- [ ] Optimal alpha: ___
- [ ] normalize=True vs False (L-8_a15): ___
- [ ] Alpha trop cao (30) → incoherent output? ___
- [ ] MGFAA có đủ mạnh với TOFU task không? ___
- [ ] Trade-off speed vs quality vs gradient methods: ___
- [ ] **Key finding cho paper**: ___

---

## Cross-EXP Comparison Table

> Fill sau khi chạy hết, dùng best config của mỗi EXP.

| Method | EXP | Category | model_utility | forget_quality | privleak | extraction_strength | **Composite** | Notes |
|--------|-----|----------|:---:|:---:|:---:|:---:|:---:|-------|
| GradAscent (forget10) | EXP_00 | Baseline | — | — | — | — | — | |
| GradDiff | EXP_01 | Standard | — | — | — | — | — | |
| NPO | EXP_01 | Standard | — | — | — | — | — | |
| SimNPO | EXP_01 | Standard | — | — | — | — | — | Paper best |
| RMU | EXP_01 | Standard | — | — | — | — | — | |
| NPO-seq (step3) | EXP_02 | Sequential | — | — | — | — | — | |
| WGA | EXP_03 | Community | — | — | — | — | — | |
| UNDIAL | EXP_03 | Community | — | — | — | — | — | |
| DPO | EXP_03 | Community | — | — | — | — | — | |
| PDU | EXP_04 | Community | — | — | — | — | — | |
| **ERBWP** (best λ) | EXP_04 | **Novel** | — | — | — | — | — | **Ours** |
| **CSAES** (best) | EXP_05 | **Novel** | — | — | — | — | — | **Ours** |
| **MGFAA** (best) | EXP_06 | **Novel** | — | — | — | — | — | **Ours** |

---

## Paper Narrative (fill sau khi có số)

### Main Claim

> _"Our proposed methods (ER-BWP, C-SAES, MGFAA) achieve [X] while [Y demonstrated improvement] compared to [Z baseline]."_

**Draft**: ___

### Key Findings

1. **ER-BWP vs NPO**: Entropy coupling λ(t) ___
   - Effect on utility: ___
   - Effect on forget_quality: ___

2. **C-SAES surgical precision**: Activation-space intervention ___
   - Best layer: ___ — why this makes sense: ___
   - Comparison to weight-space methods: ___

3. **MGFAA gradient-free efficiency**: ___
   - Speed: ___ vs gradient methods: ___
   - Quality trade-off: ___

4. **Sequential robustness** (EXP_02 evidence):
   - Degradation magnitude: ___
   - Which method is most robust: ___

### Ablation Insights

| Ablation | Variable | Finding |
|----------|----------|---------|
| ERBWP λ_max | 0.1 vs 0.5 vs 1.0 | — |
| ERBWP decay | 0.5 vs 1.0 | — |
| CSAES layer | -4 vs -8 vs -12 | — |
| CSAES proj_coeff | 0.5 vs 1.0 vs 2.0 | — |
| MGFAA alpha | 5 vs 15 vs 30 | — |
| MGFAA normalize | True vs False | — |

### Limitations Observed

- [ ] ___
- [ ] ___
- [ ] ___

---

## Run Log

| Date | EXP | Config tag | Kaggle URL | Status | Notable |
|------|-----|-----------|------------|--------|---------|
| | EXP_00 | GradAscent baseline | | ⬜ pending | |
| | EXP_01 | SimNPO/RMU/NPO/GradDiff | | ⬜ pending | |
| | EXP_02 | NPO sequential chain | | ⬜ pending | |
| | EXP_03 | WGA/UNDIAL/DPO | | ⬜ pending | |
| | EXP_04 | ERBWP lambda sweep | | ⬜ pending | |
| | EXP_05 | CSAES layer×proj sweep | | ⬜ pending | |
| | EXP_06 | MGFAA layer×alpha sweep | | ⬜ pending | |

**Status legend**: ⬜ pending · 🔄 running · ✅ done · ❌ failed · 🔁 rerun needed

---

## Raw JSON Dump (paste output từ `saves/eval/EXP_XX_results.json`)

<details>
<summary>EXP_00</summary>

```json
{}
```
</details>

<details>
<summary>EXP_01</summary>

```json
{}
```
</details>

<details>
<summary>EXP_02</summary>

```json
{}
```
</details>

<details>
<summary>EXP_03</summary>

```json
{}
```
</details>

<details>
<summary>EXP_04</summary>

```json
{}
```
</details>

<details>
<summary>EXP_05</summary>

```json
{}
```
</details>

<details>
<summary>EXP_06</summary>

```json
{}
```
</details>
