# Chapter 26 — Inside a Decision Model: BERT, Laya and Vega

## Introduction

Chapter 25 explained *what* a decision model is: a state, a question and a list of allowed answers go in, and one probability per answer comes out, in a single pass and without generating text. This chapter and the next explain *how one is built*. This chapter is about the **architecture**. The next is about **data, training, calibration and evaluation**.

We take two real designs apart, because they make opposite choices from the same starting point:

- **Laya**, an open model built on a small **bidirectional encoder** (a BERT descendant). Everything about it is public: weights, code, training loop, benchmark scripts.
- **Vega**, an open model built on a large **causal decoder** (a Qwen language model) with a LoRA adapter and a small readout head.

Laya is the main worked example, so we start where it starts: with BERT. The handbook has only sketched BERT so far (Chapter 15, Part 4), and a decision model leans on details that sketch left out: what `[CLS]` and `[SEP]` are, how BERT is pretrained, how a head is added on top.

The parts of this chapter:

1. **Encoders first:** BERT properly. Why it exists, its input, the `[CLS]` token, pretraining, fine-tuning, and ModernBERT.
2. **Laya, layer by layer:** the input layout, the `[MASK]` slots, the head, and the size of every piece.
3. **Vega, the decoder way:** a different answer to the same problem, and a speed comparison.

> **Sources and numbers.** Facts about Laya come from its repository, model cards and documentation; facts about Vega from its model card and code; facts about BERT and ModernBERT from their papers. Where a claim is the authors' own, the text says so. Small worked examples use illustrative numbers. Some of them were produced by short programs, included next to this chapter as `Chapters-26-27-experiments.js`, so that you can rerun them (`node Chapters-26-27-experiments.js all`).

---

## Part 1 — Encoders first: BERT, properly

### 1.1 Why an encoder-only model exists

**The problem.** A GPT reads left to right. When it processes "bank" in "the **bank** of the river", it has not yet seen "river". That is exactly what the causal mask of Chapter 4 enforces, and it is what makes generation possible. But suppose we are *not* generating. Suppose we only want to understand a complete piece of text, for instance to say "this email is about billing".

**Why the previous step is not enough.** The causal mask throws information away: early tokens never see later ones.

**The fix: remove the mask.** Let every token look at every other token, in both directions.

```text
GPT (causal mask)                 BERT (no mask)
 each row = a token,               every token sees every token
 each ✓ = "can look at"

        t1 t2 t3 t4                       t1 t2 t3 t4
   t1   ✓                            t1   ✓  ✓  ✓  ✓
   t2   ✓  ✓                         t2   ✓  ✓  ✓  ✓
   t3   ✓  ✓  ✓                      t3   ✓  ✓  ✓  ✓
   t4   ✓  ✓  ✓  ✓                   t4   ✓  ✓  ✓  ✓
```

**What it costs.** With no mask, a token cannot be "the next word to predict", because the answer is already visible. A BERT cannot write text. It produces **one understanding vector per input token**, and you build on those vectors.

The architecture is Chapters 4 to 7 with the mask removed: attention, feed-forward network, residuals and LayerNorm, stacked. The original BERT-base has 12 layers, a hidden size of 768, 12 attention heads and about 110 million parameters. BERT-large has 24 layers, a hidden size of 1,024, 16 heads and about 340 million. Its vocabulary is 30,000 WordPiece tokens, and it was pretrained on BooksCorpus (800 million words) and English Wikipedia (2,500 million words).

### 1.2 What BERT is used for

BERT is a **reader**, not a writer. Its usual output is a vector, and what you do with the vectors depends on the task.

**Translation: no.** Translation needs output of a different length in another language, written one token at a time. BERT has no way to write. That job belongs to encoder–decoder models and decoder LLMs (Chapter 15).

**Fill in the blank: yes.** That is what BERT is trained on (Part 1.5). It is rarely a product by itself. By learning to fill blanks across billions of words, BERT ends up with good vectors for everything else.

The real jobs, from the BERT paper and its benchmarks:

| Task | Input shape | Where the answer is read from |
|---|---|---|
| Sentiment, spam, intent | **1 text** | the `[CLS]` vector, into a small classifier |
| Named-entity tagging ("which words are people or places?") | **1 text** | each token's vector, one label per token |
| Fill in the blank | **1 text** with a `[MASK]` | the vector at the `[MASK]` position |
| Embedding one text for search | **1 text** | one pooled vector |
| Question answering from a passage | **question + passage** | per-token scores for where the answer starts and ends |
| Does sentence A imply sentence B? | **2 texts** | `[CLS]`, a three-way classifier |
| Are these two questions duplicates? | **2 texts** | `[CLS]`, yes/no |
| Re-ranking search results | **query + document** | `[CLS]`, a relevance score |

BERT always takes **one input sequence**, which holds one text or two. Most uses need only one. Two-text inputs are used when the *task itself* is about the relationship between two texts. The paper describes the single-text case as "a degenerate text-∅ pair": a pair with nothing in the second half.

> **A common question: "Isn't BERT just an LLM without the decoder?"** Close, with one hidden trap. The blocks inside GPT and the blocks inside BERT are almost the same. The difference is the **causal mask** (Chapter 15, Part 2):
>
> ```text
> GPT  = Transformer blocks + causal mask (cannot look ahead)  + a head that predicts the next word
> BERT = Transformer blocks, no mask       (sees everything)   + heads that read vectors out
> ```
>
> The "decoder" of the original Transformer did two extra things: it used a causal mask *and* it attended to an encoder's output (cross-attention). GPT-style models kept the mask and dropped the cross-attention. So BERT is not "an LLM minus a part". It is the same blocks with the mask removed, which gives up writing and gains seeing everything.

### 1.3 What goes in: tokens, `[CLS]`, `[SEP]`, segments, positions

**The problem.** The model takes a list of tokens. BERT is used on a single text or on a **pair**. How does it tell the pieces apart, and where does the *summary* of the whole input come from?

**The fix: three small ideas.**

```text
[CLS]  I  was  charged  twice  [SEP]  please  refund  it  [SEP]
  │    └────── sentence A ──────┘      └─── sentence B ───┘
  │
  special first token: the "summary slot"
```

1. **`[CLS]`** is always the first token. It is a placeholder, not a word. The BERT paper says its final hidden state serves as the summary of the whole sequence for classification tasks.
2. **`[SEP]`** separates sentence A from sentence B.
3. **A segment embedding.** A learned vector added to every token saying "I belong to sentence A" or "sentence B".

What the model actually receives per token is the sum of three vectors of the same size (768 for BERT-base):

```text
input vector =  token embedding    (which word piece: a row in the embedding table, Chapter 3)
              + segment embedding  (A or B: one of two learned vectors)
              + position embedding (which slot: 1st, 2nd, 3rd ...)
```

An illustrative sum, with 4 numbers instead of 768:

```text
token "charged"           [ 0.50   0.10  -0.30   0.20]
+ position (3rd slot)     [ 0.05  -0.10   0.20   0.00]
+ segment A (row 0)       [ 0.10  -0.20   0.00   0.30]
= input to layer 1        [ 0.65  -0.20  -0.10   0.50]
```

**Fixed, learned or computed?**

| Thing | Kind |
|---|---|
| The `[CLS]` token's *input* vector (plus position and segment A) | **learned**, then the same for every input |
| The segment A and B vectors | **learned** |
| The `[CLS]` position's *output* vector after the last layer | **computed** freshly for each input |

> **A common question: "Where do segment embeddings come from?"** Three separate things happen:
>
> | Step | What happens | Learned? |
> |---|---|---|
> | Deciding the label | The data-preparation code numbers each token: everything up to and including the first `[SEP]` gets 0 (A), everything after gets 1 (B). | **no**, plain code |
> | The two vectors | A tiny table with **2 rows**, each as wide as the token embedding. It starts random and is trained along with everything else. | **yes** |
> | Using it | Look up the row for the token's label and **add** it to the token and position vectors. | computed |
>
> It is the same idea as the position table of Chapter 3, with a different table: about 30,000 rows for tokens, up to 512 for positions, and 2 for segments. **If you feed BERT a single sentence, the segment embedding is still used**: every token is labelled A, so the same vector is added to every token, and it tells the model nothing. Training teaches the model to use the two vectors because pair tasks ("do these sentences go together?", "which part of the passage answers the question?") cannot be solved without knowing which token belongs to which part.
>
> Later encoders found they can do without it. ModernBERT drops BERT's pair-matching task, and Laya's code passes only token ids and an attention mask to the encoder, with no segment ids.

### 1.4 What `[CLS]` really is

**The problem.** BERT gives one vector per token. But many tasks need *one* vector for the whole text. How do we get a single summary from many vectors?

**Why the obvious options are weak.** Using one real word's vector biases the summary toward that word. Averaging all the vectors works, and is a common choice, but treats every token equally and has nothing in it that is *trained for the task*.

**The fix: a slot with no word of its own.** `[CLS]` is an ordinary token whose only job is to give a summary somewhere to live. Because it carries no word meaning, nothing biases it, and because BERT has no mask, it can look at every token in every layer.

**Why it is placed *first*.** The position does not matter to attention, so first is just a convention. In a GPT the situation is different: with the causal mask only the **last** token has seen everything, so a GPT-style classifier reads the last token. In BERT every token has seen everything, so any token could hold the summary, and a dedicated neutral one is used.

**How it becomes a summary: one attention step.** Take one head in one layer, with made-up 2-number vectors so the arithmetic can be followed:

```text
tokens:                  [CLS]     charged    twice     refund
CLS asks (its query):    [1.0, 0.5]

key of each token:       [0.2,0.1] [0.9,0.3]  [0.5,0.4] [0.8,0.8]
score = q·k / √2:         0.177     0.742      0.495     0.849
softmax → weights:        0.164     0.289      0.226     0.321      (sum = 1)

value of each token:     [0.1,0.0] [1.0,0.2]  [0.4,0.6] [0.2,1.0]

new [CLS] = Σ weight × value = [0.46, 0.51]
```

This is Chapter 4 applied to one position. The new `[CLS]` vector is a blend of the other tokens, weighted by relevance (it also looks at itself, with weight 0.164). The same thing happens at every position in parallel. After 12 layers with 12 heads in each, the final vector `C` has absorbed information from the whole sentence in many layered ways.

**Why it becomes a *useful* summary: the loss.** Nothing in the architecture forces `[CLS]` to summarise well. A loss is attached to it. In pretraining, `C` is used for next-sentence prediction; in fine-tuning, a classification layer reads `C` with the loss `log(softmax(C·Wᵀ))` (Part 1.6). Backpropagation (Chapter 10) then changes the attention weights so that `[CLS]` pulls in whatever the task needs. A sentiment task teaches it to gather sentiment cues; a topic task teaches different cues. The slot is the same, and what it learns to hold is whatever the loss rewards.

**A warning: an untrained `[CLS]` is a poor summary.** The Sentence-BERT paper reports that taking BERT's raw `[CLS]` output, or averaging its outputs, gives poor sentence embeddings, often worse than simply averaging older GloVe word vectors. The summary becomes good only after training with a loss that rewards good sentence vectors.

> **A common question: "Is a dedicated summary token needed in a GPT classifier too?"** A first answer ("needed in BERT, not in GPT") overstates it. A dedicated token helps in both kinds of model. What differs is **where it can sit**:
>
> ```text
> BERT, token FIRST:               GPT, token FIRST (a [CLS] copy):
>  [CLS] the cat sat                [CLS] the cat sat
>   ✓     ✓   ✓   ✓  sees all       ✓     ✗   ✗   ✗   sees only itself: useless
>
> GPT, token LAST (an end/summary token):
>  the cat sat [SUM]
>   ✓   ✓   ✓    ✓   the only position that can see everything
> ```
>
> Decoders do use such tokens. GPT-1 (2018) added randomly initialised start and end tokens to every task input, and fed the last position's final activation to a linear output layer. E5-Mistral turns a decoder LLM into an embedding model by appending an `[EOS]` token and taking the last layer's vector at that token. A "pause token" paper appended learnable tokens to give a decoder extra computation before answering, with gains reported only when the model was both pretrained and fine-tuned with them.
>
> Why a dedicated token helps, even in a GPT (the usual explanations): it is **neutral** (the last real word's vector is busy predicting the next word, so a summary competes with that job), **consistent** (always the same token), and **extra working space**. The trade-off is that a brand-new token in a model that has never seen it starts untrained. Vega, in Part 3, uses exactly this idea: the last token of its prompt is the "summary" the head reads.

### 1.5 Pretraining BERT: what `[MASK]` means in BERT

**The problem.** BERT sees the whole sentence, so it cannot be trained to predict the next word: the answer would be sitting in its input. We need a training task that needs **no human labels** and **forces the model to use both left and right context**.

**The fix: fill in the blanks.** This is the old "cloze test" from language classes. Hide some words and make the model guess them from the words on both sides. Any text becomes training data, because the hidden word is its own label.

**How it works**, on "my dog is hairy":

1. **Choose positions.** Pick 15% of the tokens at random. Say it picks "hairy".
2. **Corrupt them, with a three-way split** (the paper's recipe):

| Chance | What replaces the chosen token | Result |
|---|---|---|
| 80% | the `[MASK]` token | my dog is **[MASK]** |
| 10% | a random word | my dog is **apple** |
| 10% | nothing (kept as is) | my dog is **hairy** |

3. **Run the whole sequence through BERT.** No mask in the attention, so the vector at the chosen position has seen words on both sides.
4. **Read the vector at the chosen position** and send it through an output layer that produces a probability for each of the 30,000 vocabulary words ("as in a standard LM", the paper says).
5. **Score with cross-entropy** (Chapter 9) against the original word. With illustrative numbers: if the model gives "hairy" 0.62, the loss is −ln 0.62 = 0.48 (a good guess); if it gives 0.05, the loss is 3.00.

Only the chosen positions count towards the loss. The other 85% of tokens are read but not scored.

**Why the odd 80/10/10 split?** The paper gives two reasons.

- **A mismatch problem.** `[MASK]` never appears in real text, so it will not appear when BERT is later used for classification. If the model saw `[MASK]` only at the positions it had to predict, it could learn "I only need to think hard at `[MASK]`". Keeping or randomising some chosen words breaks that habit.
- **No hiding place.** As the paper puts it, the model cannot tell which words it will be asked to predict or which were swapped for random ones, so it has to keep a useful contextual representation of every input token. Random replacement touches only 1.5% of all tokens (10% of 15%), which the paper says does not seem to hurt understanding.

**The second task: next-sentence prediction.** BERT also trained on pairs of sentences. In 50% of examples sentence B really follows A (label "IsNext"); in the other 50% it is a random sentence ("NotNext"). The `[CLS]` output `C` predicts which. That is the loss attached to `[CLS]` in pretraining. The training loss is the sum of the two: the mean masked-word loss plus the mean next-sentence loss.

**The scale.** BERT-base trained for 1,000,000 steps, with 256 sequences of 512 tokens per step (128,000 tokens per step), about 40 passes over a 3.3-billion-word corpus, with Adam at a learning rate of 1e-4, a 10,000-step warm-up and linear decay. BERT-base used 4 Cloud TPUs in a Pod configuration (16 chips); BERT-large used 16 (64 chips). That is Chapter 11, applied to a real model.

> **A common question: "Is Laya's `[MASK]` the same as BERT's?"** No, and this is the most common confusion in this topic. In BERT, `[MASK]` is a **blank**: something was hidden there, and the model must guess it. In Laya (Part 2), `[MASK]` is a **slot** placed in front of each option: nothing was hidden, and the output vector at that position is read to score the option. Laya never asks BERT to guess a word under the mask. One plausible reason for reusing the token is that the encoder spent pretraining learning to produce a useful, context-aware vector at exactly that kind of position. The authors do not state this reason, so treat it as an inference.

### 1.6 Fine-tuning: putting a head on a pretrained BERT

**The problem.** After pretraining, BERT gives a good vector for every token, but it cannot *answer* anything. How do we turn a general reader into a specific task solver, with only a few thousand labelled examples?

**The fix, in three moves.**

1. **Start from the pretrained weights**, not from random ones.
2. **Add a tiny new output layer**, the "head".
3. **Train everything**, BERT's layers and the head, on your labelled examples, with a small learning rate.

The paper says fine-tuning needs only one extra output layer, and that every parameter is updated. Each task gets its own fine-tuned copy, all starting from the same pretrained weights.

**The head changes shape with the task.**

| Task | What the head reads | What it outputs |
|---|---|---|
| Classify a whole text (sentiment, intent, entailment) | the `[CLS]` vector `C` | probabilities over K classes |
| Tag each word (names, places) | every token's vector `Tᵢ` | one label per token |
| Pull an answer out of a passage | every passage token's vector | a start position and an end position |

For question answering the paper adds two learned vectors, `S` (start) and `E` (end). The score for an answer spanning tokens i to j is `S·Tᵢ + E·Tⱼ`, with j ≥ i. For "there is no answer", the paper scores the span that starts and ends at `[CLS]`.

**One worked example: the classification head.** With K = 3 classes (billing, technical, other), the head is a table `W` with one row per class (`W` is **learned**; `C` is **computed** for each text). Illustrative numbers with a 4-number vector instead of 768:

```text
C (the [CLS] output)     = [ 0.6, -0.2,  0.9,  0.3]

W row "billing"          = [ 0.5,  0.1,  0.4, -0.2]   →  logit  0.58
W row "technical"        = [-0.3,  0.6,  0.0,  0.5]   →  logit -0.15
W row "other"            = [ 0.2, -0.4, -0.5,  0.1]   →  logit -0.22

softmax → probabilities    billing 0.518, technical 0.250, other 0.233
```

If the true label is "billing", the loss is −ln 0.518 = 0.66 (the paper's `log(softmax(C·Wᵀ))`, Chapter 9). If it were "technical" the loss would be 1.39. Backpropagation starts from the simple signal "probability minus answer":

```text
gradient on the logits:     billing −0.48 (push UP),  technical +0.25, other +0.23 (push DOWN)
gradient on W's billing row = −0.48 × C = [−0.29, 0.10, −0.43, −0.15]
```

The signal does not stop at `W`. It keeps flowing back through `C` into every attention and feed-forward weight in BERT. That is what "train everything" means: the encoder is gently reshaped so `[CLS]` collects the evidence the task needs.

**Why it works so well and so cheaply.**

- **Almost nothing is new.** `W` has 3 × 768 = 2,304 numbers, against about 110 million in BERT-base: roughly 0.002%.
- **It is fast.** The paper says all of its results can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the same pretrained model.
- **Small learning rates.** The recipe uses learning rates of 5e-5, 3e-5 or 2e-5, batch sizes of 16 or 32, and 2 to 4 epochs. Pretraining used 1e-4. Small steps nudge the pretrained weights without overwriting what they learned.

**The limitation that matters for this chapter.** A head with K rows has its classes **baked in**. Add a class and you need a new row and retraining. A decision model needs a head whose number of outputs changes with every request. That is what Part 2 builds.

### 1.7 Modern encoders: what ModernBERT changes

BERT dates from 2018. Since then, decoder LLMs developed better building blocks that encoders mostly did not receive. The ModernBERT paper (December 2024) describes encoders as the "workhorse of numerous production pipelines" that saw "limited Pareto improvements" after BERT, and applies the modern parts to an encoder. It is still BERT in spirit: no causal mask, `[CLS]`, `[SEP]`, `[MASK]`, fill-in-the-blank training. What changed is the machinery, much of which this handbook already covers.

| Part | BERT (2018) | ModernBERT | Why | Handbook |
|---|---|---|---|---|
| Positions | learned absolute table, 512 slots | **RoPE** | proven in short- and long-context models, "ease of context extension" | Chapter 16 |
| Norm placement | post-norm | **pre-norm**, plus one extra LayerNorm after the embeddings | stabilises training | Chapters 7, 16 |
| FFN activation | GELU | **GeGLU**, a gated FFN | consistent gains in recent work | Chapter 16 |
| Bias terms | present | **removed** in linear layers (except the final output layer) and in LayerNorm | spend the parameters on weights that matter | Chapter 16 |
| Attention | every layer global | **every third layer global, the rest a 128-token sliding window** | long inputs get cheap | Chapter 17 |
| Speed tricks | padded batches | **unpadding**, FlashAttention 3 (global layers) and 2 (local), `torch.compile` | no wasted compute on padding | Chapter 17 |
| Shape | 12 layers, wide | **22 layers (base), 28 (large)**, "deep and narrow" | better quality at the same size, sized to common GPUs | |
| Tokenizer | WordPiece, 30,000 | modified OLMo **BPE, 50,368 tokens**, same special tokens | better token efficiency and code handling | Chapter 2 |
| Context | 512 | **8,192 natively** | | |

Two details are worth keeping. The tokenizer keeps BERT's special tokens (`[CLS]`, `[SEP]`, `[MASK]`) and templating for backwards compatibility, so everything in Parts 1.3 to 1.6 still applies. And the sliding-window layers matter for Laya's code: when it builds its own attention masks, it passes the encoder's sliding-window setting along so its mask respects the local layers.

| Training | BERT | ModernBERT |
|---|---|---|
| Data | ~3.3 billion words | **2 trillion tokens** (web, code, science) |
| Masking | 15% | **30%** |
| Next-sentence task | yes | **dropped** |
| Optimiser | Adam | **StableAdamW** |
| Learning-rate schedule | warm-up, then linear decay | **warm-up, a long constant stretch, a short decay** |
| Base-model rate | 1e-4 | 8e-4 |

(Words and tokens are different units, so treat the data comparison as rough, but it is several hundred times more text.) The large model was initialised from the trained base model rather than from random weights.

**Sizes, with a check you can reproduce.** Base: 22 layers, hidden size 768, 149M parameters. Large: 28 layers, hidden size 1,024, 395M. The paper's GLU expansion figures, 2,304 for base and 5,248 for large, give the width of the gated input projection (two halves). With that reading, the arithmetic for the large model is:

```text
per layer:  attention 4 × 1024²                      = 4.19M
            gated FFN: 1024 × 5248 + 2624 × 1024      = 8.06M
            total per layer                           = 12.26M      × 28 layers = 343M
            + embeddings 50,368 × 1024                = 51.6M
            ≈ 395M      matches the paper

base:       5.01M × 22 = 110M + embeddings 38.7M ≈ 149M   matches
```

The other reading of "expansion" gives 207M and 620M, which do not match, so this interpretation is the one that reproduces the paper.

**What the authors report, and admit.** On classification and retrieval, ModernBERT reaches state-of-the-art results among encoders of similar size, processing short inputs about twice as fast as DeBERTaV3 and long inputs about twice as fast as the next fastest model (the authors' own benchmarks). Stated limits: English only, trained largely on web data and so inheriting its biases, trained with fill-in-the-blank only (they suggest adding a second objective), and not a text generator.

### 1.8 The mask is a dial, not a switch

Because the only structural difference between BERT and GPT is the mask, it is tempting to treat encoders and decoders as two separate species. Research suggests they are better seen as two ends of a dial.

- **One network with several mask patterns.** UniLM trains one shared Transformer with bidirectional, left-to-right and sequence-to-sequence masks; GLM and the UL2 family read the prompt bidirectionally and write the output left to right.
- **Decoder to encoder works with adaptation.** LLM2Vec replaces a decoder's causal mask with an all-ones matrix, adds a short masked-prediction stage and contrastive learning, and reports strong text encoders (a new unsupervised state of the art on the MTEB embedding benchmark).
- **Converting one kind into the other is only partly successful.** The Ettin study trained encoders as decoders and decoders as encoders for 50 billion extra tokens each way. The converted models did not surpass models trained on the target objective from the start.

The dial reappears in this chapter: Laya uses a custom mask so that its options cannot attend to each other (Part 2), and Vega's choice of a causal backbone decides where it must put the state and the readout (Part 3).

> **A note on embedding models.** Embedding models (Sentence-BERT, E5, BGE, and large ones such as Gemini Embedding) are also encoders, with the same kind of layers, trained with a different objective: texts that mean the same thing should land close together, rather than texts being classified. They are what Chapter 25, Part 5 compares decision models against.

---

## Part 2 — Laya, layer by layer

### 2.1 Where it came from

Laya did not appear from nowhere. Its author, Nandakishor M, had published two earlier papers on related ideas, and Laya generalises both.

| Date | Work | Core idea | Notes |
|---|---|---|---|
| 30 March 2025 | *SalesRLAgent* (arXiv:2503.23303) | Treat "will this sale convert?" as a sequential decision, with a policy network producing a conversion probability, a value network supporting reinforcement learning, and a module estimating its own confidence. | The input is **frozen** 3,072-dimension embeddings from an Azure OpenAI model, so no reinforcement learning ever reached a transformer. Trained on **synthetic GPT-4o conversations**, in about 6 hours on a CPU. The paper reports 96.7% accuracy and 85 ms against 3,450 ms for GPT-4 (self-reported). |
| 23 September 2025 | *Confidence-Aware Routing* (arXiv:2510.01237) | Estimate, **before** a language model answers, whether it is likely to hallucinate, and route the query to local generation, retrieval, a larger model or a human. | Combines three signals (agreement with a reference embedding, how consistently the layers converge, and a learned confidence estimator). Tested on a 360M-parameter model. The paper lists its own limits: dependence on the reference model, fixed thresholds, domain specificity, and small scale. |
| September 2026 | **Laya** | Make probabilities a typed, request-time product, trained **end to end** on an encoder. | The act/escalate head descends from the routing paper; the probability outputs from the sales model. |

### 2.2 The input: one sequence per question

Chapter 25 showed a decision as three fields. Laya turns **each question** into one text sequence. For the department question about our ticket, the layout is:

```text
[CLS] choice question: Which department should handle this? [SEP]
[MASK] billing: invoices, payments, refunds
[MASK] technical: bugs, outages, system errors
[MASK] other: everything else [SEP]
Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan. [SEP]
└──────────── the question and options ────────────┘ └────────────── the state ───────────────┘
```

The rules, taken from the code:

- `type` and `instructions` become the first line: "choice question: …".
- Each `criteria` entry becomes one line behind its own **`[MASK]` marker**, written "label: description". For a `score` question the lines read "level 0: not urgent", "level 1: soon", "level 2: blocking". For a `noul` question there are always two lines, "false: no, the statement does not hold" and "true: yes, the statement holds" (unless you override the wording).
- The question and options share a **budget**: 192 tokens in the English checkpoint, 256 in the others. Each option is cut off at 48 tokens, and when there are too many options each gets a smaller share. This is where the "roughly 20 options" limit comes from.
- Whatever remains of the context window (512 or 1,024 tokens) goes to the **state**, cut if necessary.
- If a request asks several questions about one state, the state is **tokenised once** and reused. Each question is still its own row of a batch. "All the questions are answered in one forward pass" means one *batched* pass, not one giant sequence.

### 2.3 The `[MASK]` slots

Each `[MASK]` is a **slot**: a position with no word of its own, placed before each option. This is the same trick as `[CLS]` (Part 1.4): *a token with no word of its own, whose output vector is read out for a decision.* Where `[CLS]` is one slot for the whole text, Laya gives **each option its own slot**.

Because the encoder is bidirectional, the slot in front of "billing" can see the question, its own option text, the other options and the state, all at once. After the last layer, its vector summarises "this option, in this question, against this state". A small scorer then turns each slot vector into one number.

> **A common question: "Does BERT's masked training still happen in Laya?"** The base encoder was pretrained with fill-in-the-blank, long before Laya. Laya's own training options are a reinforcement-style objective or plain cross-entropy (Chapter 27, Part 2). Its code contains no masked-word objective.

### 2.4 The pipeline, with shapes

```text
tokens (say 60)                                     shape [60]
  │
  ▼ ModernBERT encoder (28 layers)                  shape [60 × 1024]
  │   one final hidden vector per token. These are the encoder's last hidden
  │   states, NOT its word-prediction layer.
  ▼ + a learned "question type" vector              (a table of 3 rows: choice / score / noul),
  │   added to every position
  ▼ 2 extra Transformer layers (the "head")         shape [60 × 1024]
  │   attention over all positions, trained from scratch, pre-norm, no positional encoding
  ▼ pick out the vector at each [MASK] position     shape [3 × 1024]     (3 options)
  │
  ▼ scorer: LayerNorm → Linear → GELU → Linear      shape [3]            one logit per option
  │
  ▼ softmax (after a calibrated temperature)        shape [3]            probabilities
```

A small worked step, with illustrative numbers. Three options give three slot vectors and three logits:

```text
logits [2.1, -0.4, -1.3]
  temperature 1.0 → [0.896, 0.074, 0.030]     sharp
  temperature 1.5 → [0.774, 0.146, 0.080]     softer: less over-confident
```

Dividing by a temperature above 1 flattens the distribution. Chapter 27, Part 4 returns to this.

**Fixed, learned or computed?**

| Piece | Kind |
|---|---|
| Where the `[MASK]` markers go | computed by plain code from your options |
| Encoder weights | learned in pretraining, then fine-tuned |
| Question-type vectors, the 2 head layers, the scorer | learned, from scratch, in fine-tuning |
| The calibration temperatures (one per question type) | fitted afterwards (Chapter 27, Part 4) |
| How many options there are | chosen by you, per request |

### 2.5 Why each piece exists

**The slots and the shared scorer.** The scorer is a *single* small network applied to every slot with the same weights. It cannot know which option it is looking at except through the slot's vector, so that vector has to carry "this option, in this question, against this state". Bidirectional attention provides exactly that. Because the scorer is shared, adding an option just adds a slot, and nothing in the model has a fixed number of outputs.

**The extra layers.** The public material does not give an ablation, so this is an inference. The encoder was pretrained to predict words. The two extra layers are new, cheap (about 25 million parameters) and trained for this task alone. They let the slot vectors attend to each other and to the state once more with decision-specific weights. The documentation does say that training only the head with a frozen encoder "gains much less" than a full fine-tune.

**What goes in and out of the head.** The head receives the encoder's **final hidden vectors**, one per token, plus the question-type vector. It hands back vectors of the same shape, and only the slot positions are read. Nothing comes from BERT's word-prediction layer.

**Order-blindness.** In a plain sequential layout, an option's place in the list changes its position vector, and the code reports that five identical options produced logits spread by 3.59. Laya's code has a **parallel layout** in which every option starts at the same position and masks stop options from attending to each other. The head has no positional encoding, so reordering the options just reorders the logits. This is the mask dial of Part 1.8 in action: a custom pattern, neither causal nor fully open. The README still lists "position sensitivity" among the limitations and suggests averaging over several orderings, so the design intent is clear, but a given checkpoint should not be assumed fully order-blind.

### 2.6 The size of every piece (a check against the model card)

Two head layers at width 1,024 cost about 25.2M parameters (12.6M each). The scorer adds about 1.05M, the act head about 0.26M, and the type table 3,072 numbers, about 26.5M in all. Add ModernBERT-large's 395M and the total is **about 421M**, matching the model card.

### 2.7 The act/escalate head, and its status

There is one more small head. It reads the `[CLS]` vector plus four summary numbers (the top probability, the gap between the top two, the entropy, and the option count) and is meant to output "act" or "escalate". The model card is candid that "`action.act_probability` carries no usable signal yet" (AUROC 0.30, which is worse than chance). So the head exists, but its output should not be used. Use the calibrated probabilities and your own threshold (Chapter 27, Part 4).

### 2.8 What you actually download

For a GPT-style model, the download is the trained weights. For Laya, the checkpoint folder holds:

- `model.safetensors`: the encoder and the head (843 MB for the English model, in half precision, which matches 421M parameters at two bytes each),
- the tokenizer files,
- `rl_agent_config.json`: the settings that must match training (the token budgets, the number of head layers, the act-head costs, **and the calibration temperatures**).

Calibration lives in that config, not in the weights, which matters in Chapter 27, Part 5.

### 2.9 Speed in practice

On a T4 GPU the model card reports about 39.5 ms for one question (English checkpoint) and 32.8 ms (multilingual), 158.6 ms and 72.3 ms for ten questions in one call, and 103 to 332 questions per second batched. On a CPU the README reports roughly 193 to 464 ms for the multilingual router. These are the authors' own figures; Part 3 compares them with Vega's.

### 2.10 Limits

- **Few options.** The shared option budget means that about 20 options is the practical ceiling. On Banking77 (77 labels) the model card reports 0.425 accuracy against Jev's 0.870, and both Laya checkpoints score exactly 0.425, which looks like a budget ceiling and not a capability gap.
- **Order sensitivity** (above).
- **Short context.** 512 to 1,024 tokens by default.
- **Text only.**
- **The `score` kind is its weakest**, per the model card (0.372 on a five-level sentiment benchmark), and, the card warns, `noul` can follow the wording of its option labels instead of the state.
- **The base checkpoints are weak zero-shot** on its own typed-decisions test (0.362, below a baseline that always guesses the common answer, 0.461). Fine-tuning does most of the work (Chapter 27, Parts 2 and 6).

---

## Part 3 — Vega, the decoder way

Laya answers the question "how do you get one probability per option?" with a bidirectional encoder and slots. Vega (the largest of the Decision 2.0 models, Chapter 25, Part 6.3) answers it with a **causal decoder**. Comparing the two shows how much the attention mask dictates the design.

### 3.1 The backbone

Vega is built on `Qwen/Qwen3.8-27B`. Its configuration (the model's own config file) shows:

| Property | Value |
|---|---|
| Layers | 64: **48 linear-attention** and **16 full-attention**, in a repeating pattern of three linear then one full |
| Hidden size | 5,120 |
| Attention heads | 24 query heads sharing 4 key/value heads (grouped-query attention) |
| Vocabulary | 248,320 tokens |
| Native context | 262,144 positions (Vega supports 32,768) |

The mixed linear and full attention is the hybrid design of Chapter 17, and the 24-versus-4 head split is grouped-query attention (GQA). Chapter 21 describes the Qwen family.

### 3.2 The input: state first

Vega builds a text prompt, with the **state first**:

```text
Context:
<state>

Task type: choice
Question:
<instructions>
Options:
<option>{"key": "billing", "description": "invoices, payments, refunds"}</option>
<option>{"key": "technical", "description": "bugs, outages, system errors"}</option>
...
Select the single option best supported by the context and instructions.
Decision:
```

The fields match Chapter 25: `type`, `instructions` and `criteria` (here as `key` plus `description`).

### 3.3 The readout: endpoints and a query

```text
tokens → Qwen backbone (causal, no vocabulary head, no KV cache) → one hidden vector per token
           │
           ├─ take the vector at the LAST token of each <option> block    ("endpoint" vectors)
           └─ take the vector at the LAST token of the whole prompt       (the "query" vector)
                     ▼
        score_i = key(LN(endpoint_i)) · query(LN(query)) / √256
                  + w · GELU( W_c·LN(endpoint_i) + W_q·LN(query) )
                     ▼
              softmax over the options → probabilities
```

A check of this reading: four projections of 5,120 → 256, plus a scalar and layer norms, give 5.26 million parameters. Stored in 32-bit floats that is 21.1 MB, which is exactly the size of the head file in the repository.

### 3.4 What Vega keeps from Qwen

A natural question: is only *part* of the Qwen system retained? The loader takes the language-model part of the full multimodal model and uses its final hidden vectors.

| Part of Qwen3.8-27B | In Vega? |
|---|---|
| Input word embeddings | **kept** |
| All **64 transformer layers** and the final norm | **kept**, none removed |
| The output layer that predicts the next word (the "LM head", 248,320 outputs, about 1.27B parameters, untied from the embeddings in this model) | **dropped** |
| The vision tower (this base model is natively multimodal) | **dropped** (a code comment says "vision tower unused") |
| The generation loop and KV cache | **not used** (`use_cache=False`) |

Added on top: a LoRA adapter (below) and the 5.3M-parameter head.

An accounting check: the base repository holds about 27.78B parameters. Vega's config counts 25.62B for the text model, and the untied LM head is 1.27B, which leaves about 0.89B. That is consistent with the vision tower, though it has not been checked tensor by tensor.

**So Vega removes no blocks.** It keeps all 64 layers, discards the word-prediction output, and retrains behaviour with LoRA. The speed difference from generating text comes from skipping the decode loop, not from a smaller network.

> **A common question: "What is the vision tower?"** It is a **separate stack of Transformer layers dedicated to images**, what Chapter 19 calls the vision encoder. In this Qwen model: 27 layers, width 1,152, 16 heads, images cut into 16 × 16 patches, each 2 × 2 group of patches merged, then projected up to the language model's width of 5,120 so that image vectors look like text-token vectors. The config also defines special token ids for "image", "video", and "vision start" and "vision end", marking where picture tokens sit in the sequence.
>
> There are two levels to keep apart. **The modules** (vision tower, projector, text model) are chosen by the architect and are separable: they are separate tensors, and Vega shows it by loading only the language model. **What each head or layer does inside a module** is learned, not assigned (Chapter 5: real heads range from clearly specialised to generic to nearly redundant). It is the fixed / learned / computed lens again. Joint training tunes the pieces to each other, but they stay separable. (Joint training may have shifted the text layers slightly, so a tower-less model is not guaranteed identical to one trained on text alone.)

### 3.5 How it is adapted: a LoRA "soup"

The adapter file holds **3.74 billion** trainable parameters, in 32-bit floats (14.9 GB), which is 13% of the base. It is rank 512, but the config shows it was built as a **soup**: **four separate LoRA runs of rank 128 each**, stitched together along the rank dimension (B matrices scaled by 1/4, the head averaged). So the team trained at least four adapter runs and merged them. The base model is not in the repository: it is a separate 55.6 GB download.

The training loss combines cross-entropy over the valid options, an optional Brier term and a replay KL term, configured as `training_mode: lora`, starting from Qwen's post-trained (chat-tuned) model with a randomly initialised head. That is **supervised cross-entropy with no reinforcement learning** (Chapter 27, Part 2). The KL term on replayed rows keeps the model from forgetting general decisions it already handled.

### 3.6 Laya and Vega side by side

| | Laya | Vega |
|---|---|---|
| Backbone | 395M ModernBERT, bidirectional | 27B Qwen, causal, hybrid attention |
| State position | last | first |
| Slot for each option | **before** the option text (`[MASK]`) | **after** the option text (its last token) |
| Summary token | `[CLS]` at the front | the final token of the prompt |
| Head | 2 Transformer layers + MLP scorer (~26M) | bilinear + MLP over two vectors (~5M) |
| Adaptation | full fine-tune | LoRA, rank 512 (a soup of four) |
| Context and options | 512 to 1,024 tokens; about 20 options | 32,768 tokens; up to 255 options |

### 3.7 The mask decides the design

Everything in the table's "position" rows follows from one fact. **Vega is causal**, so each of its choices is forced:

- The state comes **first**, so that later tokens can see it.
- The readout sits at the **end** of each option, because a token can only see what came before it.
- The summary token is the **last token of the prompt**, exactly the "`[CLS]` at the end" placement of Part 1.4.

**Laya is bidirectional**, so it can put its slots *before* the option text and the state *last*, and still have every slot read everything.

**Do the two recompute the state for every question?** By default, **both do**. Vega's default "exact path" encodes each question as its own row `[state + question + options]`, so, as its own docstring says, the shared part of the input is encoded again for every question. Vega also has an **opt-in** `share_context` mode, off by default, that computes the shared prefix once and continues each question from it: the full-attention layers reuse the prefix's stored keys and values (Chapter 12's KV cache, Chapter 22's prefix caching), and the linear-attention layers carry a recurrent state. It is not bit-for-bit identical (near ties can flip, and a `tau` setting sends close calls back to the exact path), and it only applies with at least two questions. Laya only tokenises the state once; it has no way to share encoder work, because its question tokens come *before* the state and attention runs both ways, so every state vector depends on the question.

How much sharing saves, for a 300-token state and 80 tokens per question-plus-options:

```text
5 questions:   default 5 × 380 = 1,900 token-passes    shared 300 + 5×80 = 700      → 2.7× fewer
10 questions:  default 3,800                            shared 1,100                 → 3.5× fewer
```

### 3.8 Speed: the theory and the claims

**Compute per token.** A forward pass costs about 2 × (non-embedding parameters) FLOPs per token:

| | per token | 300-token request |
|---|---|---|
| Vega (about 24.3B non-embedding) | 48.7 GFLOP | **14.6 TFLOP** |
| Laya (about 370M: encoder 343M + head 26M) | 0.74 GFLOP | **0.22 TFLOP** |
| Ratio | **about 66×** | |

On the same hardware, then, Vega should be about **66 times slower** in pure compute.

**What the model cards report.** All six Decision 2.0 cards report a median latency for a single-question request on a single GPU (Chapter 25's table). From Lux (7.94B) to Vega (29.4B), parameters grow 3.7× and latency 3.9×: roughly proportional, as theory predicts for large models. At the small end latency barely moves (4.9 ms for 0.6B, 6.0 ms for 0.75B): a fixed overhead floor.

**Why the measured gap to Laya looks small (about 2×).** Laya's published latency (32.8 to 39.5 ms) is on a **T4 GPU**, about 65 FP16 TFLOPS at peak. Three reasons it is not 66×:

1. **Different hardware.** Vega's cards say only "single GPU". Its optimised path in the code is written for AMD ROCm GPUs of the MI300 family (`gfx942`), and the same team's write-up for a smaller model reports training on an AMD MI325X. If the figure was measured on such a GPU (about 1,307 BF16 TFLOPS peak, roughly 20 times a T4), the card's number is plausible: a 300 to 500 token prompt would imply 240 to 400 effective TFLOPS, 19 to 31% of that peak. That fits, but it is an inference and not a stated fact. A 27B model in bf16 needs about 50 GB for weights and could not run on a 16 GB T4 at all.
2. **The overhead floor.** Laya's compute alone on a T4 would be about 4 to 19 ms for 100 to 512 tokens (at 30% of peak), yet it measures 33 to 40 ms.
3. **Vega's engineering:** HIP graphs, fused kernels and a leaner LoRA path, all aimed at removing overhead.

On the **same GPU**, Laya should be many times faster than Vega, tens of times in compute. Nobody has published that comparison, so it is an estimate and not a measurement.

**Memory and size.** Vega's base alone is 55.6 GB (plus the 14.9 GB adapter), so it wants an 80 GB-class GPU or a quantised build. Laya is **843 MB** and runs on a CPU.

### 3.9 What the extra compute buys

- **Length:** 32,768 tokens versus 512 to 1,024. A user report (a Laya issue) describes near-chance accuracy for Laya on long, noisy multilingual posts, where a decoder-based model handled the same task.
- **Options:** up to 255 versus about 20.
- **Reading strength:** a 27B backbone reads text much more deeply.

Laya wins on speed, size and cost. A reasonable split is Laya or a smaller family member for short, high-volume routing, and a larger decoder-based model where inputs are long or option lists are large.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "BERT is a classifier." | BERT is an **encoder** that outputs vectors. A classifier is a head added on top. Embedding models are encoders too, trained with a different objective. |
| "BERT can translate or write." | It cannot write. It reads. Translation needs an encoder–decoder or a decoder. |
| "`[CLS]` carries the sentence's meaning from the start." | Its *input* is a constant. The meaning is gathered layer by layer, and only because a loss is attached to it. A raw `[CLS]` is a poor summary. |
| "`[CLS]` is only for BERT." | A dedicated summary token helps in decoders too, but it must sit **last**, not first. |
| "Laya's `[MASK]` is BERT's `[MASK]`." | BERT's is a blank to guess. Laya's is a **slot** whose output vector is read to score an option. |
| "The extra layers are a different kind of Transformer." | They are ordinary Transformer layers. What makes them a decision head is what they are trained to do. |
| "Vega removed blocks from Qwen." | It kept all 64 layers and dropped only the word-prediction output and the vision tower. |

---

## Quick reference

```text
ENCODER (BERT family): no causal mask → every token sees every token → one vector per token
  input   = token + segment + position embeddings; [CLS] first, [SEP] between sentences
  [CLS]   = a neutral summary slot; useful only because a loss reads it
  pretrain= fill in blanks: 15% chosen → 80% [MASK] / 10% random / 10% unchanged
  finetune= pretrained weights + tiny head, train everything with small learning rates
  ModernBERT = RoPE, GeGLU, pre-norm, alternating local/global attention, 8,192 tokens, 2T tokens

LAYA (encoder design)
  per question: [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
  encoder → + type vector → 2 head layers → read each [MASK] slot → scorer → logit → softmax(÷T) → probabilities
  421M parameters = encoder 395M + head ~26M;  limits: ~20 options, 512-1,024 tokens, text only

VEGA (decoder design)
  Qwen 27B (64 layers, hybrid attention) + LoRA soup (3.74B params) + 5M-param head
  prompt: state FIRST ... options ... "Decision:"; readout at the last token of each option and of the prompt
  causal mask forces state first and readout last; supports up to 255 options and 32,768 tokens
```

---

## What's next

This chapter took the architecture apart: an encoder with a slot for each option (Laya) and a decoder with a readout at the end of each option (Vega). The next chapter covers what turns such a network into a useful decision model: where the labelled data comes from, how the model is trained to give good probabilities, what training costs, how to keep its confidence honest, and how to test it fairly.
