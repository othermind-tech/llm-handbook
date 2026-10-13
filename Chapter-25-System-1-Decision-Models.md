# Chapter 25 — System 1 Decision Models: Fast, Typed, Calibrated Decisions Without Generating Text

## Introduction

Every model in this handbook so far *writes*. Even when we asked a question with a one-word answer ("is this email spam?"), the model produced tokens one at a time, in the loop of Chapter 12 (inference, the KV cache and sampling), and we read the answer out of the text it wrote.

A lot of real software does not need writing. It needs a **pick**: which team gets this ticket, which tool should run next, is this command safe, does this customer want a refund. A pick has a fixed list of possible answers. It often has to arrive in milliseconds. And the program that receives it needs to know **how sure** the model is, as a number, not as a sentence.

This chapter is about a new family of models built for that job: **decision models**, also called **System 1 models**.

The name is borrowed from psychology. Daniel Kahneman describes two modes of human thinking: fast, automatic judgement (System 1) and slow, deliberate reasoning (System 2). Chapter 20's reasoning models are the System 2 side: they spend thousands of tokens thinking before they answer. A decision model is the System 1 side: it reads, decides in a single pass, and stops.

> **A decision model is usually not a new kind of Transformer.** It is a Transformer (often a BERT-style encoder, sometimes a decoder LLM) with a different *input layout*, a different *readout* and a different *training objective*. The novelty is in the job, and in the discipline of making the output typed and honest.

This chapter is the concept and the landscape: what a decision is, the three kinds of question, the vocabulary, where decision models fit in a system, and the models that exist today (Jev, Laya, Vega and others). It stays at the level of ideas and does not train or build a model.

The parts of this chapter:

1. **The problem:** why a text generator is an awkward tool for choosing.
2. **What a decision is:** state, question, allowed answers, result.
3. **The three kinds of question:** choice, score and noul (yes/no), and why there are three.
4. **Vocabulary:** the terms you will meet.
5. **Decision versus generation:** where decision models fit, and why not a plain classifier or embeddings.
6. **The landscape (October 2026):** Jev, Laya, Vega and the others.
7. **What is genuinely new here:** and what is an old idea in a new package.

> **How current this is.** The first widely noticed decision-model product, TypeSafe AI's Jev, was announced in September 2026, and most of the facts in this chapter are weeks old. Where a statement comes from a vendor, or from a model's own authors, the text says so. Treat names, sizes, prices and benchmark numbers as a snapshot. The ideas are the lasting part.

---

## Part 1 — The problem: why use a text generator to make a choice?

Take a support inbox. A message arrives:

> *"Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."*

Software needs three things from it: which team should handle it (billing, technical, other), how urgent it is, and whether the customer is threatening to leave.

### The language-model way

```text
prompt: "Classify this message... Answer with one word: billing, technical or other."
   ↓
the model writes:   "bill"   "ing"   "."          ← one forward pass per token (Chapter 12)
   ↓
the program parses the text and hopes it is one of the allowed words
```

This works, and for many applications it is fine. But look at what it costs when you do it thousands of times a minute inside a larger system:

1. **A loop for a one-word answer.** Generating text means one forward pass per token. The answer "billing" may be two or three tokens, plus whatever the model chooses to add around it.
2. **The answer is free text.** The program must parse it. The model may say "Billing team", "billing." or a sentence of explanation. You can force a format (structured output modes exist), but you are still asking a writer to behave like a switch.
3. **The confidence is words, not a number.** "I'm fairly sure it's billing" can't be put in an `if` statement. Token probabilities exist inside the model, but they are not calibrated (calibration, defined in Part 4), and they are probabilities over the next *token*, not over *your list of allowed answers*.
4. **The expensive tokens are the output tokens** (Chapter 24, routing and cost). For a decision, the output is the one part you did not need to pay for.

### What we actually want

```text
state (the message) + a question + a list of allowed answers
                          │
                          ▼
                 ONE forward pass, no writing
                          │
                          ▼
         a probability for every allowed answer
```

One pass. A fixed answer list. A number for every option. That is a decision model.

### "Couldn't a small language model do this?"

Partly, and the idea is worth seeing because it shows what the new models are doing. A decoder LLM can be used this way directly: put the options in the prompt, run **one** forward pass, and read the probabilities of the option labels at the next-token position instead of letting the model write. One independent test of decision models used exactly such a baseline (a 1.5B-parameter Qwen model with "parallel constrained decoding": score each allowed answer in parallel, never generate freely). The dedicated decision models go further, with a purpose-built readout and training, and they report probabilities that are meant to be *honest* (see Calibration in Part 4).

### Two names for the same idea

Kahneman's *System 1 / System 2* split was adopted by TypeSafe AI, whose model Jev is called a "System One model". Some vendors avoid the term and simply say **decision model**. This chapter uses both.

---

## Part 2 — What a decision is

A **decision** has four parts:

| Part | Meaning | Example |
|---|---|---|
| **State** | the text (or data) to judge | "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan." |
| **Question** | what we are asking about it | "Which department should handle this?" |
| **Allowed answers** | a closed list, written by *you*, at request time | billing, technical, other |
| **Result** | a probability for each allowed answer | billing 0.91, technical 0.05, other 0.04 (illustrative) |

That is all a decision model does. The state goes in, the question and the answer list go in, and probabilities come out. No text is generated.

### Writing a decision down: three fields

The open decision models in this chapter (Laya and Vega) share a request shape. A decision is written as a **question** with three fields:

| Field | What it is | Needed for |
|---|---|---|
| **`type`** | which kind of question: `choice`, `score` or `noul` (Part 3) | every question. It tells the system how to read and judge the answer. |
| **`instructions`** | the question itself, in plain English | every question |
| **`criteria`** | the **allowed answers**, each with a short description | `choice` (a label plus a description for each), `score` (the ordered levels). For `noul` it is optional: the defaults are "no, the statement does not hold" and "yes, the statement holds". |

Each question also has a name of your choosing (such as `department`). The field names above are the ones Laya and Vega use. TypeSafe's Jev offers the same three kinds of question, but its exact field names are not covered here.

Here is the ticket written down, with each line explained:

```python
state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."

questions = {
  "department": {                                   # the name of this question
      "type": "choice",                             # the kind: pick one
      "instructions": "Which department should handle this?",       # the question
      "criteria": {"billing":   "invoices, payments, refunds",      # answer: description
                   "technical": "bugs, outages, system errors",
                   "other":     "everything else"}},
  "urgency": {
      "type": "score",                              # the kind: an ordered scale
      "instructions": "How urgent is this?",
      "criteria": ["not urgent", "soon", "blocking"]},              # levels 0, 1, 2 in order
  "churn_risk": {
      "type": "noul",                               # the kind: yes/no
      "instructions": "Does the user threaten to cancel or leave?"},  # no criteria: yes/no defaults
}
```

One request carries **several** questions about the **same** state.

> **The criteria are text the model reads.** That is the key design fact. The allowed answers are not baked into the model's weights the way the classes of an ordinary classifier are. They are part of the *input*, so the same model can answer a different list tomorrow without retraining. It also means that each answer must explain itself: "billing: invoices, payments, refunds" gives the model something to read, while a bare "A" gives it nothing.

### A question learners ask: "Is the state like a prompt?"

Yes, with one refinement. In a language model, instructions and evidence share one prompt. Here the **state is the evidence**, kept apart from the question and the answer list, and then everything is laid into one input sequence. The "prefill" idea from Chapter 12 is a good mental model: a decision model is essentially **prefill only**. It reads the whole input in one pass (the prefill) and then stops. There is no decoding loop.

---

## Part 3 — The three kinds of question

Decision models offer three kinds of question. TypeSafe calls them *primitives*. An independent paper on Jev describes the three as choosing among candidate options (Choice), making binary judgments (Noul) and rating an input against predefined levels (Score). Laya uses the names `choice`, `score` and `noul`. ("Noul" is just the name these systems use for a yes/no check.)

| Kind | Plain meaning | Allowed answers | What comes back (illustrative) |
|---|---|---|---|
| **choice** | pick one from a list | billing / technical / other | 0.91 / 0.05 / 0.04 |
| **score** | place it on an ordered scale | not urgent (0) / soon (1) / blocking (2) | 0.10 / 0.25 / 0.65, and an expected level of 0.10×0 + 0.25×1 + 0.65×2 = **1.55** |
| **noul** | is this statement true? | yes / no | P(yes) = 0.88 |

### Why there are three kinds: errors are not all alike

Suppose the true urgency is level 2 ("blocking"). Compare five predictions. Two ways of measuring the error are shown: **cross-entropy** (Chapter 9), and the **ranked probability score** (RPS, lower is better), which compares *cumulative* probabilities and so cares about *how far* the probability is from the right level.

```text
                                  cross-entropy    ranked probability score
A  [0.10, 0.25, 0.65]  right           0.43              0.13
B  [0.10, 0.65, 0.25]  off by one      1.39              0.57
C  [0.65, 0.25, 0.10]  off by two      2.30              1.23
D  [0.30, 0.60, 0.10]  near miss       2.30              0.90
E  [0.60, 0.30, 0.10]  far miss        2.30              1.17
```

Look at D and E. Both give the true level 0.10, so cross-entropy scores them identically at 2.30. But E puts its weight two levels away and D puts it one level away, and RPS tells them apart. Cross-entropy treats every wrong class as equally wrong, which is right for `choice`: "billing" is not closer to "technical" than to "other". It is wrong for `score`, where order matters. This is why `score` is a separate kind with its own measure of error.

### A question learners ask: "Is a score one number?"

It depends on what you mean, and the answer is worth getting right.

A score question does **not** return a single probability somewhere between 0 and 2. Keep two scales apart: **probabilities** always run from 0 to 1, and **levels** (0, 1, 2) are the *answers*. A score returns:

- a **probability for each level**, which together sum to 1 (for example 0.10, 0.25, 0.65), and
- one derived **expected level** (1.55 in the example), which does sit in the 0-to-2 range.

The single number is calculated *from* the distribution. Laya's code computes it literally as the sum of level × probability.

A different design is possible: the model outputs one number and cut-offs decide the level (this is called ordinal regression). It is simpler, but it throws away the model's **uncertainty**. Consider two situations with the *same* expected level:

```text
sure it's level 1:        [0.0, 1.0, 0.0]   expected level = 1.0
torn between 0 and 2:     [0.5, 0.0, 0.5]   expected level = 1.0
```

A single number cannot tell these apart, but they should lead to different actions: act on the first, escalate the second. The expected level can even land on a level the model considers **impossible**: in the second case it is 1.0, yet level 1 has probability zero. So when you need a final level:

| Method | How | Watch out for |
|---|---|---|
| **Most likely level** | take the level with the highest probability | the safest default |
| **Round the expected level** | 1.55 → 2 | can pick a level the model gave zero probability |
| **Cut-offs on the expected level** | the "one number" design | the same trap, and the thresholds are yours to choose |

For anything that triggers an action, look at the confidence or the distribution first, and treat the single number as a summary.

> **Choice and score have outputs of the same shape** (one probability per option). What differs is the *meaning* of the options: unordered for `choice`, ordered for `score`. The `noul` kind is a two-option question (false, true) reported as the probability of "true".

### What a whole answer looks like

For the ticket above, a request returns something like this (illustrative values):

```text
department   billing 0.91   technical 0.05   other 0.04          → pick: billing
urgency      level 0: 0.10  level 1: 0.25   level 2: 0.65        → expected level 1.55
churn_risk   P(yes) = 0.88
```

Three answers, one state, one pass.

---

## Part 4 — Vocabulary

A short glossary, so that the rest of the chapter reads easily. Most of these words are explained properly where they first matter.

| Term | Meaning |
|---|---|
| **State** | the text or data being judged (the evidence) |
| **Question / instructions** | what is being asked, in plain words |
| **Criteria / options / allowed answers** | the closed list of answers, each with a short description |
| **Primitive** (kind of question) | `choice`, `score` or `noul` |
| **Typed output** | an output whose form is fixed in advance (a label from a list, a level, a yes/no probability), so a program can use it without parsing text |
| **Non-autoregressive** | not generating token by token; the answer comes from one forward pass |
| **System 1 / System 2** | fast automatic judgement versus slow deliberate reasoning; decision models versus reasoning models |
| **Logit ("dial")** | the raw score the model produces for an option, before it becomes a percentage |
| **Softmax** | the function that turns the dials into percentages that sum to 1 (Chapter 8) |
| **Temperature** | a number that divides the dials before the softmax; a higher value gives softer, less sure percentages |
| **Confidence** | how sure the model says it is. Laya uses the name for two different quantities. |
| **Calibration** | the property that "80% sure" is right about 80% of the time |
| **ECE** | *expected calibration error*: the average gap between stated confidence and actual accuracy |
| **Teacher** | a stronger model (or people) whose answers are used as the training answer key |
| **Gold / soft label** | the training target. A *soft* label is a probability for every option instead of a single right answer. |
| **Slot / endpoint** | the position in the input whose output vector is read to score one option |
| **Option budget** | the limited number of tokens available to describe all the options |
| **Threshold / coverage / abstention** | answering automatically only above a confidence level; coverage is the fraction of questions answered; abstaining means handing the rest to someone else |
| **Gating / escalation** | using the probability to decide whether to act, ask a bigger model, or ask a person |
| **Zero-shot vs fine-tuned** | used as shipped, versus trained further on your own decisions |
| **Held-out** | data kept out of training so it can test the model fairly |
| **Backbone / head** | the main body of the model (for example a BERT-style encoder) versus the small layers added on top to produce the decision |

---

## Part 5 — Decision versus generation: where decision models fit

### The dividing line

If the right answer **can be listed in advance**, it is a decision. If it cannot, it is generation.

| Decision (closed answers) | Generation (open answers) |
|---|---|
| Which tool should handle this? | Write the reply to the customer |
| Is this command safe to run? | Fill in the arguments for the tool |
| How severe is this finding? | Plan the next five steps |
| Does the user want a refund? | Summarise the thread |

### The pattern: a generator writes, a decision model routes and gates

```text
                         request
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
        DECISION MODEL (System 1)   LLM (System 2 / generation)
        routes, judges, gates       plans, writes, fills in details
        milliseconds, cheap         seconds, expensive
                 │                     │
                 └──────────┬──────────┘
                            ▼
                      the application
```

A 2026 paper on penetration-testing agents ("Calibrated Decision Models for Autonomous Penetration-Testing Harnesses") formalises four decision points where a System 1 model replaces free-form LLM judgement in an agent: confirming a finding, grading severity, choosing which agent to run, and confirmation loops. Its authors are candid that their evidence is an exploratory case study that motivates the design and is not a controlled experiment.

### What the probability is for

A probability is only useful if the system *does something* with it. Three common policies:

- **Act / ask / escalate.** Above 0.9, act automatically. Between 0.6 and 0.9, ask a bigger model. Below 0.6, ask a person.
- **Cost-based.** Suppose acting correctly is worth +1, acting wrongly costs 3, and escalating costs 0.5. Acting beats escalating when `p − 3(1 − p) > −0.5`, that is `4p − 3 > −0.5`, so **p > 0.625**. (These numbers are the ones Laya's author describes for its act/escalate head, used here as an illustration of the reasoning.)
- **Coverage target.** Answer only the most confident half of the questions, and send the rest elsewhere.

All three assume the probabilities are **honest**. That is the subject of calibration (Part 4), and the policies above depend on it.

### "Why not just a classifier?"

An ordinary classifier (for example a BERT encoder with a classification head, Chapter 15) has its labels fixed inside it: one output per class. Add a class and you retrain. A decision model reads the allowed answers as *part of the input*, so the list can change per request. That is the difference, and the whole reason for the extra machinery.

### "Why not embeddings and a similarity comparison?"

This is the natural alternative, and it deserves a careful answer. Embedding models (Chapter 15, Part 5; Chapter 24, RAG) turn a text into a vector, and you pick the label whose vector is closest. It is cheap, quick to set up, and a good baseline. There are three reasons it is not the whole answer.

**1. The text and the options never meet.** An embedding comparison encodes each text separately and compares the results afterwards (a *bi-encoder*). Similarity measures closeness of *topic*, not "which option is true for this text". Take the state "Please do **not** cancel my subscription" with the options "cancel subscription" and "keep subscription". Both options are on-topic. Telling them apart needs the state and the option read **together**, which is what a decision model does (a *cross-encoder*: everything is in one sequence, so attention connects them). This is a typical weakness of similarity matching, not a measured claim about any particular model.

**2. Not every question is "which label is closest".** "How urgent, 0 to 2?" and "is this statement true?" are not similarity-shaped. You can train a small classifier on top of the embeddings, but then you are training a head anyway, and the options still cannot influence how the state is read.

**3. Cosine similarity is not a probability.** A similarity of 0.71 does not mean "71% likely". You would still need to calibrate it.

In practice the two combine. Search systems retrieve candidates with embeddings and then re-rank them with a model that reads the query and each candidate together. Laya does the same: for questions with more than about 20 options, it shortlists the labels by embedding similarity (using vectors from its own encoder) and then decides with a single joint pass.

### "The tool list changes between systems. How does it adapt?"

Because the options are text, the same model can be given a different list. It adapts the way an LLM adapts to tool descriptions: by *reading them*. The limits are real:

- It sees only what is in the state and the option text. If it needs context, that text has to supply it.
- The shipped base models are weaker zero-shot than their authors' fine-tuned versions. Writing good descriptions is the free first step, and fine-tuning on your own decisions is the step that changes behaviour.

### The framing checklist

Before touching a model:

1. **List the decisions the software makes.** Each needs a question and a closed list of answers.
2. **Decision or generation?** If the right answer cannot be listed in advance, it is generation.
3. **Pick the kind:** choice, score or noul.
4. **Put the evidence in the state.** The model cannot look anything up.
5. **Write each answer so it explains itself.**
6. **Decide what the probability will do:** act, ask or escalate.
7. **Count the options.** Laya's answer text shares a budget of 192 to 256 tokens (roughly 20 options), and Jev's documented limit is 255 options. Beyond that, shortlist or route in two steps.

### Limits to keep in mind from the start

- **A closed list cannot be surprised.** If the right answer is not on it, the model will still pick one.
- **No invented text, but still sometimes wrong.** A typed output cannot hallucinate a sentence, but it can be confidently wrong. That is why calibration and evaluation are part of the design.
- **Everything it needs must be in the state.**

---

## Part 6 — The landscape (October 2026)

This part describes the models that exist, using each maker's own materials, and flags which claims are the maker's. The field is only weeks old, so expect it to move.

### 6.1 Jev (TypeSafe AI): the closed one that started it

**Who and when.** TypeSafe AI announced Jev on **15 September 2026**. The company (founded in 2024) says it worked on the model in stealth for about two years. Its chief executive, Diogo Almeida, previously worked at OpenAI on RLHF and ChatGPT, according to Wikipedia and TechCrunch. The name is a nod to the economist William Stanley Jevons and the "Jevons paradox": when something gets cheaper, people use more of it.

**What it does.** State plus typed questions in, probabilities out: the three kinds of question from Part 3. TypeSafe describes Jev as turning unstructured state into typed, probabilistic decisions, and as giving up string generation entirely.

**What is public and what is not.**

| Published | Not published |
|---|---|
| the API: input and output formats, how confidence numbers are computed | the weights |
| pricing and limits | the architecture and model size |
| evaluation write-ups by TypeSafe | a technical paper |
| | the training data (the launch post has a FAQ heading asking where the training data comes from, with no answer under it) |

TypeSafe describes the model as transformer-based. Outside observers have speculated that it might be built on an open-weight language model or a BERT-style encoder, but that is speculation.

**Claims (all TypeSafe's own).**

- End-to-end response between 70 ms and 500 ms.
- On its own workflow evaluations, 193.6× faster and 444.6× cheaper than frontier models. The launch post says those workflows were made by people on TypeSafe's own team, that the reference answers are the **average of two frontier models** (not human labels), and that, it says, the workflows are outside its training distribution.
- Price: **$0.042 per million input tokens, with output tokens free**. A decision has no generated output, so only the input is billed.
- Up to **255 options** for a choice question; text only (it says images are not supported yet). One summary of the API documentation reports an input limit of 64k tokens, which the launch post itself does not state.
- Training method named **"Reinforcement Learning for Calibrated Decisions" (RLCD)**. Wikipedia and TechCrunch both report that Jev was trained exclusively on synthetic data, attributing that to Almeida. TypeSafe's own page does not confirm it.
- Early users reported in the press: a safety classifier at Vercel running 5 to 18 times faster than an OpenAI model with greater accuracy, and an email classifier at Bryo AI 10 to 20 times cheaper than Gemini with better confidence scores but slightly lower accuracy. These are company statements.

**An ecosystem appeared quickly.** An independent paper counted **2,170 public GitHub projects** using Jev as of 22 September 2026, a week after launch. That measures adoption, not accuracy.

### 6.2 Laya (Convai Innovations): the open one we will take apart

**Who and what.** Laya is an open-weight decision model from Convai Innovations, built by Nandakishor M, released under the Apache 2.0 licence with its code, benchmark scripts and a fine-tuning notebook. It was presented as a free, local alternative to Jev.

**The three checkpoints.**

| Checkpoint | Backbone | Parameters | Context | Purpose |
|---|---|---|---|---|
| `laya` | ModernBERT-large | 421M | 512 tokens | English text, the base model |
| `laya-multilingual` | mmBERT-base | 322M | 1,024 (can run up to 8,192) | 100+ languages |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1,024 | fine-tuned on a four-workflow decision benchmark |

**Claims (the authors').** About 33 ms for one question on a T4 GPU, calibration error cut from 0.466 to 0.081 after temperature fitting, and 0.766 accuracy for the fine-tuned checkpoint on a typed-decisions test. All three are the authors' own figures (treat them as claims, not measurements).

**Lineage.** The author had earlier published two papers on related ideas: a March 2025 model that predicts the probability a sales conversation will convert, and a September 2025 paper on estimating, *before* a language model answers, whether it is likely to hallucinate, and routing the query accordingly. In his own write-up he says the underlying concepts predate Jev. The papers do describe related ideas (probability outputs, reinforcement learning, routing before generation). They do not describe the typed, request-time interface, and how Jev itself was developed is not public.

**Status in one sentence.** Laya's base checkpoints are weak zero-shot on some tests, and its authors say the value comes from fine-tuning on your own decisions. Its design is fully open, which is why it is the one decision model whose internals anyone can inspect.

### 6.3 Decision 2.0 and Vega (vLLM Semantic Router team): the decoder way

**Who and what.** Decision 2.0 is a family of six open decision models from the team behind the vLLM Semantic Router, released under Apache 2.0. They are built on **Qwen** language models. **Vega** is the largest. It is *not* a release from Qwen's own team. It is built on a Qwen base model.

**How it differs.** Where Laya is a 421M-parameter bidirectional encoder, the Decision 2.0 models are **decoder LLMs** adapted with LoRA (Chapter 13) and a small readout head, and they accept the same request shape as Laya (`state`, and questions with `type`, `instructions`, `criteria`).

| Model | Parameters | Context | Median latency per single-question request (card) |
|---|---|---|---|
| Kai | 0.60B | 8,192 | 4.9 ms |
| Eos | 0.75B | 16,384 | 6.0 ms |
| Sol | 1.88B | 16,384 | 7.2 ms |
| Nox | 4.21B | 16,384 | 12.9 ms |
| Lux | 7.94B | 16,384 | 18.4 ms |
| **Vega** | **29.37B** | **32,768** | **71.4 ms** |

The latencies are the cards' own ("on a single GPU", hardware not named). The pattern is the one theory predicts: small models sit on a fixed overhead floor of a few milliseconds, and big ones grow roughly in proportion to size. Vega supports up to 255 options.

### 6.4 Others reported so far

This list comes from news coverage and an independent comparison in the first weeks of October 2026. Treat every entry as "reported" and check it before relying on it.

- **Microsoft-Decision-1**, reported to be built on Alibaba's Qwen3.5-9B, text-only, priced at the same $0.042 per million input tokens as Jev. Microsoft's performance claims are its own.
- **OpenAI** is reported to have opened a "Decisions" API in public beta.
- **Cloudflare**'s hosted Clef 27B and Clef-flash 9B.
- **Perplexity Decider 1.1 (27B)**, an open-weight model, reported to top one independent comparison.
- Further open or hosted entries: Liquid AI's `d1`, meraGPT's Decider 1, Featherless's Simple Jev, OpenDecider (several sizes, including a 400M encoder), and the Open-Jev datasets and models.
- A news headline also reports that AWS, Upstage and Ollama have agreed on a common decision-model API, a sign that the field is converging on one request shape.

### 6.5 Side by side

| | Jev | Laya | Vega (Decision 2.0) |
|---|---|---|---|
| Open weights | no (hosted API) | yes, Apache-2.0 | yes, Apache-2.0 |
| Backbone | not disclosed | BERT-style encoder (ModernBERT-large; mmBERT-base) | Qwen decoder with LoRA |
| Size | not disclosed | 421M (322M multilingual) | 29.4B listed |
| Context | not stated by TypeSafe (64k reported for the API) | 512 to 1,024 by default | 32,768 |
| Options per question | up to 255 | about 20 (token budget) | up to 255 |
| Speed claim | 70 to 500 ms end to end | about 33 to 40 ms on a T4 | 71.4 ms, hardware not named |
| Who measured | TypeSafe | the authors, plus third parties | the team, plus its stated reproduction |

### 6.6 How to read the claims

Whenever a number appears, ask four questions:

1. **Who measured it?** The vendor, the model's authors, or an outsider?
2. **On what data?** Was the test data part of the training mix?
3. **Which version?** A base checkpoint used as shipped, or one fine-tuned on that very test's training split?
4. **Compared with what?** A baseline that ignores the input can score surprisingly well.

As one data point: the independent tests so far found larger decoder-based models, and Jev, ahead of Laya's *base* checkpoints when used zero-shot, which is consistent with Laya's authors saying that fine-tuning does most of the work.

---

## Part 7 — What is genuinely new here

It helps to separate what is old from what is new, because the topic is surrounded by marketing.

**Old ideas, repackaged:**

- **Encoders with a classification head** (BERT, 2018). See Chapter 15.
- **Zero-shot classification with labels given as text.** Models that take the candidate labels at request time have existed for years (for example NLI-based zero-shot classifiers, and "label-agnostic" pretraining). Vega is even tagged as a zero-shot-classification model on Hugging Face.
- **Cross-encoders** for re-ranking, where the query and candidate are read together.
- **Cross-entropy and temperature scaling**, the standard tools for training and calibrating a classifier.
- **Routing** with a small classifier (Chapter 24, Part 7), which decision models generalise.

**What looks new in 2026:**

1. **A shared interface.** Typed question kinds, answer lists given per request, and several questions about one state answered in one pass.
2. **Calibrated probabilities as the product.** The number is meant to be trusted and acted on (act, ask, escalate), not just ranked.
3. **Training on many decision datasets at once**, plus teacher-labelled or synthetic data, so that one model reads whatever answer list it is given.
4. **Packaging and price:** milliseconds, input-only billing, and an emerging common API.

**What is not known:** how Jev works inside. Everything in this chapter about internals comes from the *open* models.

> **Where this sits among the chapters.** System 2 (Chapter 20) buys accuracy by spending tokens. System 1 buys speed and a trustworthy number by spending none. In an agent (Chapter 24) the two work together: the fast model routes and gates, and the slow model plans and writes.

---

## Common misconceptions, cleared up early

| Misconception | Reality |
|---|---|
| "A decision model is a small language model." | Not necessarily. Laya is a BERT-style encoder. Vega is a large decoder with a readout head. None of them generate text. |
| "It cannot be wrong, because it cannot hallucinate." | It cannot invent a sentence, but it can pick the wrong option, with confidence. |
| "The list of answers is learned inside the model." | The list is request-time *text* that the model reads. That is why it can change per request, and why each answer must explain itself. |
| "A score question returns one number." | It returns a probability for each level, and the single number (the expected level) is calculated from them. |
| "Calibrated means accurate." | Calibrated means *honest about its uncertainty*. A calibrated model can still be wrong often. (See Calibration in Part 4.) |
| "Embeddings and similarity do the same job." | Similarity compares topics and reads the text and the options separately. A decision model reads them together and gives a probability. |
| "Jev's design is known." | TypeSafe has not published the architecture, weights, training data or a technical paper. |
| "Vega comes from the Qwen team." | It comes from the vLLM Semantic Router team, built on a Qwen base model. |
| "A decision model replaces the LLM in an agent." | It pairs with it: the fast model routes and gates, the LLM plans and writes. |
| "System 1 models are a new architecture." | Mostly a new job, readout and training objective on familiar Transformers. |

---

## Quick reference

```text
Decision = STATE (evidence) + QUESTION + ALLOWED ANSWERS (request-time text) → PROBABILITY per answer
One forward pass (prefill only). No generation. Several questions per state, answered together.

Request fields (Laya / Vega):  type · instructions · criteria      (criteria = the answers, each described)

Three kinds:   choice  pick one            unordered options        → probability per option
               score   ordered levels      ordered options          → probability per level + expected level
               noul    yes/no statement    two options (false/true) → P(true)
Why three:     errors differ: cross-entropy treats all wrong answers alike; ordered levels need a
               distance-aware score (ranked probability score)

Fits where:    decision (closed list) → decision model;  open answer → LLM.   Pair them: route/gate vs plan/write
Probability →  act / ask / escalate (thresholds, cost rule p > 0.625 in the example), needs calibration

Landscape (Oct 2026):  Jev (TypeSafe, closed, undisclosed internals) · Laya (open, ModernBERT, 421M)
               · Decision 2.0 / Vega (open, Qwen decoder + LoRA, 0.6B to 29B) · others reported
Read claims:   who measured · on what data · which version · compared with what
```

---

## What's next

You now have the concept, the vocabulary and the map. Two threads continue from here. To see how a decision model is built, the open ones are the place to look: their model cards, source code and training documentation are listed in [References](References.md), section 14, and the encoder background they rely on is in Chapter 15. For the most concrete view of any Transformer, the **Appendix** follows a complete forward pass and a complete backward pass through a tiny GPT with small real numbers, so you can see every matrix multiplication, softmax and gradient that the chapters described in words.
