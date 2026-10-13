// Companion to Chapters 26 and 27: the small experiments behind the worked numbers in those chapters.
// MIT License. See LICENSE-MIT-code.txt in this repository.
//
// Run one:  node Chapters-26-27-experiments.js calibration
// Run all:  node Chapters-26-27-experiments.js all
// Experiments: bert  dials  rps  rl  calibration  gate  margins  costs
//
// Everything is plain JavaScript with a seeded random generator, so the output is the same on every run.
// These are TOY models. They illustrate mechanisms. They are not measurements of Laya or Vega.

const R = { seed: 1 };
const rnd = () => { R.seed = (R.seed * 1664525 + 1013904223) % 4294967296; return R.seed / 4294967296; };
const randn = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
const softmax = (z, T = 1) => { const m = Math.max(...z); const e = z.map(x => Math.exp((x - m) / T)); const s = e.reduce((a, b) => a + b); return e.map(x => x / s); };
const pct = v => (v * 100).toFixed(1) + "%";
const f = (x, d = 3) => x.toFixed(d);

// ---------------------------------------------------------------------------------------------
function bert() {
  console.log("== [CLS] attention step (Chapter 26, Part 1.4) ==");
  const K = [[0.2, 0.1], [0.9, 0.3], [0.5, 0.4], [0.8, 0.8]], V = [[0.1, 0.0], [1.0, 0.2], [0.4, 0.6], [0.2, 1.0]], q = [1.0, 0.5];
  const sc = K.map(k => (q[0] * k[0] + q[1] * k[1]) / Math.sqrt(2));
  const e = sc.map(Math.exp), s = e.reduce((a, b) => a + b), w = e.map(x => x / s);
  const out = [0, 1].map(d => w.reduce((a, wi, i) => a + wi * V[i][d], 0));
  console.log("scores", sc.map(x => f(x)).join(" "), "| weights", w.map(x => f(x)).join(" "), "| new CLS", out.map(x => f(x)).join(" "));

  console.log("\n== Classification head (Chapter 26, Part 1.6) ==");
  const C = [0.6, -0.2, 0.9, 0.3];
  const W = [[0.5, 0.1, 0.4, -0.2], [-0.3, 0.6, 0.0, 0.5], [0.2, -0.4, -0.5, 0.1]];
  const z = W.map(r => r.reduce((a, wi, i) => a + wi * C[i], 0)), p = softmax(z);
  console.log("logits", z.map(x => f(x)).join(" "), "| probs", p.map(x => f(x)).join(" "));
  console.log("loss if billing:", f(-Math.log(p[0])), "| if technical:", f(-Math.log(p[1])));
  console.log("gradient on logits (billing true):", p.map((x, i) => f(x - (i === 0 ? 1 : 0))).join(" "));
  console.log("gradient on W[billing]:", C.map(c => f((p[0] - 1) * c)).join(" "));
  console.log("new parameters for K=3, H=768:", 3 * 768, "=", f(3 * 768 / 110e6 * 100, 4) + "% of 110M");

  console.log("\n== ModernBERT-large size check (Chapter 26, Part 1.7) ==");
  const attn = 4 * 1024 * 1024, ffn = 1024 * 5248 + 2624 * 1024, per = attn + ffn;
  console.log("per layer", f(per / 1e6, 2) + "M x 28 =", f(per * 28 / 1e6, 0) + "M, + embeddings", f(50368 * 1024 / 1e6, 1) + "M = ", f((per * 28 + 50368 * 1024) / 1e6, 0) + "M");
}

// ---------------------------------------------------------------------------------------------
function dials() {
  console.log("== The dials: cross-entropy against a soft teacher key (Chapter 27, Part 2.3) ==");
  const t = [0.7, 0.2, 0.1];
  const ce = (t, p) => -t.reduce((a, ti, i) => a + ti * Math.log(p[i]), 0);
  let z = [0.5, 0.4, -0.2];
  console.log("floor of the loss (teacher's own uncertainty):", f(-t.reduce((a, ti) => a + ti * Math.log(ti), 0)));
  const lr = 2.0;
  for (let step = 0; step <= 8; step++) {
    const p = softmax(z), nudge = t.map((ti, i) => ti - p[i]);
    console.log(`step ${step}: dials [${z.map(x => f(x, 2)).join(", ")}] probs [${p.map(x => f(x, 2)).join(", ")}] loss ${f(ce(t, p))} nudge [${nudge.map(x => (x >= 0 ? "+" : "") + f(x, 2)).join(", ")}]`);
    z = z.map((zi, i) => zi + lr * nudge[i]);
  }
  z = [0.5, 0.4, -0.2];
  console.log("same start, hard label [1,0,0]:");
  for (let step = 0; step <= 8; step += 4) { const p = softmax(z); console.log(`  step ${step}: probs [${p.map(x => f(x, 2)).join(", ")}]`); for (let k = 0; k < 4; k++) { const pp = softmax(z); z = z.map((zi, i) => zi + lr * ([1, 0, 0][i] - pp[i])); } }

  console.log("\n== Two teachers disagree (Chapter 27, Part 1.3) ==");
  const gold = [0.5, 0, 0.5], hard = [1, 0, 0];
  const cands = { "overconfident [0.98,0.01,0.01]": [0.98, 0.01, 0.01], "matches teachers [0.495,0.01,0.495]": [0.495, 0.01, 0.495], "wrong side [0.02,0.01,0.97]": [0.02, 0.01, 0.97] };
  const cee = (tg, p) => -tg.reduce((a, g, i) => a + (g > 0 ? g * Math.log(p[i]) : 0), 0);
  for (const [n, p] of Object.entries(cands)) console.log(n.padEnd(40), "loss vs soft:", f(cee(gold, p), 2), "| loss vs hard:", f(cee(hard, p), 2));
}

// ---------------------------------------------------------------------------------------------
function rps() {
  console.log("== Cross-entropy versus ranked probability score for an ordered level (Chapter 25, Part 3) ==");
  const cum = p => p.reduce((a, x, i) => { a.push((a[i - 1] || 0) + x); return a; }, []);
  const rpsf = (p, t) => { const cp = cum(p), ct = cum(p.map((_, i) => (i === t ? 1 : 0))); let s = 0; for (let k = 0; k < p.length - 1; k++) s += (cp[k] - ct[k]) ** 2; return s; };
  const cases = [["A right", [0.10, 0.25, 0.65]], ["B off by one", [0.10, 0.65, 0.25]], ["C off by two", [0.65, 0.25, 0.10]], ["D near miss", [0.30, 0.60, 0.10]], ["E far miss", [0.60, 0.30, 0.10]]];
  for (const [n, p] of cases) console.log(n.padEnd(14), JSON.stringify(p), "cross-entropy", f(-Math.log(p[2]), 2), "| ranked probability score", f(rpsf(p, 2), 2));
  console.log("expected level for A:", f(0 * 0.10 + 1 * 0.25 + 2 * 0.65, 2));
}

// ---------------------------------------------------------------------------------------------
function rl() {
  console.log("== One RL step, using Laya's formulas on a toy question (Chapter 27, Part 2.6) ==");
  R.seed = 12345;
  const reward = (q, t, wsph = 0.75) => {
    const logs = t.reduce((a, ti, i) => a + ti * Math.max(Math.log(Math.max(q[i], 1e-12)), -9.21), 0);
    const norm = Math.sqrt(q.reduce((a, x) => a + x * x, 0));
    return logs + wsph * (t.reduce((a, ti, i) => a + ti * q[i], 0) / norm);
  };
  const t = [0.7, 0.2, 0.1], z = [0.5, 0.4, -0.2], p = softmax(z);
  console.log("target", t.join(", "), "| model probs", p.map(x => f(x)).join(", "));
  const ceDir = t.map((ti, i) => ti - p[i]);
  console.log("soft CE update direction (t - p):", ceDir.map(x => f(x)).join(", "));
  const exact = z.map((_, i) => { const h = 1e-5, zp = z.slice(); zp[i] += h; const zm = z.slice(); zm[i] -= h; return (reward(softmax(zp), t) - reward(softmax(zm), t)) / (2 * h); });
  console.log("exact d(reward)/d(logits):       ", exact.map(x => f(x)).join(", "));
  function estimate(G, sigma) {
    const samples = [];
    for (let g = 0; g < G; g++) {
      let eps = z.map(() => randn() * sigma); const mean = eps.reduce((a, b) => a + b) / eps.length; eps = eps.map(e => e - mean);
      const q = softmax(z.map((zi, i) => zi + eps[i])); samples.push({ eps, q, r: reward(q, t) });
    }
    const rm = samples.reduce((a, s) => a + s.r, 0) / G; let adv = samples.map(s => s.r - rm);
    const sd = Math.sqrt(adv.reduce((a, x) => a + x * x, 0) / (G - 1)); adv = adv.map(a => a / (sd + 1e-6));
    return { samples, adv, grad: z.map((_, i) => samples.reduce((a, s, g) => a + adv[g] * s.eps[i] / (sigma * sigma), 0) / G) };
  }
  const one = estimate(4, 0.4);
  one.samples.forEach((s, g) => console.log(`  try ${g + 1}: noise [${s.eps.map(x => f(x, 2)).join(", ")}] answer [${s.q.map(x => f(x, 2)).join(", ")}] reward ${f(s.r)} advantage ${f(one.adv[g], 2)}`));
  console.log("  one RL step (adv*eps/sigma^2, averaged):", one.grad.map(x => f(x, 2)).join(", "));
  const cos = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0) / (Math.hypot(...a) * Math.hypot(...b));
  for (const [G, sg] of [[4, 0.4], [4, 0.1], [64, 0.4]]) {
    const cs = []; for (let n = 0; n < 2000; n++) cs.push(cos(estimate(G, sg).grad, exact));
    const mc = cs.reduce((a, b) => a + b) / cs.length, sdv = Math.sqrt(cs.reduce((a, c) => a + (c - mc) ** 2, 0) / cs.length);
    console.log(`G=${G}, sigma=${sg}: cosine with the exact reward gradient: mean ${f(mc, 2)} (sd ${f(sdv, 2)}) | one-step size ~${f(Math.hypot(...estimate(G, sg).grad), 1)} vs CE ${f(Math.hypot(...ceDir), 2)}`);
  }
  console.log("cosine(CE direction, exact reward gradient) =", f(cos(ceDir, exact)));
}

// ---------------------------------------------------------------------------------------------
function calibration() {
  console.log("== An over-confident toy model, fixed with one temperature (Chapter 27, Part 4) ==");
  R.seed = 777;
  const TRUE_T = 2.5, K = 3;
  const makeSet = n => { const out = []; for (let i = 0; i < n; i++) { const z = Array.from({ length: K }, () => randn() * 2.0); const pt = softmax(z, TRUE_T); let u = rnd(), y = 0; for (; y < K - 1; y++) { u -= pt[y]; if (u <= 0) break; } out.push({ z, y }); } return out; };
  const calib = makeSet(1000), test = makeSet(4000);
  const stats = (set, T) => set.map(({ z, y }) => { const p = softmax(z, T); const c = Math.max(...p); return { conf: c, ok: p.indexOf(c) === y ? 1 : 0, nll: -Math.log(Math.max(p[y], 1e-12)) }; });
  const ece = (s, bins = 10) => { let e = 0; for (let b = 0; b < bins; b++) { const lo = b / bins, hi = (b + 1) / bins; const sel = s.filter(x => (b === 0 ? x.conf >= lo : x.conf > lo) && x.conf <= hi); if (sel.length) e += (sel.length / s.length) * Math.abs(sel.reduce((a, x) => a + x.conf, 0) / sel.length - sel.reduce((a, x) => a + x.ok, 0) / sel.length); } return e; };
  const mean = (s, g) => s.reduce((a, x) => a + g(x), 0) / s.length;
  let bestT = 1, bestN = Infinity; for (let T = 0.3; T <= 6; T += 0.01) { const n = mean(stats(calib, T), x => x.nll); if (n < bestN) { bestN = n; bestT = T; } }
  const before = stats(test, 1), after = stats(test, bestT);
  console.log("fitted temperature T =", f(bestT, 2), "(true value used to make the data:", TRUE_T + ")");
  console.log("BEFORE  accuracy", f(mean(before, x => x.ok)), "| average confidence", f(mean(before, x => x.conf)), "| ECE", f(ece(before)));
  console.log("AFTER   accuracy", f(mean(after, x => x.ok)), "| average confidence", f(mean(after, x => x.conf)), "| ECE", f(ece(after)));
  const edges = [0.33, 0.5, 0.7, 0.9, 1.0];
  const table = s => edges.slice(0, -1).map((lo, i) => { const hi = edges[i + 1]; const sel = s.filter(x => x.conf > lo && x.conf <= hi); return sel.length ? `${f(lo, 2)}-${f(hi, 2)}  n=${String(sel.length).padStart(4)}  says ${(mean(sel, x => x.conf) * 100).toFixed(0)}%  right ${(mean(sel, x => x.ok) * 100).toFixed(0)}%` : `${f(lo, 2)}-${f(hi, 2)}  n=   0`; });
  console.log("reliability BEFORE:"); table(before).forEach(l => console.log("  " + l));
  console.log("reliability AFTER:"); table(after).forEach(l => console.log("  " + l));
  console.log("answer only when confident (AFTER calibration):");
  for (const th of [0.0, 0.5, 0.7, 0.9]) { const sel = after.filter(x => x.conf >= th); console.log(`  confidence >= ${f(th, 1)}  answers ${(sel.length / after.length * 100).toFixed(0)}% of questions, ${(mean(sel, x => x.ok) * 100).toFixed(0)}% of them right`); }
  const b9 = before.filter(x => x.conf >= 0.9); console.log(`same threshold WITHOUT calibration: >= 0.9 answers ${(b9.length / before.length * 100).toFixed(0)}% of questions, ${(mean(b9, x => x.ok) * 100).toFixed(0)}% right`);
  let same = 0; for (const { z } of test) { const a = softmax(z, 1), b = softmax(z, bestT); if (a.indexOf(Math.max(...a)) === b.indexOf(Math.max(...b))) same++; }
  console.log("answers unchanged by the temperature:", same + " of " + test.length);
}

// ---------------------------------------------------------------------------------------------
function gate() {
  console.log("== Recalibration on a drifting stream (Chapter 27, Part 5) ==");
  R.seed = 4242;
  const K = 3;
  const item = (Ttrue, noiseSd) => { const z = Array.from({ length: K }, () => randn() * 2); const zr = z.map(v => v + randn() * noiseSd); const pt = softmax(zr, Ttrue); let u = rnd(), y = 0; for (; y < K - 1; y++) { u -= pt[y]; if (u <= 0) break; } return { z, y }; };
  const fitT = items => { let bt = 1, bn = Infinity; for (let T = 0.5; T <= 8; T += 0.05) { let n = 0; for (const it of items) n += -Math.log(Math.max(softmax(it.z, T)[it.y], 1e-12)); if (n < bn) { bn = n; bt = T; } } return bt; };
  const ece = rows => { let e = 0; const bins = 10; for (let b = 0; b < bins; b++) { const sel = rows.filter(r => (b === 0 ? r.conf >= b / bins : r.conf > b / bins) && r.conf <= (b + 1) / bins); if (sel.length) e += (sel.length / rows.length) * Math.abs(sel.reduce((a, r) => a + r.conf, 0) / sel.length - sel.reduce((a, r) => a + r.ok, 0) / sel.length); } return e; };
  console.log("PART 1  stability of the fitted temperature (true T = 2.5, 200 repeats each)");
  for (const n of [20, 50, 100, 200, 400, 1000]) { const ts = []; for (let r = 0; r < 200; r++) ts.push(fitT(Array.from({ length: n }, () => item(2.5, 0)))); const m = ts.reduce((a, b) => a + b) / ts.length, sd = Math.sqrt(ts.reduce((a, t) => a + (t - m) ** 2, 0) / ts.length); console.log(`  N=${String(n).padStart(4)}  fitted T mean ${f(m, 2)}  spread (sd) +-${f(sd, 2)}   -> typical error ${(sd / 2.5 * 100).toFixed(0)}%`); }
  const run = (policy, scenario) => {
    R.seed = 99;
    const N = 8000, driftAt = 3000; let T = 1; const buf = []; const rows = []; let refits = 0, labelled = 0, sinceFit = 0, fitted = false; const recent = [];
    for (let i = 0; i < N; i++) {
      const drifted = i >= driftAt;
      const it = scenario === "A" ? item(drifted ? 4.0 : 2.5, 0) : item(2.5, drifted ? 2.0 : 0);
      const p = softmax(it.z, T), c = Math.max(...p), ok = p.indexOf(c) === it.y ? 1 : 0; rows.push({ conf: c, ok, i }); recent.push({ conf: c, ok }); if (recent.length > 200) recent.shift();
      buf.push(it); if (buf.length > 400) buf.shift(); labelled++; sinceFit++;
      if (policy === "static") { if (!fitted && labelled === 400) { T = fitT(buf); fitted = true; refits++; } }
      else if (policy === "rolling") { if ((!fitted && labelled >= 200) || sinceFit >= 200) { T = fitT(buf); fitted = true; refits++; sinceFit = 0; } }
      else { if (!fitted && labelled >= 200) { T = fitT(buf); fitted = true; refits++; sinceFit = 0; } else if (fitted && sinceFit >= 200 && recent.length === 200 && ece(recent) > 0.08) { T = fitT(buf); refits++; sinceFit = 0; } }
    }
    const win = (a, b) => rows.filter(r => r.i >= a && r.i < b), acc = a => a.reduce((s, r) => s + r.ok, 0) / a.length;
    return { refits, b: ece(win(1000, 3000)), a1: ece(win(3000, 4000)), late: ece(win(5000, 8000)), accB: acc(win(1000, 3000)), accL: acc(win(5000, 8000)) };
  };
  for (const [scenario, label] of [["A", "the world gets noisier (true T 2.5 -> 4.0)"], ["B", "the scores stop tracking reality (extra noise)"]]) {
    console.log(`\nPART 2  scenario ${scenario}: ${label}, drift at question 3,000`);
    console.log("  policy        refits | ECE before drift | ECE first 1,000 after | ECE late | accuracy before -> late");
    for (const p of ["static", "rolling", "gated"]) { const r = run(p, scenario); console.log(`  ${p.padEnd(12)}  ${String(r.refits).padStart(5)} |      ${f(r.b)}       |        ${f(r.a1)}          |  ${f(r.late)}  |  ${pct(r.accB)} -> ${pct(r.accL)}`); }
  }
}

// ---------------------------------------------------------------------------------------------
function margins() {
  console.log("== Margin of error on an accuracy number (Chapter 27, Part 6.1) ==");
  const moe = (p, n) => 1.96 * Math.sqrt(p * (1 - p) / n) * 100;
  for (const n of [100, 130, 400, 1200, 2000]) console.log("  N=" + String(n).padStart(4), " at 77%: +-" + f(moe(0.77, n), 1), " at 50%: +-" + f(moe(0.5, n), 1));
  console.log("unpaired margin, 0.766 vs 0.727 (N=2000 each): +-" + f(Math.sqrt(0.766 * 0.234 / 2000 + 0.727 * 0.273 / 2000) * 1.96 * 100, 1) + " points (observed gap 3.9)");
  console.log("unpaired margin, 80.50 vs 79.08 (N=1200 each): +-" + f(Math.sqrt(0.805 * 0.195 / 1200 + 0.7908 * 0.2092 / 1200) * 1.96 * 100, 1) + " points (observed gap 1.4)");
}

// ---------------------------------------------------------------------------------------------
function costs() {
  console.log("== Compute and memory rules of thumb (Chapter 26, Part 3.8 and Chapter 27, Part 3) ==");
  const G = 1e9, six = (params, tokens) => 6 * params * tokens;
  const vegaNonEmb = 24.35 * G, layaNonEmb = 370e6;
  console.log("forward FLOPs per token  Vega", f(2 * vegaNonEmb / G, 1) + " GFLOP | Laya", f(2 * layaNonEmb / G, 2) + " GFLOP | ratio", f(vegaNonEmb / layaNonEmb, 0) + "x");
  for (const N of [100, 300, 500]) console.log(`  request of ${N} tokens: Vega ${f(2 * vegaNonEmb * N / 1e12, 1)} TFLOP | Laya ${f(2 * layaNonEmb * N / 1e12, 2)} TFLOP`);
  const layaFT = six(layaNonEmb, 6000 * 4 * 250), bertPre = six(85e6, 1e6 * 128000), modernPre = six(343e6, 2e12);
  console.log("Laya fine-tune (6k decisions x 4 epochs x ~250 tokens):", layaFT.toExponential(1), "FLOP");
  console.log("BERT-base pretraining:", bertPre.toExponential(1), "(" + f(bertPre / layaFT, 0) + "x) | ModernBERT-large pretraining:", modernPre.toExponential(1), "(" + (modernPre / layaFT).toExponential(1) + "x)");
  console.log("training memory at ~16 bytes/parameter:");
  for (const [n, p] of [["Laya 421M", 421e6], ["Vega backbone 27B", 27.8e9], ["a 175B model", 175e9]]) console.log(`  ${n.padEnd(20)} ~${f(p * 16 / 1e9, 0)} GB before activations`);
  const N = vegaNonEmb, ad128 = 3.74 * G / 4;
  const lora = 4 * N + 6 * ad128, loraCkpt = lora + 2 * N, layaPerTok = 6 * 0.37 * G;
  console.log("LoRA on the 27B, per token:", f(lora / G, 0), "GFLOP (", f(loraCkpt / G, 0), "with checkpointing) vs Laya full fine-tune", f(layaPerTok / G, 1), "->", f(lora / layaPerTok, 0) + "x to " + f(loraCkpt / layaPerTok, 0) + "x");
  for (const tok of [10e6, 100e6, 1e9]) { const h = tok * lora / (400e12 * 3600); console.log(`  ${(tok / 1e6).toFixed(0).padStart(5)}M tokens: ${f(h, 1)} GPU-hours per member (400 effective TFLOPS), x4 members = ${f(4 * h, 0)}`); }
  console.log("state sharing, 300-token state, 80 tokens per question:");
  for (const Q of [5, 10]) { const d = Q * 380, s = 300 + Q * 80; console.log(`  ${Q} questions: default ${d} vs shared ${s} token-passes = ${f(d / s, 2)}x fewer`); }
}

const all = { bert, dials, rps, rl, calibration, gate, margins, costs };
const which = process.argv[2] || "all";
if (which === "all") { for (const [name, fn] of Object.entries(all)) { console.log("\n######## " + name + " ########"); fn(); } }
else if (all[which]) all[which]();
else { console.log("unknown experiment:", which, "| choose one of:", Object.keys(all).join(", "), "or all"); process.exit(1); }
