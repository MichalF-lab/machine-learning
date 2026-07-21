const PptxGenJS = require("pptxgenjs");
const path = require("path");

const ROOT = __dirname;
const p = (...parts) => path.join(ROOT, ...parts);

const C = {
  deep:   "065A82",
  teal:   "1C7293",
  mid:    "21295C",
  accent: "02C39A",
  light:  "F4F7FA",
  text:   "2C3E50",
  muted:  "7B8794",
  bad:    "D64545",
  good:   "2E9E5B",
  white:  "FFFFFF"
};

const pres = new PptxGenJS();
pres.defineLayout({ name: "WIDE", width: 10, height: 5.63 });
pres.layout = "WIDE";

function header(slide, title) {
  slide.addText(title, {
    x: 0.6, y: 0.35, w: 8.8, h: 0.55,
    fontSize: 26, bold: true, color: C.deep, fontFace: "Georgia"
  });
}

function lightSlide(title) {
  const s = pres.addSlide();
  s.background = { color: C.light };
  if (title) header(s, title);
  return s;
}

function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: C.mid };
  s.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: 0.25, h: 5.63, fill: { color: C.accent }, line: { type: "none" }
  });
  return s;
}

function heatColor(v, lo, hi) {
  let t = (v - lo) / (hi - lo);
  t = Math.max(0, Math.min(1, t));
  const c1 = [0xD6, 0x45, 0x45], c2 = [0x2E, 0x9E, 0x5B];
  const rgb = c1.map((v0, i) => Math.round(v0 + (c2[i] - v0) * t));
  return rgb.map(x => x.toString(16).padStart(2, "0")).join("").toUpperCase();
}

function imageGrid(slide, items, { cols, top, areaH, gap = 0.15 }) {
  const left = 0.5, right = 0.5;
  const areaW = 10 - left - right;
  const rows = Math.ceil(items.length / cols);
  const cellW = (areaW - gap * (cols - 1)) / cols;
  const cellH = (areaH - gap * (rows - 1)) / rows;
  const imgH = cellH - 0.28;
  items.forEach((it, i) => {
    const r = Math.floor(i / cols), c = i % cols;
    const x = left + c * (cellW + gap);
    const y = top + r * (cellH + gap);
    slide.addShape(pres.ShapeType.rect, {
      x, y, w: cellW, h: imgH, fill: { color: "FFFFFF" }, line: { color: C.teal, width: 0.75 }
    });
    slide.addImage({
      path: it.src, x: x + 0.04, y: y + 0.04, w: cellW - 0.08, h: imgH - 0.08,
      sizing: { type: "contain", w: cellW - 0.08, h: imgH - 0.08 }
    });
    slide.addText(it.label, {
      x, y: y + imgH, w: cellW, h: 0.26, fontSize: 11, bold: true, align: "center", color: C.teal
    });
  });
}

function videoGrid(slide, items, { cols, top, areaH, gap = 0.25 }) {
  const left = 0.6, right = 0.6;
  const areaW = 10 - left - right;
  const rows = Math.ceil(items.length / cols);
  const cellW = (areaW - gap * (cols - 1)) / cols;
  const cellH = (areaH - gap * (rows - 1)) / rows;
  const vidH = cellH - 0.32;
  items.forEach((it, i) => {
    const r = Math.floor(i / cols), c = i % cols;
    const x = left + c * (cellW + gap);
    const y = top + r * (cellH + gap);
    slide.addText(it.label, {
      x, y, w: cellW, h: 0.28, fontSize: 12, bold: true, align: "center", color: C.deep
    });
    slide.addMedia({
      type: "video", path: it.src, x: x + (cellW - 3.2) / 2 > 0 ? x + (cellW - Math.min(cellW, vidH * 1.5)) / 2 : x,
      y: y + 0.3, w: Math.min(cellW, vidH * 1.5), h: vidH
    });
  });
}

// ============================================================ 1. TYTUŁ
{
  const s = darkSlide();
  s.addText("Wyniki: optymalizacja agenta RL", {
    x: 0.9, y: 1.5, w: 8.5, h: 0.9, fontSize: 36, bold: true, color: "FFFFFF", fontFace: "Georgia"
  });
  s.addText("w środowisku Gymnasium", {
    x: 0.9, y: 2.3, w: 8.5, h: 0.8, fontSize: 36, bold: true, color: C.accent, fontFace: "Georgia"
  });
}

// ============================================================ 2. CEL I PYTANIA BADAWCZE
{
  const s = lightSlide("Cel i pytania badawcze");
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.6, y: 1.25, w: 8.8, h: 0.7, rectRadius: 0.08, fill: { color: "E3ECF2" }, line: { type: "none" }
  });
  s.addText([
    { text: "CEL   ", options: { bold: true, color: C.deep, fontSize: 13 } },
    { text: "poprawić agenta RL względem baseline — tuning, reward shaping, zaawansowany algorytm", options: { color: C.text, fontSize: 12.5 } }
  ], { x: 0.95, y: 1.25, w: 8.2, h: 0.7, valign: "middle" });

  const qs = [
    "Który element daje największą poprawę jakości uczenia?",
    "Czy bardziej złożony algorytm zawsze wygrywa?",
    "Czy tuning jest ważniejszy niż wybór metody?"
  ];
  let qy = 2.3;
  qs.forEach((q, i) => {
    s.addText(String(i + 1).padStart(2, "0"), {
      x: 0.6, y: qy, w: 0.9, h: 0.8, fontSize: 34, bold: true, color: C.accent, fontFace: "Georgia", valign: "middle"
    });
    s.addText(q, { x: 1.6, y: qy, w: 7.6, h: 0.8, fontSize: 16, color: C.text, valign: "middle" });
    qy += 0.9;
  });
}

// ============================================================ 4. BASELINE
{
  const s = lightSlide("Baseline — DQN vs REINFORCE");
  imageGrid(s, [
    { src: p("wyniki/plots/baseline/DQN_MountainCar-v0.png"), label: "DQN" },
    { src: p("wyniki/plots/baseline/REINFORCE_MountainCar-v0.png"), label: "REINFORCE" }
  ], { cols: 2, top: 1.3, areaH: 3.85 });
}

// ============================================================ 5. HIPERPARAMETRY — PODSUMOWANIE
{
  const s = lightSlide("Wpływ hiperparametrów — podsumowanie ewaluacyjne (DQN)");
  imageGrid(s, [
    { src: p("wyniki/plots/eval_mean_all_variants_vs_gamma.png"), label: "gamma" },
    { src: p("wyniki/plots/eval_mean_all_variants_vs_epsilon_decay.png"), label: "epsilon_decay" }
  ], { cols: 2, top: 1.3, areaH: 3.85 });
}

// ============================================================ 6. GAMMA — SZCZEGÓŁY
{
  const s = lightSlide("Gamma — warianty A–E (DQN)");
  const gammaDir = "wyniki/plots/gamma";
  imageGrid(s, [
    { src: p(gammaDir, "Eksperyment_1/DQN_variant_A_DQN_variant_B_DQN_variant_C_DQN_variant_D_DQN_variant_E_gamma_0900_MountainCar-v0_DQN_shaping.png"), label: "γ = 0.90" },
    { src: p(gammaDir, "Eksperyment_2/DQN_variant_A_DQN_variant_B_DQN_variant_C_DQN_variant_D_DQN_variant_E_gamma_0950_MountainCar-v0_DQN_shaping.png"), label: "γ = 0.95" },
    { src: p(gammaDir, "Eksperyment_3/DQN_variant_A_DQN_variant_B_DQN_variant_C_DQN_variant_D_DQN_variant_E_gamma_0990_MountainCar-v0_DQN_shaping.png"), label: "γ = 0.99" },
    { src: p(gammaDir, "Eksperyment_4/DQN_variant_A_DQN_variant_B_DQN_variant_C_DQN_variant_D_DQN_variant_E_gamma_0995_MountainCar-v0_DQN_shaping.png"), label: "γ = 0.995" },
    { src: p(gammaDir, "Eksperyment_5/DQN_variant_A_DQN_variant_B_DQN_variant_C_DQN_variant_D_DQN_variant_E_gamma_0999_MountainCar-v0_DQN_shaping.png"), label: "γ = 0.999" }
  ], { cols: 3, top: 1.3, areaH: 3.85 });
}

// ============================================================ 7. EPSILON_DECAY — SZCZEGÓŁY
{
  const s = lightSlide("Epsilon_decay — warianty A–E (DQN)");
  const epsDir = "wyniki/plots/epsilon_decay";
  imageGrid(s, [
    { src: p(epsDir, "Eksperyment_0/DQN_variant_A_DQN_variant_B_DQN_variant_C_DQN_variant_D_DQN_variant_E_e_d_0990_MountainCar-v0_DQN_shaping.png"), label: "ε_decay = 0.990" },
    { src: p(epsDir, "Eksperyment_1/DQN_variant_A_DQN_variant_B_DQN_variant_C_DQN_variant_D_DQN_variant_E_e_d_0995_MountainCar-v0_DQN_shaping.png"), label: "ε_decay = 0.995" },
    { src: p(epsDir, "Eksperyment_2/DQN_variant_A_DQN_variant_B_DQN_variant_C_DQN_variant_D_DQN_variant_E_e_d_0996_MountainCar-v0_DQN_shaping.png"), label: "ε_decay = 0.996" },
    { src: p(epsDir, "Eksperyment_3/DQN_variant_A_DQN_variant_B_DQN_variant_C_DQN_variant_D_DQN_variant_E_e_d_0997_MountainCar-v0_DQN_shaping.png"), label: "ε_decay = 0.997" },
    { src: p(epsDir, "Eksperyment_4/DQN_variant_A_DQN_variant_B_DQN_variant_C_DQN_variant_D_DQN_variant_E_e_d_0998_MountainCar-v0_DQN_shaping.png"), label: "ε_decay = 0.998" },
    { src: p(epsDir, "Eksperyment_5/DQN_variant_A_DQN_variant_B_DQN_variant_C_DQN_variant_D_DQN_variant_E_e_d_0999_MountainCar-v0_DQN_shaping.png"), label: "ε_decay = 0.999" }
  ], { cols: 3, top: 1.3, areaH: 3.85 });
}

// ============================================================ heatmap table helper
function heatmapSlide(label, colHeaders, rowsData, lo, hi, colW) {
  const s = pres.addSlide();
  s.background = { color: C.light };
  s.addText(label, {
    x: 0.6, y: 0.4, w: 8.8, h: 0.6, fontSize: 22, bold: true, color: C.deep, fontFace: "Georgia"
  });
  const tableHeader = [{ text: "Wariant", options: { fill: { color: C.deep }, color: "FFFFFF", bold: true, align: "center", fontSize: 13 } }]
    .concat(colHeaders.map(h => ({ text: h, options: { fill: { color: C.deep }, color: "FFFFFF", bold: true, align: "center", fontSize: 13 } })));
  const rows = [tableHeader].concat(rowsData.map(([variant, vals]) => {
    const row = [{ text: variant, options: { fill: { color: C.teal }, color: "FFFFFF", bold: true, align: "center", fontSize: 13 } }];
    vals.forEach(v => {
      row.push({ text: String(v), options: { fill: { color: heatColor(v, lo, hi) }, color: "FFFFFF", bold: true, align: "center", fontSize: 13 } });
    });
    return row;
  }));
  s.addTable(rows, {
    x: 0.6, y: 1.2, w: 8.8, colW,
    border: { type: "solid", color: "FFFFFF", pt: 1.5 },
    valign: "middle", rowH: 0.6, autoPage: false
  });
  return s;
}

// ============================================================ 8. HEATMAPA GAMMA
{
  heatmapSlide(
    "gamma",
    ["0.90", "0.95", "0.99", "0.995", "0.999"],
    [
      ["A", [-200, -200, -200, -197, -200]],
      ["B", [-115, -116, -124, -117, -120]],
      ["C", [-184, -145, -168, -110, -108]],
      ["D", [-152, -106, -189, -189, -196]],
      ["E", [-200, -200, -200, -200, -200]]
    ],
    -200, -106,
    [1.6, 1.44, 1.44, 1.44, 1.44, 1.44]
  );
}

// ============================================================ 9. HEATMAPA EPSILON_DECAY
{
  heatmapSlide(
    "epsilon_decay",
    ["0.990", "0.995", "0.996", "0.997", "0.998", "0.999"],
    [
      ["A", [-200, -200, -197, -197, -200, -138]],
      ["B", [-143, -200, -104, -137, -130, -135]],
      ["C", [-132, -123, -115, -136, -123, -137]],
      ["D", [-178, -200, -195, -183, -181, -200]],
      ["E", [-200, -200, -200, -200, -200, -200]]
    ],
    -200, -104,
    [1.4, 1.23, 1.23, 1.23, 1.23, 1.23, 1.23]
  );
}

// ============================================================ 10. REWARD SHAPING — KRZYWE UCZENIA
{
  const s = lightSlide("Reward shaping — krzywe uczenia");
  imageGrid(s, [
    { src: p("wyniki/plots/DQN_variant_A_DQN_variant_B_DQN_variant_C_DQN_variant_D_DQN_variant_E_MountainCar-v0_DQN_shaping.png"), label: "DQN" },
    { src: p("wyniki/plots/REINFORCE_variant_A_REINFORCE_variant_B_REINFORCE_variant_C_REINFORCE_variant_D_REINFORCE_variant_E_MountainCar-v0_REINFORCE_shaping.png"), label: "REINFORCE" }
  ], { cols: 2, top: 1.3, areaH: 3.85 });
}

// ============================================================ 11. TABELA METRYK TRENINGOWYCH
{
  const s = lightSlide("Tabela — metryki treningowe reward shaping");
  const cols = ["Seria", "Śr. last-100", "Std last-100", "Best avg_100", "Epizod"];
  const dqnRows = [
    ["DQN baseline", -199.8, 2.1, -197.2, 411],
    ["wariant A", 42.0, 86.1, 42.8, 498],
    ["wariant B", -105.4, 13.1, -105.4, 500],
    ["wariant C", -109.1, 18.0, -108.7, 490],
    ["wariant D", 32.5, 59.3, 113.6, 358],
    ["wariant E", -160.3, 4.7, -156.4, 330]
  ];
  const reiRows = [
    ["REINFORCE baseline", -200.0, 0.0, -200.0, 1],
    ["wariant A", -52.2, 16.3, -50.4, 437],
    ["wariant B", -189.0, 4.6, -177.8, 1],
    ["wariant C", -182.5, 4.2, -181.6, 486],
    ["wariant D", -5.4, 6.2, -0.3, 14],
    ["wariant E", -155.9, 3.0, -155.8, 493]
  ];
  function metricTable(rowsData, x) {
    const header = cols.map(h => ({ text: h, options: { fill: { color: C.deep }, color: "FFFFFF", bold: true, fontSize: 10, align: "center" } }));
    const body = rowsData.map((r, i) => r.map((val, ci) => ({
      text: String(val),
      options: {
        fill: { color: i % 2 === 0 ? "FFFFFF" : "EDF2F6" },
        color: C.text, fontSize: 10, align: ci === 0 ? "left" : "center", bold: ci === 0
      }
    })));
    s.addTable([header].concat(body), {
      x, y: 1.55, w: 4.25, colW: [1.55, 0.75, 0.7, 0.8, 0.6],
      border: { type: "solid", color: "FFFFFF", pt: 1 }, valign: "middle", rowH: 0.34
    });
  }
  metricTable(dqnRows, 0.45);
  metricTable(reiRows, 5.05);
}

// ============================================================ 12. REWARD SHAPING — POLITYKI
{
  const s = lightSlide("Reward shaping — polityki i porównanie wariantów");
  imageGrid(s, [
    { src: p("nagrania_reward_shaping_baseline/all_policies.png"), label: "Wszystkie polityki" },
    { src: p("nagrania_reward_shaping_baseline/best_variants_comparison.png"), label: "Najlepsze warianty" }
  ], { cols: 2, top: 1.3, areaH: 3.85 });
}

// ============================================================ 13–14. NAGRANIA
{
  const s = lightSlide("Nagrania — baseline i warianty A–C (DQN)");
  const dir = "nagrania_reward_shaping_baseline";
  videoGrid(s, [
    { src: p(dir, "dqn_baseline.mp4"), label: "Baseline" },
    { src: p(dir, "mountaincar_variant_dqn_A_best.mp4"), label: "Wariant A — best" },
    { src: p(dir, "mountaincar_variant_dqn_B_best.mp4"), label: "Wariant B — best" },
    { src: p(dir, "mountaincar_variant_dqn_C_best.mp4"), label: "Wariant C — best" }
  ], { cols: 2, top: 1.3, areaH: 3.85 });
}
{
  const s = lightSlide("Nagrania — warianty D i E (DQN), best vs bad/worst");
  const dir = "nagrania_reward_shaping_baseline";
  videoGrid(s, [
    { src: p(dir, "mountaincar_dqn_variant_D_best.mp4"), label: "Wariant D — best" },
    { src: p(dir, "mountaincar_dqn_variant_D_bad.mp4"), label: "Wariant D — bad" },
    { src: p(dir, "mountaincar_dqn_variant_E_best.mp4"), label: "Wariant E — best" },
    { src: p(dir, "mountaincar_dqn_variant_E_worst.mp4"), label: "Wariant E — worst" }
  ], { cols: 2, top: 1.3, areaH: 3.85 });
}

// ============================================================ 16. ANIMACJE
{
  const s = pres.addSlide();
  s.background = { color: C.light };
  const src = p("animacje/05_car_on_hill_progression.gif");
  const w = 7.2, h = 3.3;
  const x = (10 - w) / 2, y = 1.0;
  s.addShape(pres.ShapeType.rect, {
    x, y, w, h, fill: { color: "FFFFFF" }, line: { color: C.teal, width: 0.75 }
  });
  s.addImage({ path: src, x: x + 0.05, y: y + 0.05, w: w - 0.1, h: h - 0.1, sizing: { type: "contain", w: w - 0.1, h: h - 0.1 } });
  s.addText("Auto na wzgórzu", {
    x, y: y + h + 0.1, w, h: 0.4, fontSize: 16, bold: true, align: "center", color: C.deep
  });
}

// ============================================================ 17. WNIOSKI
{
  const s = darkSlide();
  s.addText("Wnioski", { x: 0.9, y: 0.45, w: 8.5, h: 0.55, fontSize: 28, bold: true, color: "FFFFFF", fontFace: "Georgia" });

  const concl = [
    ["01", "Co poprawia jakość uczenia najbardziej?", "Reward shaping, zwłaszcza warianty A i D, daje większy i wyraźniejszy skok jakości niż sam tuning gamma i epsilon_decay."],
    ["02", "Czy złożoność zawsze wygrywa?", "Nie. Prosty DQN z dobrze zaprojektowanym reward shapingiem radykalnie bije baseline, co pokazuje, że jakość nagrody ma większe znaczenie niż złożoność samego algorytmu."],
    ["03", "Tuning czy wybór metody?", "Wybór wariantu reward shaping ma większy wpływ niż tuning hiperparametrów, ponieważ wariant E pozostaje na poziomie minus dwustu niezależnie od wartości gamma i epsilon_decay."]
  ];
  let cy = 1.2;
  concl.forEach(([n, q, a]) => {
    s.addText(n, { x: 0.9, y: cy, w: 0.8, h: 1.1, fontSize: 30, bold: true, color: C.accent, fontFace: "Georgia", valign: "top" });
    s.addText(q, { x: 1.75, y: cy, w: 7.4, h: 0.4, fontSize: 14, bold: true, color: "FFFFFF" });
    s.addText(a, { x: 1.75, y: cy + 0.38, w: 7.4, h: 0.65, fontSize: 12, color: "CADCFC" });
    cy += 1.18;
  });
}

const outputPath = p("prezentacja_koncowa_rl_gymnasium.pptx");
pres.writeFile({ fileName: outputPath }).then(() => {
  console.log("OK ->", outputPath);
}).catch(err => {
  console.error("FAIL", err);
  process.exit(1);
});
