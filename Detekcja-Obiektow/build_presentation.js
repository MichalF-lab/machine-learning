// Prezentacja: GRUPA 2 - Detekcja Obiektów
// Postępy projektu - sieci neuronowe (v2 - po poprawkach)

process.env.NODE_PATH = 'C:\\Users\\Michal\\AppData\\Roaming\\npm\\node_modules';
require('module').Module._initPaths();

const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const {
  FaBullseye, FaDatabase, FaCogs, FaChartLine, FaCubes, FaProjectDiagram,
  FaCode, FaTasks, FaRoute, FaImage, FaMicroscope, FaCar, FaPlane,
  FaCheckCircle, FaHourglassHalf, FaLayerGroup, FaCrosshairs, FaBalanceScale,
  FaRocket, FaBrain, FaNetworkWired, FaListAlt, FaFlask, FaArrowRight, FaBolt
} = require("react-icons/fa");

// === Paleta ===
const C = {
  navy:        "0F1B3D",
  navyDark:    "0A1228",
  primary:     "1E3A8A",
  teal:        "0891B2",
  tealLight:   "06B6D4",
  accent:      "F59E0B",
  accentRed:   "DC2626",
  accentGreen: "16A34A",
  bgLight:     "F8FAFC",
  bgCard:      "FFFFFF",
  bgSoft:      "EFF6FF",
  text:        "1E293B",
  textMuted:   "64748B",
  textLight:   "94A3B8",
  border:      "E2E8F0",
};

async function ico(IconComponent, color = "#" + C.primary, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

async function build() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.author = "GRUPA 2";
  pres.title = "Detekcja Obiektów - Postępy projektu";

  const I = {
    target:  await ico(FaBullseye,        "#" + C.tealLight),
    db:      await ico(FaDatabase,        "#" + C.tealLight),
    cogs:    await ico(FaCogs,            "#" + C.tealLight),
    chart:   await ico(FaChartLine,       "#" + C.tealLight),
    cubes:   await ico(FaCubes,           "#" + C.tealLight),
    diagram: await ico(FaProjectDiagram,  "#" + C.tealLight),
    code:    await ico(FaCode,            "#" + C.tealLight),
    tasks:   await ico(FaTasks,           "#" + C.tealLight),
    route:   await ico(FaRoute,           "#" + C.tealLight),
    image:   await ico(FaImage,           "#" + C.tealLight),
    micro:   await ico(FaMicroscope,      "#" + C.tealLight),
    car:     await ico(FaCar,             "#" + C.tealLight),
    plane:   await ico(FaPlane,           "#" + C.tealLight),
    check:   await ico(FaCheckCircle,     "#" + C.accentGreen),
    pending: await ico(FaHourglassHalf,   "#" + C.accent),
    layers:  await ico(FaLayerGroup,      "#" + C.tealLight),
    cross:   await ico(FaCrosshairs,      "#" + C.tealLight),
    scale:   await ico(FaBalanceScale,    "#" + C.tealLight),
    rocket:  await ico(FaRocket,          "#" + C.tealLight),
    brain:   await ico(FaBrain,           "#" + C.tealLight),
    network: await ico(FaNetworkWired,    "#" + C.tealLight),
    list:    await ico(FaListAlt,         "#" + C.tealLight),
    flask:   await ico(FaFlask,           "#" + C.tealLight),
    arrow:   await ico(FaArrowRight,      "#" + C.teal),
    bolt:    await ico(FaBolt,            "#" + C.tealLight),

    targetD:  await ico(FaBullseye,       "#" + C.primary),
    dbD:      await ico(FaDatabase,       "#" + C.primary),
    cogsD:    await ico(FaCogs,           "#" + C.primary),
    chartD:   await ico(FaChartLine,      "#" + C.primary),
    layersD:  await ico(FaLayerGroup,     "#" + C.primary),
    networkD: await ico(FaNetworkWired,   "#" + C.primary),
    flaskD:   await ico(FaFlask,          "#" + C.primary),
    rocketD:  await ico(FaRocket,         "#" + C.primary),
    crossD:   await ico(FaCrosshairs,     "#" + C.primary),
    diagramD: await ico(FaProjectDiagram, "#" + C.primary),
    listD:    await ico(FaListAlt,        "#" + C.primary),
    brainD:   await ico(FaBrain,          "#" + C.primary),
    routeD:   await ico(FaRoute,          "#" + C.primary),
    tasksD:   await ico(FaTasks,          "#" + C.primary),
    microD:   await ico(FaMicroscope,     "#" + C.primary),
    carD:     await ico(FaCar,            "#" + C.primary),
    planeD:   await ico(FaPlane,          "#" + C.primary),
    boltD:    await ico(FaBolt,           "#" + C.primary),
  };

  // === Pasek tytułowy slajdu (bez stopki na dole) ===
  function topBar(slide, title) {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 0, w: 13.3, h: 0.9,
      fill: { color: C.navy }, line: { color: C.navy }
    });
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 0.9, w: 13.3, h: 0.04,
      fill: { color: C.tealLight }, line: { color: C.tealLight }
    });
    slide.addText(title, {
      x: 0.5, y: 0, w: 12, h: 0.9,
      fontSize: 26, fontFace: "Georgia", color: C.bgLight, bold: true,
      valign: "middle", margin: 0
    });
  }

  // ===========================================================
  // SLAJD 1 - TYTUŁOWY  (bez napisu "Implementacja...", bez GRUPA 2 box, bez "Raport postępów")
  // ===========================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };

    // Diagonalna linia akcentowa
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 6.5, w: 13.3, h: 0.04,
      fill: { color: C.tealLight }, line: { color: C.tealLight }
    });

    // Mały label
    s.addText("PROJEKT 2  ·  SIECI NEURONOWE", {
      x: 0.8, y: 2.1, w: 11, h: 0.45,
      fontSize: 16, fontFace: "Calibri", color: C.tealLight, bold: true,
      charSpacing: 10, margin: 0
    });

    // Tytuł główny - centrowany pionowo
    s.addText("Detekcja", {
      x: 0.8, y: 2.7, w: 12, h: 1.4,
      fontSize: 110, fontFace: "Georgia", color: C.bgLight, bold: true, margin: 0
    });
    s.addText("obiektów", {
      x: 0.8, y: 4.0, w: 12, h: 1.4,
      fontSize: 110, fontFace: "Georgia", color: C.tealLight, italic: true, margin: 0
    });

    // Linia akcent pod tytułem
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.8, y: 5.5, w: 1.5, h: 0.05,
      fill: { color: C.accent }, line: { color: C.accent }
    });
  }

  // ===========================================================
  // SLAJD 2 - AGENDA / ZAKRES (bez stopki)
  // ===========================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgLight };
    topBar(s, "Zakres prezentacji");

    const sections = [
      { n: "01", t: "Cel projektu",       d: "Sformułowanie problemu detekcji",            ic: I.targetD },
      { n: "02", t: "Wybrane datasety",   d: "BCCD · BDD100K · VisDrone",                  ic: I.dbD },
      { n: "03", t: "Modele detekcji",    d: "Implementacja YOLO, Faster R-CNN, SSD",      ic: I.networkD },
      { n: "04", t: "Ewaluacja",          d: "Metryki mAP, precision, recall",             ic: I.chartD },
      { n: "05", t: "Status zadań",       d: "Co zrobione, co w trakcie, co dalej",        ic: I.tasksD },
      { n: "06", t: "Plany dalszych prac", d: "Trening, wizualizacje, analiza błędów",     ic: I.rocketD },
    ];

    const cw = 4.0, ch = 2.7, gx = 0.4, gy = 0.4;
    const startX = (13.3 - (3 * cw + 2 * gx)) / 2;
    const startY = 1.55;

    sections.forEach((sec, i) => {
      const col = i % 3, row = Math.floor(i / 3);
      const x = startX + col * (cw + gx);
      const y = startY + row * (ch + gy);

      s.addShape(pres.shapes.RECTANGLE, {
        x, y, w: cw, h: ch,
        fill: { color: C.bgCard },
        line: { color: C.border, width: 0.75 },
        shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.06 }
      });
      s.addShape(pres.shapes.RECTANGLE, {
        x, y, w: 0.06, h: ch,
        fill: { color: C.tealLight }, line: { color: C.tealLight }
      });

      s.addText(sec.n, {
        x: x + 0.3, y: y + 0.25, w: 1, h: 0.5,
        fontSize: 26, fontFace: "Georgia", color: C.tealLight, bold: true, italic: true, margin: 0
      });

      s.addImage({ data: sec.ic, x: x + cw - 0.95, y: y + 0.3, w: 0.55, h: 0.55 });

      s.addText(sec.t, {
        x: x + 0.3, y: y + 1.05, w: cw - 0.6, h: 0.5,
        fontSize: 18, fontFace: "Calibri", color: C.text, bold: true, margin: 0
      });

      s.addShape(pres.shapes.RECTANGLE, {
        x: x + 0.3, y: y + 1.62, w: 0.6, h: 0.03,
        fill: { color: C.accent }, line: { color: C.accent }
      });

      s.addText(sec.d, {
        x: x + 0.3, y: y + 1.78, w: cw - 0.6, h: 0.7,
        fontSize: 12, fontFace: "Calibri", color: C.textMuted, margin: 0
      });
    });
  }

  // ===========================================================
  // SLAJD 3 - CEL PROJEKTU (bez prawej kolumny "Trzy modele, trzy datasety")
  //                       (zmieniona główna treść)
  // ===========================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgLight };
    topBar(s, "01  ·  Cel projektu");

    // Lewa - wprowadzenie (rozszerzone na całą szerokość)
    s.addText("GRUPA 2", {
      x: 0.6, y: 1.2, w: 12, h: 0.35,
      fontSize: 12, fontFace: "Calibri", color: C.tealLight, bold: true, charSpacing: 6, margin: 0
    });
    s.addText("Detekcja obiektów na obrazach", {
      x: 0.6, y: 1.5, w: 12, h: 0.7,
      fontSize: 32, fontFace: "Georgia", color: C.text, bold: true, margin: 0
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 2.3, w: 0.7, h: 0.04,
      fill: { color: C.accent }, line: { color: C.accent }
    });

    // Karta z głównym opisem (na całą szerokość)
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 2.55, w: 12.1, h: 1.45,
      fill: { color: C.bgCard }, line: { color: C.border, width: 0.75 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.06 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 2.55, w: 0.08, h: 1.45,
      fill: { color: C.tealLight }, line: { color: C.tealLight }
    });

    s.addText([
      { text: "Eksperymentujemy z różnorodnymi domenami: ", options: { color: C.text } },
      { text: "medyczną, miejską i lotniczą ", options: { color: C.primary, bold: true } },
      { text: "— aby zbadać, jak architektura modelu radzi sobie z ", options: { color: C.text } },
      { text: "różną gęstością obiektów i charakterem sceny", options: { color: C.primary, bold: true } },
      { text: ".", options: { color: C.text } }
    ], {
      x: 0.85, y: 2.7, w: 11.7, h: 1.2,
      fontSize: 16, fontFace: "Calibri", margin: 0, paraSpaceAfter: 4, valign: "middle"
    });

    // Cele projektu - 6 kart w siatce 3x2
    s.addText("Główne zadania", {
      x: 0.6, y: 4.25, w: 12, h: 0.4,
      fontSize: 16, fontFace: "Calibri", color: C.primary, bold: true, charSpacing: 4, margin: 0
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 4.65, w: 0.5, h: 0.03,
      fill: { color: C.accent }, line: { color: C.accent }
    });

    const goals = [
      "Przygotowanie zbiorów z anotacjami bounding box",
      "Trening modeli w porównywalnych warunkach",
      "Ocena jakości metrykami mAP, precision, recall",
      "Analiza wpływu architektury i parametrów",
      "Kompromis dokładność vs. szybkość",
      "Wizualizacje detekcji i analiza FP / FN"
    ];

    const gw = 4.0, gh = 0.95, ggx = 0.05, ggy = 0.15;
    const gStartX = (13.3 - (3 * gw + 2 * ggx)) / 2;
    const gStartY = 4.95;

    goals.forEach((g, i) => {
      const col = i % 3, row = Math.floor(i / 3);
      const x = gStartX + col * (gw + ggx);
      const y = gStartY + row * (gh + ggy);

      s.addShape(pres.shapes.RECTANGLE, {
        x, y, w: gw, h: gh,
        fill: { color: C.bgCard }, line: { color: C.border, width: 0.5 }
      });
      s.addShape(pres.shapes.OVAL, {
        x: x + 0.25, y: y + (gh - 0.45) / 2, w: 0.45, h: 0.45,
        fill: { color: C.tealLight }, line: { color: C.tealLight }
      });
      s.addText(String(i + 1), {
        x: x + 0.25, y: y + (gh - 0.45) / 2, w: 0.45, h: 0.45,
        fontSize: 18, fontFace: "Georgia", color: "FFFFFF", bold: true,
        align: "center", valign: "middle", margin: 0
      });
      s.addText(g, {
        x: x + 0.85, y: y, w: gw - 1.0, h: gh,
        fontSize: 12.5, fontFace: "Calibri", color: C.text, valign: "middle", margin: 0
      });
    });
  }

  // ===========================================================
  // SLAJD 4 - DATASETY (bez "Trzy domeny..." i bez sekcji FORMAT ANOTACJI)
  // ===========================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgLight };
    topBar(s, "02  ·  Wybrane datasety");

    const ds = [
      {
        name: "BCCD",
        sub: "Blood Cell Count and Detection",
        icon: I.micro,
        type: "Medyczny  ·  Mały zbiór",
        classes: "3 klasy",
        items: ["RBC — czerwone krwinki", "WBC — białe krwinki", "Platelets — płytki krwi"],
        color: C.accentRed
      },
      {
        name: "BDD100K",
        sub: "Berkeley DeepDrive — sceny uliczne",
        icon: I.car,
        type: "Driving  ·  Duży zbiór",
        classes: "10 klas",
        items: ["pieszy, rowerzysta, motocyklista", "samochód, ciężarówka, autobus, pociąg", "rower, sygnalizacja, znaki drogowe"],
        color: C.primary
      },
      {
        name: "VisDrone",
        sub: "Zdjęcia z drona — duża gęstość",
        icon: I.plane,
        type: "Aerial  ·  Duża gęstość",
        classes: "10 klas",
        items: ["pieszy, ludzie, rower", "auto, van, ciężarówka, autobus", "tricycle, awning-tricycle, motor"],
        color: C.accentGreen
      }
    ];

    const cw = 4.05, gx = 0.25;
    const startX = (13.3 - (3 * cw + 2 * gx)) / 2;
    const startY = 1.5;
    const ch = 5.6;

    ds.forEach((d, i) => {
      const x = startX + i * (cw + gx);

      s.addShape(pres.shapes.RECTANGLE, {
        x, y: startY, w: cw, h: ch,
        fill: { color: C.bgCard },
        line: { color: C.border, width: 0.75 },
        shadow: { type: "outer", color: "000000", blur: 10, offset: 3, angle: 90, opacity: 0.08 }
      });
      s.addShape(pres.shapes.RECTANGLE, {
        x, y: startY, w: cw, h: 0.55,
        fill: { color: d.color }, line: { color: d.color }
      });
      s.addText(d.type, {
        x: x + 0.3, y: startY + 0.05, w: cw - 0.6, h: 0.45,
        fontSize: 11, fontFace: "Calibri", color: "FFFFFF", bold: true, charSpacing: 4,
        valign: "middle", margin: 0
      });

      s.addImage({ data: d.icon, x: x + cw - 1.15, y: startY + 0.85, w: 0.8, h: 0.8 });

      s.addText(d.name, {
        x: x + 0.35, y: startY + 0.85, w: cw - 1.4, h: 0.7,
        fontSize: 30, fontFace: "Georgia", color: C.text, bold: true, margin: 0
      });

      s.addText(d.sub, {
        x: x + 0.35, y: startY + 1.55, w: cw - 0.7, h: 0.4,
        fontSize: 12, fontFace: "Calibri", color: C.textMuted, italic: true, margin: 0
      });

      s.addShape(pres.shapes.RECTANGLE, {
        x: x + 0.35, y: startY + 2.1, w: 0.6, h: 0.04,
        fill: { color: d.color }, line: { color: d.color }
      });

      s.addText("KLASY", {
        x: x + 0.35, y: startY + 2.3, w: cw - 0.7, h: 0.3,
        fontSize: 10, fontFace: "Calibri", color: C.textMuted, bold: true, charSpacing: 4, margin: 0
      });
      s.addText(d.classes, {
        x: x + 0.35, y: startY + 2.6, w: cw - 0.7, h: 0.55,
        fontSize: 24, fontFace: "Calibri", color: d.color, bold: true, margin: 0
      });

      s.addText(
        d.items.map((it, k) => ({
          text: it,
          options: { bullet: { code: "25CF" }, color: C.text, fontSize: 13, breakLine: k < d.items.length - 1 }
        })),
        {
          x: x + 0.35, y: startY + 3.4, w: cw - 0.7, h: 2.1,
          fontFace: "Calibri", margin: 0, paraSpaceAfter: 6
        }
      );
    });
  }

  // ===========================================================
  // SLAJD 5 - YOLO  (techniczne szczegóły zamiast generycznych)
  // ===========================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgLight };
    topBar(s, "03  ·  Model 1/3  ·  YOLO");

    // Lewa - hero
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 1.3, w: 4.4, h: 5.7,
      fill: { color: C.navy }, line: { color: C.navy }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 1.3, w: 0.08, h: 5.7,
      fill: { color: C.accentRed }, line: { color: C.accentRed }
    });

    s.addText("YOU ONLY LOOK ONCE", {
      x: 0.9, y: 1.55, w: 4.0, h: 0.35,
      fontSize: 11, fontFace: "Calibri", color: C.accentRed, bold: true, charSpacing: 4, margin: 0
    });

    s.addText("YOLO", {
      x: 0.9, y: 2.0, w: 4.0, h: 1.7,
      fontSize: 110, fontFace: "Georgia", color: C.bgLight, bold: true, margin: 0
    });

    s.addText("v8n", {
      x: 0.9, y: 3.7, w: 4.0, h: 0.6,
      fontSize: 32, fontFace: "Georgia", color: C.tealLight, italic: true, margin: 0
    });

    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.9, y: 4.5, w: 0.7, h: 0.04,
      fill: { color: C.accentRed }, line: { color: C.accentRed }
    });

    s.addText("Single-stage  ·  anchor-free", {
      x: 0.9, y: 4.65, w: 4.0, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: C.tealLight, italic: true, margin: 0
    });

    // 3 statystyki - kolumna
    const ystats = [
      { v: "225",   l: "warstw" },
      { v: "3.2M",  l: "parametrów" },
      { v: "8.7G",  l: "FLOPs" }
    ];
    ystats.forEach((st, i) => {
      const yy = 5.15 + i * 0.55;
      s.addText(st.v, {
        x: 0.9, y: yy, w: 1.8, h: 0.5,
        fontSize: 24, fontFace: "Georgia", color: C.bgLight, bold: true, valign: "middle", margin: 0
      });
      s.addText(st.l, {
        x: 2.75, y: yy, w: 2.15, h: 0.5,
        fontSize: 11, fontFace: "Calibri", color: C.textLight, charSpacing: 4,
        valign: "middle", margin: 0
      });
    });

    // Prawa - architektura techniczna (3 boxy)
    const detailY = 1.3;
    const detailX = 5.3;
    const detailW = 7.4;

    // Box 1: Architektura
    s.addShape(pres.shapes.RECTANGLE, {
      x: detailX, y: detailY, w: detailW, h: 2.55,
      fill: { color: C.bgCard }, line: { color: C.border, width: 0.75 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.06 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: detailX, y: detailY, w: 0.08, h: 2.55,
      fill: { color: C.accentRed }, line: { color: C.accentRed }
    });
    s.addImage({ data: I.networkD, x: detailX + 0.2, y: detailY + 0.18, w: 0.4, h: 0.4 });
    s.addText("Architektura sieci", {
      x: detailX + 0.7, y: detailY + 0.15, w: detailW - 1, h: 0.45,
      fontSize: 16, fontFace: "Calibri", color: C.text, bold: true, valign: "middle", margin: 0
    });

    // 3 etapy: Backbone → Neck → Head
    const stages = [
      { lbl: "BACKBONE",  name: "CSPDarknet53",   sub: "C2f modules · gradient flow" },
      { lbl: "NECK",      name: "PAN-FPN",         sub: "Path Aggregation Network" },
      { lbl: "HEAD",      name: "Decoupled head",  sub: "anchor-free · 3 skale (P3/P4/P5)" }
    ];
    const sw2 = (detailW - 0.5) / 3;
    stages.forEach((st, i) => {
      const sx = detailX + 0.2 + i * sw2;
      const sy = detailY + 0.75;

      s.addShape(pres.shapes.RECTANGLE, {
        x: sx, y: sy, w: sw2 - 0.1, h: 1.6,
        fill: { color: C.bgSoft }, line: { color: C.border, width: 0.5 }
      });
      s.addShape(pres.shapes.RECTANGLE, {
        x: sx, y: sy, w: sw2 - 0.1, h: 0.04,
        fill: { color: C.accentRed }, line: { color: C.accentRed }
      });
      s.addText(st.lbl, {
        x: sx + 0.15, y: sy + 0.15, w: sw2 - 0.4, h: 0.3,
        fontSize: 9, fontFace: "Calibri", color: C.accentRed, bold: true, charSpacing: 4, margin: 0
      });
      s.addText(st.name, {
        x: sx + 0.15, y: sy + 0.45, w: sw2 - 0.3, h: 0.45,
        fontSize: 14, fontFace: "Calibri", color: C.text, bold: true, margin: 0
      });
      s.addText(st.sub, {
        x: sx + 0.15, y: sy + 0.95, w: sw2 - 0.3, h: 0.55,
        fontSize: 10, fontFace: "Calibri", color: C.textMuted, italic: true, margin: 0
      });

      // Strzałka
      if (i < 2) {
        s.addShape(pres.shapes.LINE, {
          x: sx + sw2 - 0.13, y: sy + 0.8, w: 0.13, h: 0,
          line: { color: C.accentRed, width: 2, endArrowType: "triangle" }
        });
      }
    });

    // Box 2: Loss & Training
    s.addShape(pres.shapes.RECTANGLE, {
      x: detailX, y: detailY + 2.7, w: detailW, h: 1.95,
      fill: { color: C.bgCard }, line: { color: C.border, width: 0.75 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.06 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: detailX, y: detailY + 2.7, w: 0.08, h: 1.95,
      fill: { color: C.tealLight }, line: { color: C.tealLight }
    });
    s.addImage({ data: I.boltD, x: detailX + 0.2, y: detailY + 2.85, w: 0.4, h: 0.4 });
    s.addText("Funkcja straty i trening", {
      x: detailX + 0.7, y: detailY + 2.83, w: detailW - 1, h: 0.45,
      fontSize: 15, fontFace: "Calibri", color: C.text, bold: true, valign: "middle", margin: 0
    });

    const lossItems = [
      { l: "Klasyfikacja",  r: "BCE Loss (Binary Cross-Entropy)" },
      { l: "Lokalizacja",   r: "CIoU Loss (Complete IoU)" },
      { l: "Distribution",  r: "DFL — Distribution Focal Loss" },
      { l: "Optymalizator", r: "SGD lub AdamW · cosine LR schedule" }
    ];
    lossItems.forEach((it, i) => {
      const yy = detailY + 3.4 + i * 0.3;
      s.addShape(pres.shapes.RECTANGLE, {
        x: detailX + 0.25, y: yy, w: 0.06, h: 0.22,
        fill: { color: C.tealLight }, line: { color: C.tealLight }
      });
      s.addText(it.l, {
        x: detailX + 0.4, y: yy, w: 1.7, h: 0.22,
        fontSize: 11, fontFace: "Calibri", color: C.textMuted, bold: true, charSpacing: 2, valign: "middle", margin: 0
      });
      s.addText(it.r, {
        x: detailX + 2.15, y: yy, w: detailW - 2.4, h: 0.22,
        fontSize: 11, fontFace: "Calibri", color: C.text, valign: "middle", margin: 0
      });
    });

    // Box 3: Code snippet
    s.addShape(pres.shapes.RECTANGLE, {
      x: detailX, y: detailY + 4.8, w: detailW, h: 1.9,
      fill: { color: C.navyDark }, line: { color: C.navyDark }
    });
    s.addText("# Pipeline treningowy", {
      x: detailX + 0.2, y: detailY + 4.9, w: detailW - 0.4, h: 0.3,
      fontSize: 10, fontFace: "Consolas", color: C.tealLight, italic: true, margin: 0
    });
    s.addText([
      { text: "from ultralytics import YOLO", options: { color: C.tealLight, fontSize: 11.5, breakLine: true } },
      { text: "from data_interface import load_project_data", options: { color: C.tealLight, fontSize: 11.5, breakLine: true } },
      { text: "", options: { fontSize: 11.5, breakLine: true } },
      { text: "yaml_path = load_project_data('bccd', model_type='yolo')", options: { color: C.bgLight, fontSize: 11.5, breakLine: true } },
      { text: "model = YOLO('yolov8n.pt')", options: { color: C.bgLight, fontSize: 11.5, breakLine: true } },
      { text: "model.train(data=yaml_path, epochs=5, imgsz=640)", options: { color: C.accent, fontSize: 11.5 } }
    ], {
      x: detailX + 0.2, y: detailY + 5.2, w: detailW - 0.4, h: 1.55,
      fontFace: "Consolas", margin: 0, paraSpaceAfter: 1
    });
  }

  // ===========================================================
  // SLAJD 6 - FASTER R-CNN  (bez stopki)
  // ===========================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgLight };
    topBar(s, "03  ·  Model 2/3  ·  Faster R-CNN");

    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 1.3, w: 4.6, h: 5.7,
      fill: { color: C.navy }, line: { color: C.navy }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 1.3, w: 0.08, h: 5.7,
      fill: { color: C.tealLight }, line: { color: C.tealLight }
    });

    s.addText("REGION-BASED  ·  TWO-STAGE", {
      x: 0.9, y: 1.55, w: 4.0, h: 0.35,
      fontSize: 11, fontFace: "Calibri", color: C.tealLight, bold: true, charSpacing: 4, margin: 0
    });

    s.addText("Faster", {
      x: 0.9, y: 2.0, w: 4.0, h: 1.0,
      fontSize: 64, fontFace: "Georgia", color: C.bgLight, bold: true, margin: 0
    });
    s.addText("R-CNN", {
      x: 0.9, y: 2.95, w: 4.0, h: 1.0,
      fontSize: 64, fontFace: "Georgia", color: C.tealLight, italic: true, margin: 0
    });

    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.9, y: 4.15, w: 0.7, h: 0.04,
      fill: { color: C.accent }, line: { color: C.accent }
    });

    s.addText("Two-stage detector", {
      x: 0.9, y: 4.3, w: 4.0, h: 0.4,
      fontSize: 16, fontFace: "Calibri", color: C.tealLight, italic: true, margin: 0
    });

    s.addText("Najwyższa dokładność. RPN (Region Proposal Network) generuje propozycje, druga sieć je klasyfikuje i koryguje.", {
      x: 0.9, y: 4.75, w: 4.0, h: 1.4,
      fontSize: 12, fontFace: "Calibri", color: C.bgSoft, margin: 0
    });

    s.addText("STATUS:  pętla treningowa", {
      x: 0.9, y: 6.55, w: 4.0, h: 0.35,
      fontSize: 11, fontFace: "Calibri", color: C.accentGreen, bold: true, charSpacing: 4, margin: 0
    });

    // Prawa
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.5, y: 1.3, w: 7.3, h: 3.3,
      fill: { color: C.bgCard }, line: { color: C.border, width: 0.75 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.06 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.5, y: 1.3, w: 0.08, h: 3.3,
      fill: { color: C.tealLight }, line: { color: C.tealLight }
    });

    s.addText("Implementacja", {
      x: 5.7, y: 1.45, w: 6.95, h: 0.4,
      fontSize: 16, fontFace: "Calibri", color: C.text, bold: true, margin: 0
    });

    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.7, y: 1.9, w: 6.95, h: 2.55,
      fill: { color: C.navyDark }, line: { color: C.navyDark }
    });
    s.addText([
      { text: "import torchvision", options: { color: C.tealLight, fontSize: 11, breakLine: true } },
      { text: "from torchvision.models.detection.faster_rcnn \\", options: { color: C.tealLight, fontSize: 11, breakLine: true } },
      { text: "    import FastRCNNPredictor", options: { color: C.tealLight, fontSize: 11, breakLine: true } },
      { text: "", options: { fontSize: 11, breakLine: true } },
      { text: "model = torchvision.models.detection.fasterrcnn_resnet50_fpn(", options: { color: C.bgLight, fontSize: 11, breakLine: true } },
      { text: "    weights='DEFAULT')", options: { color: C.bgLight, fontSize: 11, breakLine: true } },
      { text: "in_features = model.roi_heads.box_predictor.cls_score.in_features", options: { color: C.bgLight, fontSize: 11, breakLine: true } },
      { text: "model.roi_heads.box_predictor = FastRCNNPredictor(", options: { color: C.accent, fontSize: 11, breakLine: true } },
      { text: "    in_features, num_classes=4)   # fine-tuning", options: { color: C.accent, fontSize: 11 } }
    ], {
      x: 5.85, y: 2.0, w: 6.65, h: 2.4,
      fontFace: "Consolas", margin: 0, paraSpaceAfter: 1
    });

    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.5, y: 4.75, w: 7.3, h: 2.25,
      fill: { color: C.bgCard }, line: { color: C.border, width: 0.75 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.06 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.5, y: 4.75, w: 0.08, h: 2.25,
      fill: { color: C.tealLight }, line: { color: C.tealLight }
    });
    s.addText("Konfiguracja treningu", {
      x: 5.7, y: 4.85, w: 6.95, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: C.text, bold: true, margin: 0
    });

    const frcnnFeats = [
      { l: "Backbone",       r: "ResNet-50 + FPN  (pre-trained ImageNet)" },
      { l: "Stage 1 — RPN",  r: "Region Proposal Network · obiektowność + bbox" },
      { l: "Stage 2 — RoI",  r: "RoI Align · klasyfikacja + box regression" },
      { l: "Optymalizator",  r: "SGD · lr=0.005 · momentum=0.9 · wd=5e-4" },
      { l: "Ewaluacja",      r: "torchmetrics MeanAveragePrecision (xyxy)" }
    ];

    frcnnFeats.forEach((f, i) => {
      const yy = 5.3 + i * 0.32;
      s.addShape(pres.shapes.RECTANGLE, {
        x: 5.7, y: yy, w: 0.06, h: 0.25,
        fill: { color: C.accent }, line: { color: C.accent }
      });
      s.addText(f.l, {
        x: 5.85, y: yy, w: 2.0, h: 0.25,
        fontSize: 10, fontFace: "Calibri", color: C.textMuted, bold: true, charSpacing: 2, valign: "middle", margin: 0
      });
      s.addText(f.r, {
        x: 7.9, y: yy, w: 4.85, h: 0.25,
        fontSize: 11, fontFace: "Calibri", color: C.text, valign: "middle", margin: 0
      });
    });
  }

  // ===========================================================
  // SLAJD 7 - SSD (custom)  (bez stopki, bez zmian merytorycznych)
  // ===========================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgLight };
    topBar(s, "03  ·  Model 3/3  ·  SSD  (własna implementacja)");

    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 1.3, w: 4.0, h: 5.7,
      fill: { color: C.navy }, line: { color: C.navy }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 1.3, w: 0.08, h: 5.7,
      fill: { color: C.accent }, line: { color: C.accent }
    });

    s.addText("SINGLE SHOT", {
      x: 0.85, y: 1.55, w: 3.5, h: 0.4,
      fontSize: 11, fontFace: "Calibri", color: C.accent, bold: true, charSpacing: 6, margin: 0
    });
    s.addText("MULTIBOX DETECTOR", {
      x: 0.85, y: 1.9, w: 3.5, h: 0.4,
      fontSize: 11, fontFace: "Calibri", color: C.tealLight, charSpacing: 6, margin: 0
    });

    s.addText("SSD", {
      x: 0.85, y: 2.5, w: 3.5, h: 1.5,
      fontSize: 110, fontFace: "Georgia", color: C.bgLight, bold: true, margin: 0
    });

    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.85, y: 4.1, w: 0.7, h: 0.04,
      fill: { color: C.accent }, line: { color: C.accent }
    });

    s.addText("Single-stage  ·  multi-scale", {
      x: 0.85, y: 4.25, w: 3.5, h: 0.4,
      fontSize: 14, fontFace: "Calibri", color: C.tealLight, italic: true, margin: 0
    });

    s.addText("Implementowany od podstaw — szkielet VGG-16 + dodatkowe warstwy konwolucyjne. Predykcja na 6 mapach cech.", {
      x: 0.85, y: 4.7, w: 3.5, h: 1.5,
      fontSize: 11, fontFace: "Calibri", color: C.bgSoft, margin: 0
    });

    s.addText("STATUS:  szkielet sieci + anchory", {
      x: 0.85, y: 6.55, w: 3.5, h: 0.35,
      fontSize: 10, fontFace: "Calibri", color: C.accentGreen, bold: true, charSpacing: 3, margin: 0
    });

    // Środek
    s.addShape(pres.shapes.RECTANGLE, {
      x: 4.85, y: 1.3, w: 4.5, h: 5.7,
      fill: { color: C.bgCard }, line: { color: C.border, width: 0.75 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.06 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 4.85, y: 1.3, w: 0.08, h: 5.7,
      fill: { color: C.tealLight }, line: { color: C.tealLight }
    });

    s.addText("Mapy cech & anchory", {
      x: 5.05, y: 1.45, w: 4.2, h: 0.4,
      fontSize: 15, fontFace: "Calibri", color: C.text, bold: true, margin: 0
    });
    s.addText("od dużych do małych skal", {
      x: 5.05, y: 1.78, w: 4.2, h: 0.3,
      fontSize: 10, fontFace: "Calibri", color: C.textMuted, italic: true, margin: 0
    });

    const maps = [
      { name: "Conv4_3 (VGG)",   size: "38 × 38", anch: 4, det: "małe obiekty" },
      { name: "Conv7 (FC→Conv)", size: "19 × 19", anch: 6, det: "średnie" },
      { name: "Conv8_2",         size: "10 × 10", anch: 6, det: "średnie" },
      { name: "Conv9_2",         size: "5 × 5",   anch: 6, det: "duże" },
      { name: "Conv10_2",        size: "3 × 3",   anch: 4, det: "b. duże" },
      { name: "Conv11_2",        size: "1 × 1",   anch: 4, det: "całe sceny" }
    ];

    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.05, y: 2.2, w: 4.2, h: 0.32,
      fill: { color: C.bgSoft }, line: { color: C.border, width: 0.5 }
    });
    s.addText("WARSTWA", {
      x: 5.1, y: 2.2, w: 1.55, h: 0.32,
      fontSize: 9, fontFace: "Calibri", color: C.primary, bold: true, charSpacing: 2, valign: "middle", margin: 0
    });
    s.addText("ROZMIAR", {
      x: 6.65, y: 2.2, w: 1.0, h: 0.32,
      fontSize: 9, fontFace: "Calibri", color: C.primary, bold: true, charSpacing: 2, align: "center", valign: "middle", margin: 0
    });
    s.addText("ANCH", {
      x: 7.65, y: 2.2, w: 0.6, h: 0.32,
      fontSize: 9, fontFace: "Calibri", color: C.primary, bold: true, charSpacing: 2, align: "center", valign: "middle", margin: 0
    });
    s.addText("WYKRYWA", {
      x: 8.25, y: 2.2, w: 1.0, h: 0.32,
      fontSize: 9, fontFace: "Calibri", color: C.primary, bold: true, charSpacing: 2, valign: "middle", margin: 0
    });

    maps.forEach((m, i) => {
      const yy = 2.55 + i * 0.36;
      if (i % 2 === 0) {
        s.addShape(pres.shapes.RECTANGLE, {
          x: 5.05, y: yy, w: 4.2, h: 0.34,
          fill: { color: "F8FAFC" }, line: { color: C.border, width: 0 }
        });
      }
      s.addText(m.name, {
        x: 5.1, y: yy, w: 1.55, h: 0.34,
        fontSize: 10, fontFace: "Consolas", color: C.text, valign: "middle", margin: 0
      });
      s.addText(m.size, {
        x: 6.65, y: yy, w: 1.0, h: 0.34,
        fontSize: 11, fontFace: "Consolas", color: C.primary, bold: true, align: "center", valign: "middle", margin: 0
      });
      s.addText(String(m.anch), {
        x: 7.65, y: yy, w: 0.6, h: 0.34,
        fontSize: 11, fontFace: "Consolas", color: C.accent, bold: true, align: "center", valign: "middle", margin: 0
      });
      s.addText(m.det, {
        x: 8.25, y: yy, w: 1.0, h: 0.34,
        fontSize: 9, fontFace: "Calibri", color: C.textMuted, italic: true, valign: "middle", margin: 0
      });
    });

    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.05, y: 4.85, w: 4.2, h: 0.7,
      fill: { color: C.navy }, line: { color: C.navy }
    });
    s.addText("Łącznie anchorów:", {
      x: 5.2, y: 4.9, w: 2.0, h: 0.6,
      fontSize: 11, fontFace: "Calibri", color: C.bgSoft, valign: "middle", margin: 0
    });
    s.addText("8 732", {
      x: 7.2, y: 4.9, w: 2.0, h: 0.6,
      fontSize: 24, fontFace: "Georgia", color: C.tealLight, bold: true, align: "right", valign: "middle", margin: 0
    });

    s.addText("Stosunki boków:", {
      x: 5.05, y: 5.7, w: 4.2, h: 0.3,
      fontSize: 10, fontFace: "Calibri", color: C.textMuted, bold: true, charSpacing: 3, margin: 0
    });
    s.addText("[1, 2, 0.5]  ·  [1, 2, 3, 0.5, 0.333]  ·  [1, 2, 0.5]", {
      x: 5.05, y: 6.0, w: 4.2, h: 0.35,
      fontSize: 11, fontFace: "Consolas", color: C.text, margin: 0
    });
    s.addText("+ dodatkowy anchor o skali  √(s_k · s_k+1)", {
      x: 5.05, y: 6.35, w: 4.2, h: 0.3,
      fontSize: 10, fontFace: "Calibri", color: C.textMuted, italic: true, margin: 0
    });

    // Prawa
    s.addShape(pres.shapes.RECTANGLE, {
      x: 9.55, y: 1.3, w: 3.25, h: 5.7,
      fill: { color: C.bgCard }, line: { color: C.border, width: 0.75 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.06 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 9.55, y: 1.3, w: 0.08, h: 5.7,
      fill: { color: C.accentGreen }, line: { color: C.accentGreen }
    });
    s.addText("Postępy SSD", {
      x: 9.75, y: 1.45, w: 2.95, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: C.text, bold: true, margin: 0
    });

    const ssdProgress = [
      { d: "Klasa SSD (nn.Module)",         status: "done" },
      { d: "VGG-16 backbone (do conv4_3)",  status: "done" },
      { d: "Conv5 + Conv6 + Conv7",         status: "done" },
      { d: "Extra layers conv8–conv11",     status: "done" },
      { d: "Loc & cls heads (6 map)",       status: "done" },
      { d: "Generator anchorów (8732)",     status: "done" },
      { d: "MultiBox loss",                  status: "todo" },
      { d: "Hard Negative Mining",           status: "todo" },
      { d: "NMS + dekodowanie",              status: "todo" },
      { d: "Pętla treningowa",               status: "todo" }
    ];

    ssdProgress.forEach((p, i) => {
      const yy = 1.95 + i * 0.47;
      const ic = p.status === "done" ? I.check : I.pending;
      s.addImage({ data: ic, x: 9.75, y: yy + 0.04, w: 0.3, h: 0.3 });
      s.addText(p.d, {
        x: 10.15, y: yy, w: 2.55, h: 0.4,
        fontSize: 11, fontFace: "Calibri", color: p.status === "done" ? C.text : C.textMuted,
        valign: "middle", margin: 0
      });
    });
  }

  // ===========================================================
  // SLAJD 8 - METRYKI / EWALUACJA  (bez stopki)
  // ===========================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgLight };
    topBar(s, "04  ·  Ewaluacja modeli");

    s.addText([
      { text: "Ujednolicony pipeline metryk dla wszystkich modeli — ", options: { color: C.textMuted, italic: true } },
      { text: "torchmetrics.MeanAveragePrecision", options: { color: C.primary, italic: true, bold: true } },
      { text: " (format xyxy)", options: { color: C.textMuted, italic: true } }
    ], {
      x: 0.6, y: 1.05, w: 12.2, h: 0.4,
      fontSize: 14, fontFace: "Calibri", margin: 0
    });

    const metrics = [
      { name: "mAP",    sub: "mean Average Precision",      desc: "uśrednione AP po wszystkich klasach i progach IoU od 0.5 do 0.95", color: C.primary },
      { name: "mAP50",  sub: "AP @ IoU = 0.50",             desc: "główna metryka porównawcza — łagodniejszy próg detekcji",        color: C.tealLight },
      { name: "mAP75",  sub: "AP @ IoU = 0.75",             desc: "rygorystyczna miara dokładności lokalizacji bboxów",             color: C.teal },
      { name: "Recall", sub: "mar_100  (max 100 detekcji)", desc: "frakcja prawdziwych obiektów wykrytych przez model",            color: C.accent }
    ];

    const mw = 3.0, mgx = 0.15;
    const mStartX = (13.3 - (4 * mw + 3 * mgx)) / 2;

    metrics.forEach((m, i) => {
      const x = mStartX + i * (mw + mgx);
      s.addShape(pres.shapes.RECTANGLE, {
        x, y: 1.7, w: mw, h: 2.4,
        fill: { color: C.bgCard }, line: { color: C.border, width: 0.75 },
        shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.06 }
      });
      s.addShape(pres.shapes.RECTANGLE, {
        x, y: 1.7, w: mw, h: 0.08,
        fill: { color: m.color }, line: { color: m.color }
      });
      s.addText(m.name, {
        x: x + 0.2, y: 1.9, w: mw - 0.4, h: 0.85,
        fontSize: 36, fontFace: "Georgia", color: m.color, bold: true, valign: "middle", margin: 0
      });
      s.addText(m.sub, {
        x: x + 0.2, y: 2.8, w: mw - 0.4, h: 0.3,
        fontSize: 11, fontFace: "Calibri", color: C.textMuted, italic: true, margin: 0
      });
      s.addShape(pres.shapes.RECTANGLE, {
        x: x + 0.2, y: 3.15, w: 0.5, h: 0.03,
        fill: { color: m.color }, line: { color: m.color }
      });
      s.addText(m.desc, {
        x: x + 0.2, y: 3.25, w: mw - 0.4, h: 0.7,
        fontSize: 10.5, fontFace: "Calibri", color: C.text, margin: 0
      });
    });

    // Box z kodem evaluate()
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 4.3, w: 7.0, h: 2.7,
      fill: { color: C.bgCard }, line: { color: C.border, width: 0.75 },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.06 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 4.3, w: 0.08, h: 2.7,
      fill: { color: C.tealLight }, line: { color: C.tealLight }
    });
    s.addText("Funkcja evaluate() — szkielet", {
      x: 0.8, y: 4.4, w: 6.6, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: C.text, bold: true, margin: 0
    });

    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.8, y: 4.8, w: 6.7, h: 2.05,
      fill: { color: C.navyDark }, line: { color: C.navyDark }
    });
    s.addText([
      { text: "from torchmetrics.detection.mean_ap import MeanAveragePrecision", options: { color: C.tealLight, fontSize: 10, breakLine: true } },
      { text: "", options: { fontSize: 10, breakLine: true } },
      { text: "metric = MeanAveragePrecision(box_format=\"xyxy\")", options: { color: C.bgLight, fontSize: 10, breakLine: true } },
      { text: "with torch.no_grad():", options: { color: C.bgLight, fontSize: 10, breakLine: true } },
      { text: "    for images, targets in val_loader:", options: { color: C.bgLight, fontSize: 10, breakLine: true } },
      { text: "        outputs = model(images)", options: { color: C.bgLight, fontSize: 10, breakLine: true } },
      { text: "        metric.update(preds, gts)", options: { color: C.accent, fontSize: 10, breakLine: true } },
      { text: "results = metric.compute()  # mAP, mAP50, mAP75, recall", options: { color: C.tealLight, fontSize: 10 } }
    ], {
      x: 0.95, y: 4.9, w: 6.4, h: 1.9,
      fontFace: "Consolas", margin: 0, paraSpaceAfter: 1
    });

    s.addShape(pres.shapes.RECTANGLE, {
      x: 7.85, y: 4.3, w: 4.95, h: 2.7,
      fill: { color: C.navy }, line: { color: C.navy }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 7.85, y: 4.3, w: 0.08, h: 2.7,
      fill: { color: C.accent }, line: { color: C.accent }
    });
    s.addImage({ data: I.flask, x: 8.05, y: 4.45, w: 0.45, h: 0.45 });
    s.addText("Co będziemy mierzyć", {
      x: 8.6, y: 4.45, w: 4.1, h: 0.45,
      fontSize: 15, fontFace: "Calibri", color: C.bgLight, bold: true, valign: "middle", margin: 0
    });

    const exp = [
      "Dokładność: mAP, mAP50, mAP75",
      "Szybkość: FPS / czas pojedynczej inferencji",
      "Precision-Recall — krzywa per klasa",
      "Wpływ rozmiaru danych (BCCD vs BDD)",
      "Analiza FP / FN — gdzie modele zawodzą"
    ];
    exp.forEach((e, i) => {
      const yy = 5.1 + i * 0.36;
      s.addShape(pres.shapes.OVAL, {
        x: 8.05, y: yy + 0.08, w: 0.13, h: 0.13,
        fill: { color: C.tealLight }, line: { color: C.tealLight }
      });
      s.addText(e, {
        x: 8.3, y: yy, w: 4.4, h: 0.3,
        fontSize: 11, fontFace: "Calibri", color: C.bgLight, valign: "middle", margin: 0
      });
    });
  }

  // ===========================================================
  // SLAJD 9 - STATUS POSTĘPÓW
  // ===========================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.bgLight };
    topBar(s, "05  ·  Status zadań");

    s.addText("Trzy kolumny — kanban projektu", {
      x: 0.6, y: 1.05, w: 12.2, h: 0.4,
      fontSize: 13, fontFace: "Calibri", color: C.textMuted, italic: true, margin: 0
    });

    const cols = [
      {
        title: "ZROBIONE",
        color: C.accentGreen,
        items: [
          "Wybór tematu i datasetów",
          "Loadery: BCCD, BDD100K, VisDrone",
          "Wspólny interfejs danych",
          "Auto-rozpakowywanie ZIP (Colab)",
          "Augmentacje (Albumentations)",
          "YOLO — integracja z Ultralytics",
          "Faster R-CNN — fine-tuning ResNet-50 + FPN",
          "SSD — backbone VGG16 + extra layers",
          "SSD — generator 8 732 anchorów",
          "Pipeline ewaluacji (torchmetrics mAP)"
        ]
      },
      {
        title: "W TRAKCIE",
        color: C.accent,
        items: [
          "Pełny trening modeli (BCCD)",
          "Walidacja pętli treningowej Faster R-CNN",
          "SSD — implementacja MultiBox loss",
          "Mapowanie klas między datasetami",
          "Testy lokalne datasetów"
        ]
      },
      {
        title: "DO ZROBIENIA",
        color: C.primary,
        items: [
          "Trening modeli na BDD100K i VisDrone",
          "SSD — Hard Negative Mining",
          "SSD — NMS + dekodowanie predykcji",
          "Wizualizacje detekcji (bbox overlays)",
          "Analiza false positives / false negatives",
          "Pomiar FPS / czasu inferencji",
          "Tabela porównawcza 3 modele × 3 datasety",
          "Wnioski końcowe i raport"
        ]
      }
    ];

    const cw = 4.05, gx = 0.25;
    const startX = (13.3 - (3 * cw + 2 * gx)) / 2;
    const startY = 1.55;
    const ch = 5.55;

    cols.forEach((c, i) => {
      const x = startX + i * (cw + gx);

      s.addShape(pres.shapes.RECTANGLE, {
        x, y: startY, w: cw, h: ch,
        fill: { color: C.bgCard }, line: { color: C.border, width: 0.75 },
        shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.06 }
      });
      s.addShape(pres.shapes.RECTANGLE, {
        x, y: startY, w: cw, h: 0.6,
        fill: { color: c.color }, line: { color: c.color }
      });
      s.addText(c.title, {
        x: x + 0.2, y: startY, w: cw - 0.4, h: 0.6,
        fontSize: 14, fontFace: "Calibri", color: "FFFFFF", bold: true, charSpacing: 6,
        valign: "middle", margin: 0
      });
      s.addText(String(c.items.length), {
        x: x + cw - 0.7, y: startY, w: 0.5, h: 0.6,
        fontSize: 18, fontFace: "Georgia", color: "FFFFFF", bold: true, italic: true,
        align: "right", valign: "middle", margin: 0
      });

      c.items.forEach((it, k) => {
        const yy = startY + 0.85 + k * 0.44;
        s.addShape(pres.shapes.OVAL, {
          x: x + 0.25, y: yy + 0.08, w: 0.15, h: 0.15,
          fill: { color: c.color }, line: { color: c.color }
        });
        s.addText(it, {
          x: x + 0.5, y: yy, w: cw - 0.7, h: 0.4,
          fontSize: 11, fontFace: "Calibri", color: C.text, valign: "middle", margin: 0
        });
      });
    });
  }

  // ===========================================================
  // SLAJD 10 - PODSUMOWANIE
  // ===========================================================
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };

    s.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 0, w: 0.15, h: 7.5,
      fill: { color: C.tealLight }, line: { color: C.tealLight }
    });

    s.addText("PODSUMOWANIE", {
      x: 0.6, y: 0.5, w: 10, h: 0.4,
      fontSize: 12, fontFace: "Calibri", color: C.tealLight, bold: true, charSpacing: 8, margin: 0
    });

    s.addText("Gdzie jesteśmy", {
      x: 0.6, y: 0.9, w: 10, h: 0.95,
      fontSize: 56, fontFace: "Georgia", color: C.bgLight, bold: true, margin: 0
    });

    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: 2.0, w: 1.0, h: 0.05,
      fill: { color: C.accent }, line: { color: C.accent }
    });

    const stats = [
      { val: "3",      lbl: "modele",     sub: "YOLO · Faster R-CNN · SSD" },
      { val: "3",      lbl: "datasety",   sub: "BCCD · BDD100K · VisDrone" },
      { val: "8 732",  lbl: "anchory",    sub: "wygenerowane dla SSD" }
    ];

    const sw = 4.0, sgx = 0.15;
    const sStartX = (13.3 - (3 * sw + 2 * sgx)) / 2;

    stats.forEach((st, i) => {
      const x = sStartX + i * (sw + sgx);
      s.addShape(pres.shapes.RECTANGLE, {
        x, y: 2.4, w: sw, h: 1.7,
        fill: { color: C.navyDark }, line: { color: C.tealLight, width: 0.5 }
      });
      s.addText(st.val, {
        x: x + 0.2, y: 2.55, w: sw - 0.4, h: 1.0,
        fontSize: 64, fontFace: "Georgia", color: C.tealLight, bold: true, align: "center", valign: "middle", margin: 0
      });
      s.addText(st.lbl, {
        x: x + 0.2, y: 3.5, w: sw - 0.4, h: 0.3,
        fontSize: 12, fontFace: "Calibri", color: C.bgSoft, bold: true, charSpacing: 6, align: "center", margin: 0
      });
      s.addText(st.sub, {
        x: x + 0.2, y: 3.78, w: sw - 0.4, h: 0.3,
        fontSize: 10, fontFace: "Calibri", color: C.textLight, italic: true, align: "center", margin: 0
      });
    });

    s.addText("NASTĘPNE KROKI", {
      x: 0.6, y: 4.5, w: 10, h: 0.35,
      fontSize: 11, fontFace: "Calibri", color: C.accent, bold: true, charSpacing: 8, margin: 0
    });

    const next = [
      "Dokończenie SSD — MultiBox loss + NMS",
      "Pełen trening 3 modeli na BCCD jako baseline",
      "Skalowanie do BDD100K i VisDrone",
      "Tabela porównawcza  i  wizualizacje detekcji",
      "Analiza błędów (FP / FN) i wnioski końcowe"
    ];

    next.forEach((n, i) => {
      const col = i < 3 ? 0 : 1;
      const row = col === 0 ? i : i - 3;
      const x = col === 0 ? 0.6 : 7.0;
      const yy = 5.0 + row * 0.55;

      s.addShape(pres.shapes.OVAL, {
        x, y: yy + 0.05, w: 0.32, h: 0.32,
        fill: { color: C.accent }, line: { color: C.accent }
      });
      s.addText(String(i + 1), {
        x, y: yy + 0.05, w: 0.32, h: 0.32,
        fontSize: 14, fontFace: "Georgia", color: C.navy, bold: true, align: "center", valign: "middle", margin: 0
      });
      s.addText(n, {
        x: x + 0.5, y: yy, w: 5.8, h: 0.45,
        fontSize: 13, fontFace: "Calibri", color: C.bgLight, valign: "middle", margin: 0
      });
    });
  }

  await pres.writeFile({ fileName: "D:/CLCO/grupa2_detekcja/GRUPA2_Detekcja_Obiektow_Postepy.pptx" });
  console.log("OK -> D:/CLCO/grupa2_detekcja/GRUPA2_Detekcja_Obiektow_Postepy.pptx");
}

build().catch(e => { console.error(e); process.exit(1); });
