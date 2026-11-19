const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const { imageSize } = (() => {
  // minimal PNG size reader
  return {
    imageSize: (p) => {
      const b = fs.readFileSync(p);
      return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
    },
  };
})();

const FIG = path.join(__dirname, "..", "benchmarks", "figures") + "/";
const DATA = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const OUT = process.argv[2] || "deck.pptx";

// ---------------------------------------------------------------- palette
const DARK = "0F3B3A", TEAL = "0F4C4A", GOLD = "D9A21B", GOLD_TXT = "9A6B00", INK = "16302F",
  MUTED = "5F6E6D", BG = "F7F7F4", CARD = "FFFFFF", LINE = "DCE2E1",
  GOOD_BG = "E4F1EC", GOOD_TX = "0F5A44", BAD_BG = "FBEADF", BAD_TX = "8A3B12", NEU_BG = "EAF0EF";
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Mariana Samperio Cuevas, Isaac Yael Cantú Fernández";
pres.title = "Aprendizaje Geométrico No Supervisado Mediante la Función de Christoffel–Darboux — Medio término";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

const W = 13.333, MX = 0.6, CW = W - 2 * MX;

pres.defineSlideMaster({
  title: "TITULO", background: { color: DARK },
  objects: [],
});
pres.defineSlideMaster({
  title: "CONTENIDO", background: { color: BG },
  objects: [
    { text: { text: "Presentación de medio término · PEF Christoffel–Darboux", options: { x: MX, y: 7.02, w: 8, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.55, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, align: "right" },
});
pres.defineSlideMaster({
  title: "RESPALDO", background: { color: BG },
  objects: [
    { text: { text: "Respaldo · Presentación de medio término · PEF Christoffel–Darboux", options: { x: MX, y: 7.02, w: 8, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.55, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, align: "right" },
});

// ---------------------------------------------------------------- helpers
function header(s, kicker, title) {
  s.addText(kicker.toUpperCase(), { x: MX, y: 0.38, w: CW, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: GOLD_TXT, charSpacing: 2, margin: 0, isTextBox: true });
  s.addText(title, { x: MX, y: 0.68, w: CW, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: INK, margin: 0, isTextBox: true, valign: "middle" });
}
function card(s, x, y, w, h, opts = {}) {
  const tone = opts.tone || "card";
  const fill = { card: CARD, good: GOOD_BG, bad: BAD_BG, neu: NEU_BG, dark: DARK }[tone];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: tone === "card" ? LINE : fill, width: 0.75 } });
  const tc = { card: INK, good: GOOD_TX, bad: BAD_TX, neu: INK, dark: "FFFFFF" }[tone];
  const bc = { card: "33413F", good: GOOD_TX, bad: BAD_TX, neu: "33413F", dark: "DCEBE8" }[tone];
  const runs = [];
  if (opts.title) runs.push({ text: opts.title, options: { bold: true, fontSize: opts.titleSize || 16, color: tc, fontFace: BODY, breakLine: true, paraSpaceAfter: 4 } });
  const body = opts.body || [];
  (Array.isArray(body) ? body : [body]).forEach((b, i, arr) => {
    const isObj = typeof b === "object";
    const text = isObj ? b.text : b;
    runs.push({ text, options: { fontSize: opts.size || 14, color: bc, fontFace: BODY, bullet: opts.bullets ? { indent: 14 } : false, breakLine: i < arr.length - 1, paraSpaceAfter: opts.gap ?? 5, bold: isObj && b.bold } });
  });
  if (runs.length) s.addText(runs, { x: x + 0.18, y: y + 0.12, w: w - 0.36, h: h - 0.24, valign: "top", margin: 0, isTextBox: true, fit: "none" });
}
function img(s, file, x, y, opts = {}) {
  const sz = imageSize(FIG + file);
  let w = opts.w, h = opts.h;
  if (w && !h) h = w * sz.h / sz.w;
  if (h && !w) w = h * sz.w / sz.h;
  if (opts.maxH && h > opts.maxH) { h = opts.maxH; w = h * sz.w / sz.h; }
  if (opts.center) x = x + (opts.center - w) / 2;
  s.addImage({ path: FIG + file, x, y, w, h, altText: opts.alt || file });
  return { w, h };
}
function caption(s, text, x, y, w, h = 0.34) {
  s.addText(text, { x, y, w, h, fontFace: BODY, fontSize: 10.5, color: MUTED, margin: 0, isTextBox: true, valign: "top" });
}
function table(s, rows, x, y, w, colW, opts = {}) {
  const fs = opts.fontSize || 12;
  const data = rows.map((r, i) => r.map((c) => {
    const cell = typeof c === "object" ? c : { text: String(c) };
    const o = Object.assign({ fontFace: BODY, fontSize: fs, color: i === 0 ? "FFFFFF" : INK, bold: i === 0, fill: { color: i === 0 ? DARK : (i % 2 ? "FFFFFF" : "F0F3F2") }, valign: "middle", margin: [3, 6, 3, 6] }, cell.options || {});
    return { text: cell.text, options: o };
  }));
  s.addTable(data, { x, y, w, colW, border: { type: "solid", pt: 0.5, color: LINE }, rowH: opts.rowH || 0.32, autoPage: false });
}
const f2 = (v) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(2));
function ariCell(v) {
  const n = Number(v);
  const fill = n >= 0.85 ? "CFE8DD" : n >= 0.5 ? "FFF1C9" : "F7D9CE";
  return { text: f2(v), options: { fill: { color: fill }, align: "center" } };
}


// ---------------------------------------------------------------- extra helpers
function divider(num, title, questions, notes) {
  const s = pres.addSlide({ masterName: "TITULO" });
  s.addText(`BLOQUE ${String(num).padStart(2, "0")}`, { x: MX, y: 1.55, w: 6, h: 0.45, fontFace: BODY, fontSize: 16, bold: true, color: "E3B23C", charSpacing: 4, margin: 0, isTextBox: true });
  s.addText(title, { x: MX, y: 2.05, w: 11.5, h: 1.3, fontFace: HEAD, fontSize: 40, bold: true, color: "FFFFFF", margin: 0, isTextBox: true, valign: "top" });
  s.addText("En este bloque", { x: MX, y: 3.75, w: 6, h: 0.35, fontFace: BODY, fontSize: 14, bold: true, color: "9FC3BD", margin: 0, isTextBox: true });
  s.addText(questions.map((q, i) => ({ text: q, options: { bullet: { indent: 16 }, breakLine: i < questions.length - 1, paraSpaceAfter: 6 } })),
    { x: MX, y: 4.15, w: 11.4, h: 2.4, fontFace: BODY, fontSize: 17, color: "E6F2EF", margin: 0, isTextBox: true, valign: "top" });
  // progress dots
  for (let i = 1; i <= 9; i++) {
    s.addShape(pres.shapes.OVAL, { x: MX + (i - 1) * 0.42, y: 6.85, w: 0.2, h: 0.2, fill: { color: i === num ? "E3B23C" : (i < num ? "5E8C86" : "24504E") }, line: { color: i === num ? "E3B23C" : (i < num ? "5E8C86" : "24504E") } });
  }
  if (notes) s.addNotes(notes);
}
function steps(s, x, y, w, h, arr, opts = {}) {
  const n = arr.length, gap = 0.32, bw = (w - gap * (n - 1)) / n;
  arr.forEach((st, i) => {
    const bx = x + i * (bw + gap);
    const hl = opts.highlight === i;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y, w: bw, h, rectRadius: 0.08, fill: { color: hl ? DARK : CARD }, line: { color: hl ? DARK : LINE, width: 0.75 } });
    s.addShape(pres.shapes.OVAL, { x: bx + 0.16, y: y + 0.16, w: 0.42, h: 0.42, fill: { color: GOLD }, line: { color: GOLD } });
    s.addText(String(i + 1), { x: bx + 0.16, y: y + 0.16, w: 0.42, h: 0.42, fontFace: BODY, fontSize: 14, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(st.t, { x: bx + 0.68, y: y + 0.13, w: bw - 0.8, h: 0.5, fontFace: BODY, fontSize: opts.titleSize || 14, bold: true, color: hl ? "FFFFFF" : INK, margin: 0, isTextBox: true, valign: "middle" });
    const runs = [];
    if (st.f) runs.push({ text: st.f, options: { fontFace: HEAD, fontSize: opts.fSize || 14, color: hl ? "E3B23C" : TEAL, breakLine: true, paraSpaceAfter: 6 } });
    runs.push({ text: st.b, options: { fontFace: BODY, fontSize: opts.size || 12.5, color: hl ? "DCEBE8" : "33413F" } });
    s.addText(runs, { x: bx + 0.18, y: y + 0.72, w: bw - 0.36, h: h - 0.85, margin: 0, isTextBox: true, valign: "top" });
    if (i < n - 1) s.addText("→", { x: bx + bw - 0.02, y: y + h / 2 - 0.25, w: gap + 0.04, h: 0.5, fontSize: 20, color: GOLD, align: "center", margin: 0, isTextBox: true });
  });
}
function numbered(s, x, y, w, items, opts = {}) {
  const lh = opts.lh || 0.62, st = opts.start || 1;
  items.forEach((it, i) => {
    const yy = y + i * lh;
    s.addShape(pres.shapes.OVAL, { x, y: yy + 0.04, w: 0.36, h: 0.36, fill: { color: opts.dark ? "E3B23C" : GOLD }, line: { color: GOLD } });
    s.addText(String(i + st), { x, y: yy + 0.04, w: 0.36, h: 0.36, fontFace: BODY, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(it, { x: x + 0.5, y: yy, w: w - 0.5, h: lh - 0.04, fontFace: BODY, fontSize: opts.size || 13, color: opts.dark ? "E6F2EF" : "33413F", margin: 0, isTextBox: true, valign: "top" });
  });
}
const get3d = (m, v) => DATA.c3d.find((r) => r.method === m && r.variant === v).mean;

// =====================================================================================
// 1. TITLE
{
  const s = pres.addSlide({ masterName: "TITULO" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 0.75, w: 2.65, h: 0.42, rectRadius: 0.2, fill: { color: "E3B23C" }, line: { color: "E3B23C" } });
  s.addText("UDEM · INGENIERÍA", { x: MX, y: 0.75, w: 2.65, h: 0.42, fontFace: BODY, fontSize: 13, bold: true, color: DARK, align: "center", valign: "middle", charSpacing: 2, margin: 0, isTextBox: true });
  s.addText("Aprendizaje Geométrico No Supervisado Mediante la Función de Christoffel–Darboux", { x: MX, y: 1.45, w: 9.2, h: 2.3, fontFace: HEAD, fontSize: 42, bold: true, color: "FFFFFF", margin: 0, isTextBox: true, valign: "top" });
  s.addText("Presentación de medio término del Proyecto de Evaluación Final (PEF)", { x: MX, y: 3.85, w: 9, h: 0.45, fontFace: BODY, fontSize: 18, color: "BFD6D2", margin: 0, isTextBox: true });
  s.addText([
    { text: "Mariana Samperio Cuevas · 639835 · ITC", options: { bold: true, breakLine: true } },
    { text: "Isaac Yael Cantú Fernández · 594461 · IMT", options: { bold: true, breakLine: true } },
    { text: "Asesor: Dr. Lino Gustavo Garza Gaona", options: { color: "BFD6D2" } },
  ], { x: MX, y: 5.55, w: 7, h: 1.1, fontFace: BODY, fontSize: 15, color: "FFFFFF", margin: 0, isTextBox: true, paraSpaceAfter: 4 });
  s.addText("Octubre 2026", { x: 9.8, y: 6.3, w: 2.9, h: 0.35, fontFace: BODY, fontSize: 14, color: "BFD6D2", align: "right", margin: 0, isTextBox: true });
  s.addNotes("Todo el material numérico proviene del repositorio point-field (benchmarks/). El documento explicativo que acompaña esta presentación detalla cada paso.");
}

// 2. AGENDA
{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Contenido", "Agenda — 9 bloques");
  const items = ["Problema, objetivos y marco teórico", "CD en 2D", "Librería de generación (point-field)", "Protocolo del benchmark",
    "Evaluación inicial en 2D", "Mejoras de CD", "Reevaluación en 2D", "Primera implementación en 3D", "Estado y siguientes pasos"];
  items.forEach((t, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = MX + col * 4.1, y = 1.75 + row * 1.6, w = 3.9, h = 1.35;
    const dark = i === 8;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: dark ? DARK : CARD }, line: { color: dark ? DARK : LINE, width: 0.75 } });
    s.addText(String(i + 1).padStart(2, "0"), { x: x + 0.25, y: y + 0.18, w: 1, h: 0.5, fontFace: HEAD, fontSize: 26, bold: true, color: dark ? "E3B23C" : GOLD, margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.25, y: y + 0.68, w: w - 0.45, h: 0.55, fontFace: BODY, fontSize: 16, color: dark ? "FFFFFF" : INK, margin: 0, isTextBox: true, valign: "top" });
  });
  s.addText("Cada bloque responde: qué pregunta, con qué condiciones, qué evidencia y qué interpretación y limitaciones.", { x: MX, y: 6.6, w: CW, h: 0.35, fontFace: BODY, fontSize: 13, italic: true, color: MUTED, margin: 0, isTextBox: true });
}

// =====================================================================================
// BLOQUE 01
divider(1, "Problema, objetivos y marco teórico", [
  "¿Qué queremos representar y agrupar en una nube de puntos?",
  "¿Qué es la función de Christoffel–Darboux y cómo se calcula, paso a paso?",
  "¿Qué se ha hecho antes con ella y qué hueco atiende este PEF?",
]);

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 01 · Problema y objetivos", "¿Qué queremos representar y agrupar?");
  card(s, MX, 1.6, 6.9, 1.55, { title: "Problema", body: "Dada una nube de puntos sin etiquetas, describir su geometría (soporte, huecos, partes) y separar sus regiones con una sola formulación matemática: la función de Christoffel–Darboux, sin entrenamiento y con resultados interpretables.", size: 14 });
  card(s, MX, 3.3, 6.9, 3.55, {
    title: "Objetivos específicos (anteproyecto)", bullets: true, size: 13, gap: 3, body: [
      "Formular e implementar CD; analizar grado y regularización",
      "Generar datos sintéticos progresivos: compacto, no convexo, desconectado; ruido, oclusión y densidad variable",
      "Construir similitud, embeddings y agrupamiento no supervisado con CD",
      "Evaluar con ARI/NMI, trustworthiness/continuity y robustez",
      "Comparar con métodos de representación y agrupamiento de referencia",
      "Aplicar a nubes 3D tipo ShapeNet; documentar ventajas, limitaciones y costo",
    ],
  });
  s.addText("Redacción tomada del reporte consolidado (§16.6); verificar contra el anteproyecto antes de la entrega final.", { x: MX + 0.18, y: 6.48, w: 6.5, h: 0.3, fontFace: BODY, fontSize: 10.5, italic: true, color: MUTED, margin: 0, isTextBox: true });
  card(s, 7.7, 1.6, 5.03, 3.3, { tone: "neu", title: "Lo que recibe el método", body: "" });
  img(s, "fig_nube_ilustrativa.png", 7.85, 2.2, { w: 4.7, alt: "Nube ilustrativa sin etiquetas: dos lunas y dos anillos" });
  caption(s, "Solo coordenadas. ¿Cuántas partes hay y dónde está el soporte?", 7.9, 4.45, 4.7);
  card(s, 7.7, 5.05, 5.03, 1.8, { tone: "bad", title: "Fuera de alcance", body: "Entrenamiento de redes profundas, procesamiento en tiempo real, segmentación semántica y reconstrucción de escenas completas.", size: 13 });
  s.addNotes("Dos preguntas: dónde está el soporte (representación) y cuántas partes hay (agrupamiento). La herramienta es CD porque se calcula con álgebra lineal, es interpretable y tiene teoría detrás.");
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 01 · Marco teórico (1/3)", "De una nube de puntos a P_d(x) en cuatro pasos");
  steps(s, MX, 1.6, CW, 3.05, [
    { t: "La nube como medida", f: "μ_N = (1/N) Σᵢ δ(xᵢ)", b: "Cada punto pesa 1/N. Cualquier promedio sobre la nube es una integral contra esta medida." },
    { t: "Monomios hasta grado d", f: "v_d(x) = (1, x₁, x₂, x₁², x₁x₂, x₂², …)", b: "Todos los productos de coordenadas de grado ≤ d. En 2D hay s(2,d) = C(d+2, 2): 3, 6, 15, 28 para d = 1, 2, 4, 6." },
    { t: "Matriz de momentos", f: "M_d = (1/N) Σᵢ v_d(xᵢ) v_d(xᵢ)ᵀ + εI", b: "Promedios de productos de monomios (medias, varianzas, momentos superiores). εI (Tikhonov, ε = 10⁻⁸) permite invertirla." },
    { t: "Función de Christoffel", f: "P_d(x) = v_d(x)ᵀ M_d⁻¹ v_d(x)", b: "Un número para cada punto del espacio: pequeño donde hay datos, grande lejos de ellos. Λ_d = 1/P_d." },
  ], { highlight: 3, fSize: 13 });
  card(s, MX, 4.85, 6.0, 2.0, { tone: "neu", title: "Por qué funciona (caracterización variacional)", body: "Λ_d(x) = min { ∫p² dμ_N : deg p ≤ d, p(x) = 1 }. Si x está entre los datos, un polinomio que vale 1 en x no puede ser pequeño en los datos: Λ_d grande, P_d pequeño. Si x está lejos, basta uno que valga ≈ 0 en los datos: P_d grande.", size: 12.5 });
  card(s, 6.85, 4.85, 5.88, 2.0, { title: "Núcleo de Christoffel–Darboux", body: ["K_d(x, y) = v_d(x)ᵀ M_d⁻¹ v_d(y) mide qué tan «parecidos» son dos puntos según los momentos de la nube; P_d(x) = K_d(x, x).", "Todo se calcula con una inversión de matriz: sin entrenamiento ni optimización iterativa."], size: 12.5, gap: 4 });
  s.addNotes("Recorrer los cuatro pasos de izquierda a derecha. El cuarto es el objeto central: P_d. La caracterización variacional explica por qué P_d es pequeño en los datos y grande fuera.");
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 01 · Marco teórico (2/3)", "Ejemplo numérico: seis puntos, grado 1");
  img(s, "fig_paso_cd.png", MX, 1.5, { w: CW, alt: "Cálculo paso a paso de P_1 en seis puntos" });
  card(s, MX, 5.6, 3.9, 1.25, { title: "③ La matriz", body: "1 en la esquina; medias (0.03, 0.08); segundos momentos (0.42, 0.21, 0.17).", size: 12.5, titleSize: 14 });
  card(s, MX + 4.1, 5.6, 3.9, 1.25, { tone: "good", title: "④ Típico vs. atípico", body: "(0.1, 0.1), en medio de la nube: P₁ = 1.02. (−0.8, 0.9), fuera de la tendencia: P₁ = 32.4.", size: 12.5, titleSize: 14 });
  card(s, MX + 8.2, 5.6, 3.93, 1.25, { tone: "neu", title: "Lectura con d = 1", body: "P₁(x) − 1 = distancia de Mahalanobis² a la media: curvas de nivel elípticas.", size: 12.5, titleSize: 14 });
  s.addNotes("Las cifras se recalculan con benchmarks/fig_explicativas.py cd. El promedio de P en los datos es exactamente 3 = número de monomios: identidad de traza.");
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 01 · Marco teórico (3/3)", "Tres propiedades que usamos y un compromiso");
  const P = [
    ["Tipicidad", "P_d es pequeño sobre el soporte y crece polinomialmente fuera de él.", "Detectar huecos, bordes y atípicos sin fijar un radio (Bloque 02).", "good"],
    ["Identidad de traza", "(1/N) Σᵢ P_d(xᵢ) = s(n, d) exactamente cuando ε = 0.", "Chequeo del código (error < 10⁻³) y escala común entre modelos (Bloque 05).", "card"],
    ["Densidad (teoría citada)", "s(n, d)·Λ_d(x) tiende a la densidad de los datos cuando d → ∞ (Máté–Nevai–Totik; Lasserre, Pauwels y Putinar, 2022).", "Leer P_d como «inverso de densidad». No se demuestra en este PEF.", "neu"],
  ];
  P.forEach(([t, a, b, tone], i) => card(s, MX + i * 4.1, 1.6, 3.9, 2.9, { tone, title: t, body: [a, { text: "Uso: " + b, bold: false }], size: 13, gap: 8 }));
  card(s, MX, 4.7, CW, 2.15, { tone: "bad", title: "Compromiso del grado d", body: ["Más grado → más monomios → curvas de nivel más flexibles (pueden «ver» huecos y curvas).", "Pero M_d se vuelve casi singular: su número de condición (mayor / menor valor propio) crece órdenes de magnitud. cond = 10⁷ amplifica el redondeo de 10⁻¹⁶ a 10⁻⁹; con 10¹⁷ el resultado deja de ser confiable.", "Por eso d se elige con evidencia (Bloque 02), no a priori."], size: 13, bullets: true, gap: 4 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 01 · Estado del arte (1/3)", "Qué se ha hecho con la función de Christoffel–Darboux");
  const rows = [
    ["Trabajo", "Qué hace, en palabras simples", "Uso en este PEF"],
    ["Pauwels, Putinar y Lasserre (FoCM, 2021)", "La matriz de momentos y su función de Christoffel revelan soporte, densidad y estructura algebraica de una nube, con garantías", "Base de P_d y del diagnóstico de grado"],
    ["Lasserre, Pauwels y Putinar (Cambridge UP, 2022)", "Monografía que reúne teoría, propiedades y aplicaciones del núcleo CD", "Marco teórico y notación"],
    ["Askari et al. (2018)", "Versión con núcleos de la inversa de Christoffel para detectar atípicos en muchas variables", "Lectura «P_d grande = atípico»"],
    ["Beckermann, Putinar, Saff y Stylianopoulos (FoCM, 2021)", "Cómo cambia el núcleo al agregar pocos puntos; relación con el leverage score", "Por qué un atípico tiene P_d enorme"],
    ["Roos Hoefgeest y Slot (2022)", "Homología persistente (componentes y huecos a varias escalas) con el polinomio CD", "Pendiente: contar grupos sin dar k"],
    ["Korda, Putinar y Mezić (ACHA, 2020)", "Núcleo CD para separar el espectro del operador de Koopman desde datos", "Inspiración del promedio sobre rotaciones (3D)"],
    ["J. Approx. Theory 317 (2026); Ðurašinović (2022); CLOE", "Variante para dimensión alta; uso supervisado; CD en el espacio latente de un autoencoder", "Contexto; fuera de alcance"],
  ];
  table(s, rows, MX, 1.6, CW, [3.7, 5.5, 2.93], { fontSize: 12, rowH: 0.6 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 01 · Estado del arte (2/3)", "El hueco que atiende el PEF");
  card(s, MX, 1.6, 5.7, 4.0, { title: "Usos de CD en la literatura revisada", bullets: true, size: 14, gap: 7, body: ["Estimar el soporte de una nube", "Detectar atípicos", "Aproximar la densidad", "Topología (homología persistente)", "Sistemas dinámicos (operador de Koopman)", "Aprendizaje supervisado y detección con autoencoders"] });
  s.addText("→", { x: 6.35, y: 3.2, w: 0.6, h: 0.6, fontSize: 32, color: GOLD, align: "center", margin: 0, isTextBox: true });
  card(s, 7.03, 1.6, 5.7, 4.0, { tone: "dark", title: "Lo que hace este PEF", titleSize: 17, bullets: true, size: 14, gap: 7, body: ["Usar CD para agrupar puntos y construir embeddings, no solo para describir el soporte", "Generar geometrías con referencia conocida y dificultad controlada (point-field)", "Comparar contra K-Means, DBSCAN, OPTICS, Spectral, PCA, Isomap, UMAP… con el mismo protocolo", "Documentar dónde CD gana y dónde pierde"] });
  card(s, MX, 5.8, CW, 1.05, { tone: "neu", body: "En los PDF revisados no encontramos un marco que use CD para agrupar y embeber nubes de puntos evaluado contra los métodos estándar sobre geometrías controladas.", size: 14 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 01 · Estado del arte (3/3)", "Métodos de referencia y métricas del benchmark");
  const cols = [
    ["Agrupamiento", ["K-Means: centro más cercano, fronteras rectas", "DBSCAN (Ester et al., 1996): une puntos densos a distancia ≤ eps; no necesita k", "OPTICS (Ankerst et al., 1999): grupos de varias densidades; no necesita k", "Spectral (Ng, Jordan y Weiss, 2001): grafo de vecinos + vectores propios"]],
    ["Representación", ["PCA / Kernel PCA (Schölkopf, 1998)", "Isomap (Tenenbaum, 2000): distancias a lo largo de la nube", "Laplacian Eigenmaps (Belkin y Niyogi, 2003)", "UMAP (McInnes, 2018): conserva vecindades difusas"]],
    ["Métricas", ["ARI (Hubert y Arabie, 1985)", "NMI (Strehl y Ghosh, 2002)", "Trustworthiness y continuity (Venna y Kaski, 2001)", "Exactitud 1-NN para descriptores 3D"]],
  ];
  cols.forEach(([t, b], i) => card(s, MX + i * 4.1, 1.6, 3.9, 3.4, { title: t, body: b, bullets: true, size: 12.5, gap: 5 }));
  card(s, MX, 5.2, 6.0, 1.65, { title: "3D sin entrenamiento (comparables)", body: "Distribución de formas D2 (Osada et al., 2002) y eigen-rasgos de covarianza (linealidad, planaridad, esfericidad).", size: 13 });
  card(s, 6.85, 5.2, 5.88, 1.65, { tone: "bad", title: "3D con aprendizaje profundo (fuera de alcance)", body: "PointNet, DGCNN, FoldingNet sobre ShapeNet y ModelNet40: requieren entrenamiento; se citan como contexto.", size: 13 });
}

// =====================================================================================
// BLOQUE 02
divider(2, "CD en 2D", [
  "¿Qué geometría describen los conjuntos de nivel de P_d?",
  "¿Cómo cambia esa geometría al subir el grado d?",
  "¿Qué grado usamos y por qué?",
]);

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 02 · CD en 2D", "Procedimiento: del ajuste al soporte estimado");
  numbered(s, MX, 1.65, 5.6, [
    "Ajustar el modelo: construir M_d con los datos e invertirla.",
    "Evaluar P_d en una rejilla de 260 × 260 puntos que cubre la nube.",
    "Pintar log₁₀ P_d (P_d va de 1 a 10⁸): claro = típico, oscuro = atípico.",
    "Línea dorada: P_d = percentil 97 de los datos. Dentro queda el 97 % de la nube: el soporte estimado.",
    "Probar puntos: el centro del anillo (hueco) y puntos lejanos.",
  ], { lh: 0.78, size: 13.5 });
  img(s, "fig_paso_grado.png", 6.4, 1.6, { w: 6.33, alt: "Soporte del anillo con grados 1, 2 y 3" });
  card(s, 6.4, 3.65, 6.33, 3.2, { tone: "neu", title: "Qué muestra la figura (anillo, N = 300)", bullets: true, size: 13, gap: 5, body: ["d = 1: solo elipses; el hueco queda dentro del soporte.", "d = 2: aparecen x², xy, y² y el polinomio puede valer más en el centro que en el aro: el hueco queda fuera.", "d = 3: el aro se ajusta mejor; cond(M_d) sube de 2.2 a 4.8 × 10²."] });
  caption(s, "fig_explicativas.py grado · semilla 0 · ε = 10⁻⁸", MX, 6.55, 5.6);
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 02 · Evidencia", "P_d describe el soporte, y el grado decide cuánto detalle");
  img(s, "fig_cd_2d.png", MX, 1.55, { w: 8.55, alt: "Mapas de log P_d y conjuntos de nivel para anillo y lunas, grados 1, 2, 4 y 6" });
  const deg = DATA.deg.filter((r) => r.dataset.startsWith("anillo"));
  const rows = [["d", "s", "cond(M_d)", "hueco / med."]].concat(
    deg.map((r) => [String(r.degree), String(r.basis_size), Number(r.cond_M).toExponential(1), { text: Number(r.ratio_hueco).toFixed(1) + "×", options: { color: r.hueco_fuera_soporte ? GOOD_TX : BAD_TX, bold: true } }]));
  table(s, rows, 9.35, 1.6, 3.38, [0.42, 0.5, 1.16, 1.3], { fontSize: 12, rowH: 0.38 });
  caption(s, "Verde: el centro del anillo queda fuera del soporte (P_d > percentil 97).", 9.35, 3.6, 3.38, 0.4);
  card(s, 9.35, 4.1, 3.38, 2.75, { tone: "good", title: "Decisión", body: ["d = 2 es el menor grado que ve el hueco.", "d ≥ 4 sigue las lunas, con cond(M_d) hasta 10⁷ y solo 10.7 puntos por parámetro en d = 6.", "Se usa d = 2 en el modelo global y d = 1 en los locales (11 puntos)."], size: 12.5, gap: 4 });
  caption(s, "fig_cd_2d.py · anillo Annulus(0.75, 1.0), N = 300; lunas = no_convexo · semilla 0 · results/cd_2d_degree_diagnostics.csv", MX, 6.25, 8.55, 0.5);
  s.addNotes("Fila de arriba: anillo con tres puntos de prueba (cruces rojas). Fila de abajo: dos lunas. Cada columna es un grado; arriba de cada panel está cond(M_d). Limitación: P_d dice dónde hay datos, no cuántas partes ni cuál es cuál.");
}

// =====================================================================================
// BLOQUE 03
divider(3, "Librería de generación (point-field)", [
  "¿Por qué generar datos sintéticos con referencia conocida?",
  "¿Qué geometrías y escenarios produce la librería, con qué parámetros exactos?",
  "¿Cómo actúa cada perturbación sobre los puntos?",
]);

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 03 · Librería de generación", "point-field: geometrías con referencia conocida");
  img(s, "fig_primitivas.png", MX, 1.55, { w: 7.7, alt: "Primitivas 2D y superficies 3D de point-field" });
  card(s, MX, 5.6, 7.7, 1.25, { tone: "neu", title: "Por qué sintéticos", body: "Para saber si un método acierta hay que conocer la respuesta; aquí se conoce siempre, y la dificultad se sube de forma controlada.", size: 13 });
  card(s, 8.55, 1.6, 4.18, 5.25, { tone: "dark", title: "Uso en 4 líneas", titleSize: 15, body: "" });
  s.addText(
    'from pointfield.datasets import make\nfrom pointfield.perturbations import add_occlusion\n\nds = make("entrelazado", seed=100)\nds = add_occlusion(ds, 0.25, seed=1)\nds = ds.normalized()\n\nds.X        # puntos\nds.y        # referencia\nds.latent   # coordenadas 2D reales\nds.meta     # geometrías, poses, semillas',
    { x: 8.75, y: 2.15, w: 3.85, h: 3.4, fontFace: "Courier New", fontSize: 11, color: "E6F2EF", margin: 0, isTextBox: true, valign: "top" });
  s.addText("Muestreo uniforme en longitud de arco o área · 77 pruebas automáticas · Disk y Rectangle corregidos (eran copias de Circle)", { x: 8.75, y: 5.65, w: 3.85, h: 1.05, fontFace: BODY, fontSize: 11.5, color: "9FC3BD", margin: 0, isTextBox: true });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 03 · Escenarios", "Cuatro dificultades y dos casos reservados");
  const rows = [
    ["Escenario", "Construcción (150 puntos por grupo)", "Dificultad que representa"],
    ["compacto", "3 gaussianas: desv. (0.55, 0.30) rot. 0.5 en (0, 0); (0.35, 0.35) en (1.9, 0.6); (0.45, 0.25) rot. −0.9 en (0.9, −1.5)", "Grupos compactos que casi se tocan"],
    ["no_convexo", "2 Crescent (radio 1, grosor 0.25, medio círculo); el segundo girado π y movido a (1.0, 0.45)", "Formas no convexas entrelazadas («dos lunas»)"],
    ["entrelazado", "2 Spiral arquimedianas r = 0.5 + 0.5t, 1.5 vueltas, fases 0 y π, grosor 0.08", "Brazos enroscados uno dentro del otro"],
    ["anidado", "Annulus(0.30, 0.55) dentro de Annulus(1.25, 1.50)", "Desconectados, no separables con una recta"],
    [{ text: "espiral_3brazos *", options: { bold: true } }, "3 espirales arquimedianas, 1.2 vueltas, grosor 0.06", "Más brazos que en desarrollo"],
    [{ text: "espiral_log *", options: { bold: true } }, "2 espirales logarítmicas r = 0.4·e^(0.22t), 2 vueltas, grosor 0.05", "Otra ley radial"],
  ];
  table(s, rows, MX, 1.6, CW, [2.2, 6.6, 3.33], { fontSize: 12.5, rowH: 0.62 });
  card(s, MX, 6.05, CW, 0.8, { tone: "neu", body: "* Reservados: no se usaron al desarrollar las mejoras; sirven para comprobar que generalizan. Parámetros guardados en results/escenarios.json.", size: 13 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 03 · Conjuntos controlados", "Escenarios, perturbaciones y variación de parámetros");
  img(s, "fig_libreria.png", MX, 1.5, { w: 9.5, alt: "Escenarios del benchmark, perturbaciones y variación de parámetros" });
  card(s, 10.3, 1.6, 2.43, 1.7, { title: "Fila 1", body: "Los 6 escenarios; color = grupo de referencia", size: 12.5 });
  card(s, 10.3, 3.45, 2.43, 1.7, { tone: "neu", title: "Fila 2", body: "Las 3 perturbaciones sobre espirales y lunas", size: 12.5 });
  card(s, 10.3, 5.3, 2.43, 1.55, { title: "Fila 3", body: "Cambiar un parámetro cambia la dificultad", size: 12.5 });
  caption(s, "fig_libreria.py · 150 puntos por grupo · semilla 0", MX, 6.55, 9.5);
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 03 · Perturbaciones", "Cómo actúa cada perturbación sobre los puntos");
  img(s, "fig_paso_perturbaciones.png", MX, 1.5, { w: CW, alt: "Ruido, oclusión y densidad variable sobre las dos lunas" });
  card(s, MX, 5.45, 3.9, 1.4, { title: "Ruido (σ = 0.06)", body: "Desplaza cada punto con un vector normal de desviación 0.06·RMS: error de medición.", size: 12.5, titleSize: 14 });
  card(s, MX + 4.1, 5.45, 3.9, 1.4, { title: "Oclusión (25 %)", body: "Quita el 25 % con mayor proyección sobre una dirección u: un lado que el sensor no ve.", size: 12.5, titleSize: 14 });
  card(s, MX + 8.2, 5.45, 3.93, 1.4, { title: "Densidad variable (0.15)", body: "Conserva cada punto con p = 1 − 0.85·t a lo largo de u: barrido que pierde resolución.", size: 12.5, titleSize: 14 });
  s.addNotes("Las escalas son relativas al radio cuadrático medio (RMS) para que el mismo parámetro signifique lo mismo en todos los escenarios. Cada perturbación queda registrada en meta['perturbations'] con su dirección y semilla (semilla del caso + 1000). Normalización: centro + un solo factor; no StandardScaler porque deforma los ángulos.");
}

// =====================================================================================
// BLOQUE 04
divider(4, "Protocolo del benchmark", [
  "¿Qué pasos sigue cada experimento, de la generación a la calificación?",
  "¿Qué recibe cada método y cómo se fijan sus hiperparámetros sin usar etiquetas?",
  "¿Qué miden ARI, NMI y trustworthiness, y por qué se separan semillas de desarrollo y evaluación?",
]);

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 04 · Protocolo", "El recorrido de un experimento");
  steps(s, MX, 1.6, CW, 2.45, [
    { t: "Generar", b: "make(escenario, seed) → puntos X y referencia y" },
    { t: "Perturbar", b: "ruido, oclusión o densidad (semilla + 1000)" },
    { t: "Normalizar", b: "centrar y dividir por el RMS (un solo factor)" },
    { t: "Ajustar", b: "el método recibe solo X (y k si le toca)" },
    { t: "Calificar", b: "comparar con y: ARI, NMI, tiempo" },
    { t: "Guardar", b: "una fila en results/clustering_2d_eval.csv" },
  ], { titleSize: 13.5, size: 12 });
  card(s, MX, 4.3, 3.9, 2.55, { title: "Tamaño", body: ["6 escenarios × 4 variantes = 24 casos", "× 5 semillas = 120 nubes", "× 7 métodos = 840 corridas"], size: 13.5, gap: 6 });
  card(s, MX + 4.1, 4.3, 3.9, 2.55, { tone: "good", title: "Igualdad de condiciones", body: "Todos los métodos reciben exactamente la misma realización de cada nube, ya normalizada.", size: 13.5 });
  card(s, MX + 8.2, 4.3, 3.93, 2.55, { tone: "bad", title: "Etiquetas", body: "La referencia y solo entra en el paso 5. Excepción declarada: el techo oráculo del respaldo.", size: 13.5 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 04 · Casos", "Los 24 casos: cada uno es una fila del mapa de calor");
  img(s, "fig_casos_benchmark.png", MX, 1.45, { h: 5.45, alt: "Miniaturas de los 24 casos del benchmark" });
  card(s, 8.85, 1.6, 3.88, 5.25, { tone: "neu", title: "Cómo leerla", bullets: true, size: 13, gap: 6, body: ["Columnas: los 6 escenarios (* = reservado).", "Filas: base y las 3 perturbaciones.", "Color: grupo de referencia (el método no lo ve).", "N: puntos que quedan tras la perturbación (oclusión quita 25 %, densidad variable ≈ 40 %).", "Semilla de evaluación 100; las otras 4 semillas cambian la muestra y la dirección de la perturbación."] });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 04 · Diseño experimental", "Qué recibe cada método y con qué configuración");
  const rows = [
    ["Método", "¿Recibe k?", "Configuración (fijada antes de ver resultados)"],
    ["K-Means", "Sí", "10 inicializaciones"],
    ["DBSCAN", "No", "min_samples = 5; eps = rodilla de la curva k-distancia (sin etiquetas)"],
    ["OPTICS", "No", "min_samples = 7, xi = 0.05, min_cluster_size = 0.1 (comparación de scikit-learn)"],
    ["Spectral (kNN)", "Sí", "grafo de 10 vecinos"],
    ["CD global (inicial)", "Sí", "grado 2"],
    ["CD local-espectral (mejora 1)", "Sí", "10 vecinos, grado local 1"],
    ["CD + fase de Szegő (mejora 2)", "Sí", "igual que mejora 1 + compuerta 0.20 (fijada en desarrollo)"],
  ];
  table(s, rows, MX, 1.6, 8.2, [2.75, 1.05, 4.4], { fontSize: 12.5, rowH: 0.5 });
  card(s, 9.0, 1.6, 3.73, 2.4, { tone: "bad", title: "Ventaja declarada", body: "Recibir k es una ayuda que DBSCAN y OPTICS no tienen; se indica en todas las tablas en vez de ocultarla.", size: 13 });
  card(s, 9.0, 4.15, 3.73, 2.7, { title: "Embeddings", body: ["PCA, Kernel PCA (RBF), Isomap, Laplacian Eigenmaps, UMAP y CD local-espectral.", "Todos con k = 10 vecinos donde aplica; 2 dimensiones de salida."], size: 13, gap: 5 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 04 · Hiperparámetros sin etiquetas", "Rodilla de DBSCAN, semillas separadas y techo oráculo");
  img(s, "fig_paso_knee.png", MX, 1.5, { w: 7.4, alt: "Curva k-distancia y DBSCAN resultante" });
  caption(s, "① Distancia de cada punto a su 5.º vecino, ordenada. La rodilla (más lejos de la cuerda) da eps = 0.170. ② DBSCAN con ese eps en anidado: el anillo exterior, menos denso, se parte (ARI 0.55).", MX, 4.65, 7.4, 0.7);
  card(s, MX, 5.45, 7.4, 1.4, { tone: "neu", title: "Regla de Ester et al. (1996)", body: "La curva sube despacio en zonas densas y se dispara en puntos aislados; el cambio de pendiente separa «denso» de «ruido».", size: 12.5 });
  card(s, 8.2, 1.6, 4.53, 2.55, { title: "Desarrollo vs. evaluación", bullets: true, size: 12.5, gap: 4, body: ["Desarrollo: semillas 0–2. Solo aquí se miró el resultado para fijar la compuerta de Szegő.", "Evaluación: semillas 100–104. Todo lo reportado sale de aquí (media ± desv.).", "Escenarios reservados nunca usados en desarrollo."] });
  card(s, 8.2, 4.3, 4.53, 2.55, { tone: "bad", title: "Techo oráculo (solo respaldo)", body: "Se prueba una rejilla de hiperparámetros y se elige el mejor ARI usando las etiquetas. Es una cota de potencial, no un resultado del método.", size: 12.5 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 04 · Métricas de agrupamiento", "ARI: ¿la partición coincide con la referencia?");
  img(s, "fig_paso_metricas.png", MX, 1.5, { w: CW, alt: "ARI de cinco particiones de las mismas lunas" });
  card(s, MX, 4.05, 6.0, 2.8, { title: "Cómo se calcula", bullets: true, size: 13, gap: 5, body: ["Se toman todos los pares de puntos.", "Un par «coincide» si ambas particiones lo juntan o ambas lo separan.", "Se corrige por lo que se esperaría al azar.", "1 = idénticas (sin importar el nombre de los grupos); ≈ 0 = como al azar; puede ser negativo."] });
  card(s, 6.85, 4.05, 5.88, 2.8, { tone: "neu", title: "NMI (se reporta junto al ARI)", body: ["Información mutua normalizada: 1 = conocer una partición determina la otra; 0 = independientes.", "Las conclusiones no cambian con NMI; está en los CSV."], size: 13, gap: 5 });
  s.addNotes("Mostrar que intercambiar colores da 1.00, que K-Means con un corte recto da 0.30 y que etiquetas al azar dan ≈ 0.");
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 04 · Métricas de embeddings", "Trustworthiness: ¿se conservan los vecinos de cada punto?");
  img(s, "fig_paso_trust.png", MX, 1.5, { w: 8.7, alt: "Intrusos en la vecindad de un punto: Isomap vs PCA" });
  card(s, 9.5, 1.6, 3.23, 2.6, { title: "Trustworthiness T(k)", body: "Para cada punto, sus k = 10 vecinos en el embedding: ¿eran vecinos en la referencia? Cada intruso resta según qué tan lejos estaba.", size: 12.5 });
  card(s, 9.5, 4.35, 3.23, 1.4, { tone: "neu", title: "Continuity C(k)", body: "Lo mismo al revés: vecinos reales que se perdieron.", size: 12.5 });
  card(s, 9.5, 5.9, 3.23, 0.95, { tone: "good", body: "Ambas valen 1 si se conservan todas las vecindades.", size: 12.5 });
  caption(s, "Ejemplo: Isomap conserva 9 de 10 vecinos del punto marcado (T global 0.98); PCA solo 2 (T 0.88).", MX, 6.25, 8.7, 0.5);
}

// =====================================================================================
// BLOQUE 05
divider(5, "Evaluación inicial en 2D", [
  "¿Cómo funciona la primera versión de agrupamiento con CD («CD global»)?",
  "¿Cómo se compara con K-Means, DBSCAN, OPTICS y Spectral en los cuatro escenarios?",
  "¿Cómo se diseñó el experimento de embeddings y qué resultó?",
]);

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 05 · CD inicial", "CD global: un modelo CD por grupo, paso a paso");
  steps(s, MX, 1.6, CW, 2.6, [
    { t: "Arrancar", b: "Partir la nube con K-Means en k grupos." },
    { t: "Ajustar", b: "Un modelo CD de grado 2 con los puntos de cada grupo." },
    { t: "Reasignar", b: "Cada punto va al grupo cuyo modelo lo hace más típico (menor log P_d)." },
    { t: "Repetir", b: "2 y 3 hasta que nada cambie (máx. 30 vueltas)." },
  ]);
  card(s, MX, 4.45, 6.0, 2.4, { tone: "neu", title: "Por qué se pueden comparar los P_d", body: "El promedio de P_d en los datos de cada grupo es siempre s(n, d) (identidad de traza): todos los modelos están en la misma escala.", size: 13.5 });
  card(s, 6.85, 4.45, 5.88, 2.4, { tone: "bad", title: "Qué se espera", body: "Cada modelo es una región cerrada (como una elipse deformada). Funciona si hay un hueco de densidad entre grupos; si un grupo envuelve a otro, ninguna región cerrada puede separarlos.", size: 13.5 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 05 · Evaluación inicial", "CD inicial se comporta como K-Means");
  img(s, "fig_grid_inicial.png", MX, 1.5, { h: 5.4, alt: "Particiones de cinco métodos en cuatro casos base" });
  const pm = (m) => DATA.por_metodo[m]["principal|base"];
  card(s, 7.55, 1.6, 5.18, 1.3, { tone: "neu", title: "Cómo leerla", body: "Filas: escenarios. Columnas: métodos. Color: grupo asignado (gris = ruido). Abajo a la derecha: ARI.", size: 12.5 });
  card(s, 7.55, 3.05, 5.18, 1.35, { tone: "good", title: "Favorable", body: "Compacto: CD global 0.96 ± 0.02, el mejor de la tabla (K-Means 0.93, Spectral 0.95).", size: 13 });
  card(s, 7.55, 4.55, 5.18, 1.95, { tone: "bad", title: "Desfavorable", body: [`Lunas 0.33, espirales 0.06, anillos 0.00: igual que K-Means. Media ${f2(pm("CD global (inicial)"))} vs K-Means ${f2(pm("K-Means"))} y Spectral ${f2(pm("Spectral (kNN)"))}.`, "Motiva la mejora 1 (Bloque 06)."], size: 13 });
  caption(s, "benchmark_clustering_2d.py · figura: semilla 100; cifras: media de 100–104", 7.55, 6.6, 5.18);
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 05 · Embeddings: diseño", "Enrollar en 3D para que la prueba no sea trivial");
  img(s, "fig_paso_enrollado.png", MX, 1.5, { w: CW, alt: "Escenario 2D, enrollado en 3D y desenrollado por Isomap" });
  card(s, MX, 5.75, 3.9, 1.1, { tone: "bad", title: "Problema", body: "Un embedding 2D → 2D es casi trivial: PCA ya da T ≈ 1.", size: 12.5, titleSize: 14 });
  card(s, MX + 4.1, 5.75, 3.9, 1.1, { title: "Solución", body: "Enrollar (u, v) → (t cos t, v, t sin t) y pedir que se desenrolle.", size: 12.5, titleSize: 14 });
  card(s, MX + 8.2, 5.75, 3.93, 1.1, { tone: "good", title: "Ventaja", body: "Las coordenadas 2D originales son la respuesta correcta.", size: 12.5, titleSize: 14 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 05 · Embeddings: resultados", "El embedding CD iguala a Laplacian Eigenmaps; UMAP lidera");
  img(s, "fig_grid_embeddings.png", MX, 1.5, { w: 8.35, alt: "Embeddings 3D a 2D de escenarios enrollados" });
  const order = ["PCA", "Kernel PCA (RBF)", "Isomap", "Laplacian Eigenmaps", "UMAP", "CD local-espectral"];
  const rows = [["Método", "T", "C", "5-NN"]].concat(order.map((m) => [{ text: m, options: { bold: m.startsWith("CD") } }, f2(DATA.emb[m].T), f2(DATA.emb[m].C), f2(DATA.emb[m].kNN_acc)]));
  table(s, rows, 9.15, 1.6, 3.58, [1.88, 0.56, 0.56, 0.58], { fontSize: 11.5, rowH: 0.36 });
  card(s, 9.15, 4.3, 3.58, 2.55, { title: "Cómo leer la figura", body: ["Filas: escenarios. 1.ª columna: nube 3D que recibe el método.", "Demás: salida 2D de cada método, coloreada con la referencia. T abajo a la derecha.", "Un buen embedding «desenrolla»: colores separados, forma reconocible."], size: 12, gap: 4 });
  caption(s, "benchmark_embeddings_2d.py · media de 4 escenarios × 5 semillas · k = 10 · 5-NN = exactitud de los grupos", MX, 6.55, 8.35);
  s.addNotes("CD local-espectral se construye igual que Laplacian Eigenmaps (vectores propios de un grafo), con otros pesos de arista; por eso empatan. No se reclama ventaja del embedding en este corte.");
}

// =====================================================================================
// BLOQUE 06
divider(6, "Mejoras de CD", [
  "¿Qué limitación medida motivó cada mejora?",
  "Mejora 1: ¿cómo se construye un grafo con modelos CD locales?",
  "Mejora 2: ¿qué es la fase de Szegő, cómo se busca sin etiquetas y cuándo se activa?",
]);

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 06 · Mejoras de CD", "Cada mejora responde a una falla medida");
  const stages = [
    ["CD global (inicial)", "Un modelo CD por grupo; reasignar por menor log P_d", "0.06"],
    ["Mejora 1 · CD local-espectral", "Un modelo CD por vecindario (k = 10) define el grafo; agrupamiento espectral", "0.17"],
    ["Mejora 2 · fase de Szegő", "Añadir (cos ψ, sin ψ), ψ = θ − c·g(r), solo si la compuerta lo aprueba", "1.00"],
  ];
  stages.forEach(([t, b, a], i) => {
    const x = MX + i * 4.15, w = 3.85;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w, h: 2.0, rectRadius: 0.08, fill: { color: i === 2 ? DARK : CARD }, line: { color: i === 2 ? DARK : LINE } });
    s.addText(t, { x: x + 0.2, y: 1.72, w: w - 0.4, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: i === 2 ? "FFFFFF" : INK, margin: 0, isTextBox: true });
    s.addText(b, { x: x + 0.2, y: 2.12, w: w - 0.4, h: 0.85, fontFace: BODY, fontSize: 12.5, color: i === 2 ? "DCEBE8" : "33413F", margin: 0, isTextBox: true, valign: "top" });
    s.addText([{ text: "ARI entrelazado  ", options: { fontSize: 11, color: i === 2 ? "9FC3BD" : MUTED } }, { text: a, options: { fontSize: 22, bold: true, color: i === 2 ? "E3B23C" : (i === 0 ? BAD_TX : GOLD_TXT) } }], { x: x + 0.2, y: 3.0, w: w - 0.4, h: 0.45, fontFace: BODY, margin: 0, isTextBox: true });
    if (i < 2) s.addText("→", { x: x + w - 0.02, y: 2.3, w: 0.32, h: 0.5, fontSize: 24, color: GOLD, align: "center", margin: 0, isTextBox: true });
  });
  card(s, MX, 3.85, 6.0, 3.0, { tone: "bad", title: "Limitaciones que las motivaron", body: ["Inicial: un polinomio cerrado por grupo no sigue curvas (lunas 0.33, anillos 0.00).", "Mejora 1: resuelve lunas y anillos (1.00), pero en espirales los vecindarios cruzan de brazo cerca del centro (0.17).", "Una espiral no tiene picos de densidad: es una cresta continua."], size: 13, bullets: true });
  card(s, 6.85, 3.85, 5.88, 3.0, { title: "Idea de cada mejora", body: ["Mejora 1: muchos modelos pequeños en lugar de uno por grupo; mirada de cerca, una curva parece recta.", "Mejora 2: describir cada punto por un ángulo «corregido» que es constante dentro de cada brazo."], size: 13, bullets: true, gap: 6 });
  caption(s, "ARI = media de las semillas de evaluación 100–104 en entrelazado base", MX, 6.62, 6);
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 06 · Mejora 1", "Afinidad CD local: un modelo por vecindario");
  img(s, "fig_paso_afinidad.png", MX, 1.5, { w: CW, alt: "Afinidad CD local en un punto de la espiral" });
  numbered(s, MX, 5.6, 6.2, [
    "Vecinos: los k = 10 más cercanos de cada punto xᵢ (11 con él).",
    "Modelo CD de grado 1 solo con esos 11: P_i(x_j) − 1 = Mahalanobis local.",
  ], { lh: 0.6, size: 12.5 });
  numbered(s, 6.85, 5.6, 5.88, [
    "Peso: z = (P_i(x_j) − mediana)/MAD; A_ij = exp(−z²/2); A = max(A, Aᵀ).",
    "Espectral: vectores propios del Laplaciano del grafo A, filas normalizadas, K-Means.",
  ], { lh: 0.6, size: 12.5, start: 3 });
  s.addNotes("Diferencia con Spectral estándar: allí cada vecino pesa 1; aquí el peso depende de qué tan típico es para el modelo local. El desempeño final es casi igual (0.78 vs 0.79). Los números 3 y 4 de la derecha continúan la secuencia 1–2 de la izquierda.");
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 06 · Mejora 2", "¿Qué es la fase de Szegő y por qué ayuda en una espiral?");
  card(s, MX, 1.6, 3.9, 5.25, { tone: "dark", title: "Szegő en una lámina", titleSize: 16, size: 13, gap: 7, body: ["Gábor Szegő desarrolló los polinomios ortogonales sobre la circunferencia unitaria.", "Ahí los «monomios» no son xᵏ sino e^(ikθ) = cos kθ + i·sin kθ, y existe un núcleo CD para medidas en el círculo.", "Tomamos esa idea: describir un punto por su ángulo sobre un círculo, con (cos ψ, sin ψ), los monomios trigonométricos de grado 1.", "Es una inspiración, no una implementación completa de la teoría."] });
  card(s, MX + 4.1, 1.6, 3.9, 5.25, { title: "El devanado de una espiral", size: 13, gap: 7, body: ["Espiral arquimediana: r = a + b·t con ángulo θ = t + fase.", "Entonces θ ≈ r/b + fase: en coordenadas polares cada brazo es una recta inclinada.", "Los brazos están pegados en el plano, pero cada uno tiene su propia fase.", "Si se resta c·r con c = 1/b, la inclinación desaparece."] });
  card(s, MX + 8.2, 1.6, 3.93, 5.25, { tone: "good", title: "La fase corregida", size: 13, gap: 7, body: ["ψ = θ − c·g(r), con g(r) = r (arquimediana) o log r (logarítmica).", "ψ es casi constante dentro de cada brazo y distinta entre brazos.", "(cos ψ, sin ψ) convierte cada brazo en un arco del círculo: separables aunque se toquen en (x, y).", "c y g se buscan sin etiquetas (siguiente lámina)."] });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 06 · Mejora 2", "La fase paso a paso y cómo se busca sin etiquetas");
  img(s, "fig_paso_szego.png", MX, 1.5, { w: CW, alt: "Fase de Szegő paso a paso sobre la espiral" });
  numbered(s, MX, 5.05, 6.2, [
    "θ y r de cada punto respecto al centro de la nube.",
    "Probar 121 «vueltas» en [−3, 3] (+ 11 de refinamiento) para g = r y g = log r.",
    "Para cada una: K-Means sobre (cos ψ, sin ψ) y su silhouette; elegir el mayor.",
  ], { lh: 0.6, size: 12.5 });
  card(s, 6.85, 5.05, 5.88, 1.8, { tone: "neu", title: "Ejemplo (entrelazado, semilla 100)", body: "Encuentra 1.47 vueltas (la real: 1.5), c = 7.42. Silhouette 0.76 contra 0.47 del ángulo sin torsión.", size: 13 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 06 · Mejora 2", "La compuerta: activar la fase solo si hay devanado");
  card(s, MX, 1.6, 6.0, 2.1, { tone: "dark", title: "Regla", titleSize: 16, size: 14, body: ["ganancia = silhouette(ψ óptimo) − silhouette(θ sin torsión)", "Si ganancia ≥ 0.20: agregar (cos ψ, sin ψ) a (x, y) y aplicar la mejora 1 sobre 4 columnas. Si no: mejora 1 sin cambios."], gap: 6 });
  const rows = [["Escenario (desarrollo, semillas 0–2)", "Ganancia", "¿Se activa?"],
    ["entrelazado base", "0.36–0.44", { text: "Sí", options: { color: GOOD_TX, bold: true } }],
    ["entrelazado + ruido", "0.22–0.27", { text: "Sí", options: { color: GOOD_TX, bold: true } }],
    ["no_convexo base / + ruido", "0.19–0.24", { text: "En el borde", options: { color: GOLD_TXT, bold: true } }],
    ["anidado (todas las variantes)", "≤ 0.13", { text: "No", options: { color: BAD_TX, bold: true } }],
    ["compacto (todas)", "≈ 0.01", { text: "No", options: { color: BAD_TX, bold: true } }],
    ["cualquiera + oclusión o densidad", "≤ 0.14", { text: "No", options: { color: BAD_TX, bold: true } }]];
  table(s, rows, 6.85, 1.6, 5.88, [3.2, 1.3, 1.38], { fontSize: 12.5, rowH: 0.5 });
  card(s, MX, 3.9, 6.0, 2.95, { title: "Por qué 0.20", bullets: true, size: 13, gap: 5, body: ["Se eligió mirando solo las semillas de desarrollo 0–2.", "Separa espirales (≥ 0.22) de compacto y anidado (≤ 0.13).", "Las lunas quedan en el borde: la fase se activa sin dañar el caso limpio (ARI 1.00).", "En anidado, ninguna torsión mejora el ángulo simple (ganancia 0.02 en la semilla 100)."] });
  card(s, 6.85, 5.35, 5.88, 1.5, { tone: "bad", title: "Aviso", body: "Con oclusión o densidad el centro de la nube deja de ser el centro de la espiral: la ganancia cae y la compuerta se cierra (Bloque 07).", size: 12.5 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 06 · Antes y después", "Las tres etapas de CD sobre los mismos datos");
  img(s, "fig_mejoras.png", MX, 1.5, { w: 11.0, center: CW, alt: "Comparación de las tres etapas de CD en entrelazado y anidado" });
  card(s, MX, 6.15, CW, 0.75, { tone: "neu", body: "Columnas 1–3: partición de cada etapa con su ARI. Columna 4: silhouette contra vueltas probadas (punteada = ángulo sin torsión). Columna 5: puntos en (cos ψ, sin ψ) de la mejor fase.", size: 12.5 });
}

// =====================================================================================
// BLOQUE 07
divider(7, "Reevaluación en 2D", [
  "Mismo benchmark, mismas semillas: ¿cuánto mejoran las etapas de CD frente a las referencias?",
  "¿Qué representa cada fila del mapa de calor y qué se lee en ella?",
  "¿Dónde gana CD mejorado y dónde pierde?",
]);

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 07 · Mapa de calor (1/2)", "Escenarios principales: cada fila con su caso");
  img(s, "fig_heatmap_miniaturas_principales.png", MX, 1.45, { h: 5.5, alt: "ARI por caso con miniatura de cada fila, escenarios principales" });
  card(s, 6.0, 1.6, 6.73, 2.0, { tone: "neu", title: "Cómo leerlo", bullets: true, size: 12.5, gap: 3, body: ["Fila = caso (miniatura de la semilla 100); columna = método.", "Celda = ARI medio de 5 semillas: verde ≥ 0.85, amarillo ≈ 0.5, rojo ≈ 0.", "Izquierda: referencias; derecha: tres etapas de CD."] });
  card(s, 6.0, 3.75, 6.73, 3.1, { title: "Lectura fila por fila", bullets: true, size: 12.5, gap: 4, body: ["compacto: todos los métodos con k ≥ 0.88; la compuerta nunca se activa.", "no_convexo: mejora 1 = 1.00; con ruido la fase se activa y cuesta 0.06 (0.94).", "entrelazado: mejora 2 = 1.00 en base, el resto ≤ 0.22; con ruido 0.42; con oclusión o densidad, la compuerta se cierra.", "anidado: mejoras 1 y 2 empatan con Spectral (1.00)."] });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 07 · Mapa de calor (2/2)", "Escenarios reservados: no vistos al desarrollar las mejoras");
  img(s, "fig_heatmap_miniaturas_reservados.png", MX, 1.5, { w: 7.7, alt: "ARI por caso con miniatura, escenarios reservados" });
  card(s, 8.55, 1.6, 4.18, 2.5, { tone: "good", title: "Generaliza en base", body: "3 brazos: 1.00. Logarítmica: 0.92 (la búsqueda elige g = log r). Mejora 1 y Spectral ≤ 0.55.", size: 13 });
  card(s, 8.55, 4.25, 4.18, 2.6, { tone: "bad", title: "No generaliza con perturbaciones", body: "3 brazos perturbados ≤ 0.26; logarítmica + ruido 0.62, con oclusión o densidad ≤ 0.28.", size: 13 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 07 · Resumen por método", "CD mejorado lidera en base; con perturbaciones casi empata");
  const methods = ["K-Means", "DBSCAN", "OPTICS", "Spectral (kNN)", "CD global (inicial)", "CD local-espectral (mejora 1)", "CD + fase Szegő (mejora 2)"];
  const labels = ["K-Means", "DBSCAN", "OPTICS", "Spectral", "CD inicial", "CD mejora 1", "CD mejora 2"];
  const keys = [["principal|base", "Principales · base"], ["principal|perturbado", "Principales · perturbados"], ["reservado|base", "Reservados · base"], ["reservado|perturbado", "Reservados · perturbados"]];
  s.addChart(pres.charts.BAR, keys.map(([k, n]) => ({ name: n, labels, values: methods.map((m) => DATA.por_metodo[m][k]) })), {
    x: MX, y: 1.55, w: 7.6, h: 5.3, barDir: "bar", barGrouping: "clustered", catAxisOrientation: "maxMin",
    chartColors: ["0F4C4A", "7FB3A8", "D9A21B", "E9CF8A"], showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: BODY,
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFontSize: 10, catAxisLabelFontSize: 11, catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY,
    valGridLine: { color: "E3E7E6", size: 0.5 }, catGridLine: { style: "none" }, showTitle: true, title: "ARI medio (5 semillas de evaluación)", titleFontSize: 12, titleFontFace: BODY, titleColor: INK,
    catAxisLabelColor: INK, valAxisLabelColor: MUTED, barGapWidthPct: 50,
  });
  const rows = [["Método", "P · base", "P · pert.", "R · base"]].concat(methods.map((m, i) => [labels[i], ariCell(DATA.por_metodo[m]["principal|base"]), ariCell(DATA.por_metodo[m]["principal|perturbado"]), ariCell(DATA.por_metodo[m]["reservado|base"])]));
  table(s, rows, 8.45, 1.6, 4.28, [1.55, 0.91, 0.91, 0.91], { fontSize: 11.5, rowH: 0.4 });
  card(s, 8.45, 4.95, 4.28, 1.9, { tone: "neu", body: "P = 4 escenarios principales, R = 2 reservados. Con perturbaciones: mejora 2 0.75 vs Spectral 0.71 (diferencia 0.04).", size: 12.5 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 07 · Análisis", "Dónde gana CD mejorado y dónde pierde");
  img(s, "fig_grid_perturbaciones.png", MX, 1.5, { h: 5.05, alt: "Particiones bajo ruido, oclusión y densidad variable" });
  card(s, 7.05, 1.6, 5.68, 2.2, { tone: "good", title: "Ventajas", bullets: true, size: 12.5, gap: 3, body: ["Espirales limpias: 1.00, también en reservados — por encima del oráculo de Spectral (0.26) y CD local (0.64).", "La compuerta no se activa en ninguna semilla de compacto ni anidado."] });
  card(s, 7.05, 3.95, 5.68, 2.9, { tone: "bad", title: "Limitaciones", bullets: true, size: 12.5, gap: 3, body: ["Oclusión y densidad desplazan el centro de devanado: espirales 0.14–0.28.", "Con ruido la ganancia queda en el umbral: 0.42 ± 0.51.", "En lunas + ruido la fase cuesta 0.06.", "Con el eps correcto (oráculo), DBSCAN también resuelve espirales (0.94): la ventaja es elegir sin etiquetas.", "Pensada para espirales: no es una mejora general."] });
  caption(s, "Filas = variante perturbada (semilla 100); «fase ON/OFF» = decisión de la compuerta", MX, 6.6, 6.3, 0.3);
}

// =====================================================================================
// BLOQUE 08
divider(8, "Primera implementación en 3D (exploratoria)", [
  "¿Qué describe P_d cuando la nube es una superficie en el espacio?",
  "¿Cómo se leen un corte plano y una isosuperficie?",
  "¿Sirve la matriz de momentos como «huella» invariante de un objeto completo?",
]);

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 08 · Cómo leer las figuras 3D", "Un corte plano muestra P_d sobre una rebanada del espacio");
  img(s, "fig_paso_corte3d.png", MX, 1.5, { w: 8.9, alt: "Plano z = 0 que atraviesa el toro y P_d en ese plano" });
  card(s, 9.7, 1.6, 3.03, 2.5, { title: "Corte plano", body: "Se evalúa P_d solo en un plano que atraviesa el objeto y se pinta como en 2D. Puntos blancos: datos a menos de 0.04 del plano.", size: 12.5 });
  card(s, 9.7, 4.25, 3.03, 2.6, { tone: "neu", title: "Isosuperficie", body: "Superficie donde P_d = percentil 95 de los datos (dentro queda el 95 %), por marching cubes en una rejilla de 44³.", size: 12.5 });
  card(s, MX, 5.15, 8.9, 1.7, { tone: "good", title: "Qué se ve", body: "Con d = 2 el soporte es una envolvente difusa sin hueco. Con d = 4, P_d solo es pequeño sobre las dos circunferencias que el plano corta en el toro.", size: 13 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 08 · Soporte en 3D", "Con grado ≥ 4, P_d marca la superficie algebraica del objeto");
  img(s, "fig_3d_isosuperficies.png", MX, 1.5, { w: 9.3, alt: "Isosuperficie y cortes de P_d para toro y cono" });
  card(s, 10.1, 1.6, 2.63, 2.3, { title: "d = 2", body: "Envolvente cuadrática: buena silueta, sin el agujero del toro.", size: 12.5 });
  card(s, 10.1, 4.05, 2.63, 2.8, { tone: "good", title: "d ≥ 4", body: "El toro cumple una ecuación de grado 4; ese polinomio vale 0 en los datos y M_d se vuelve singular (cond 10¹⁷).", size: 12.5 });
  caption(s, "Toro: (x² + y² + z² + R² − r²)² = 4R²(x² + y²). En el cono, P_d marca también las prolongaciones de la pared y la base. exploratorio_3d.py · N = 1500 · semilla 0", MX, 6.3, 9.3, 0.55);
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 08 · Descriptor de objeto", "Una huella invariante a la rotación, paso a paso");
  img(s, "fig_paso_descriptor.png", MX, 1.5, { w: CW, alt: "Descriptor 3D: objetos, matrices promediadas y distancias" });
  card(s, MX, 5.3, 6.0, 1.55, { tone: "bad", title: "Problema", body: "Los momentos cambian al rotar: la misma esfera rotada da otra M_d. Hace falta un descriptor que no dependa de la pose.", size: 12.5 });
  card(s, 6.85, 5.3, 5.88, 1.55, { tone: "good", title: "Solución", body: "Promediar M_d sobre muchas rotaciones al azar (operador de Reynolds): en el límite da lo mismo para cualquier pose.", size: 12.5 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 08 · Clasificación 1-NN", "¿Se reconoce cada objeto por su vecino más parecido?");
  img(s, "fig_3d_clases.png", MX, 1.45, { w: 7.6, alt: "Siete clases de objetos 3D" });
  const ms = ["CD + Log-Euclidiana (d=2, 30 rot.)", "CD + Log-Euclidiana (d=4, 30 rot.)", "CD + Log-Euclidiana (d=4, 300 rot.)", "D2 (Osada 2002)", "Eigen-rasgos PCA"];
  const lab = ["CD d=2, 30 rot.", "CD d=4, 30 rot.", "CD d=4, 300 rot.", "D2 (Osada)", "Eigen PCA"];
  s.addChart(pres.charts.BAR, [["base", "Base"], ["ruido", "Ruido 3 %"], ["oclusion", "Oclusión 20 %"]].map(([v, n]) => ({ name: n, labels: lab, values: ms.map((m) => get3d(m, v)) })), {
    x: MX, y: 2.75, w: 7.6, h: 4.1, barDir: "col", barGrouping: "clustered", chartColors: ["0F4C4A", "7FB3A8", "D9A21B"],
    valAxisMinVal: 0, valAxisMaxVal: 1, showLegend: true, legendPos: "t", legendFontSize: 11, legendFontFace: BODY,
    valAxisLabelFontSize: 10, catAxisLabelFontSize: 11, catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY, catAxisLabelColor: INK, valAxisLabelColor: MUTED,
    valGridLine: { color: "E3E7E6", size: 0.5 }, catGridLine: { style: "none" }, showValue: true, dataLabelFontSize: 9, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", dataLabelColor: INK,
    valAxisTitle: "Exactitud 1-NN (LOO)", showValAxisTitle: true, valAxisTitleFontSize: 10, valAxisTitleColor: MUTED,
  });
  card(s, 8.45, 1.6, 4.28, 2.1, { tone: "neu", title: "Qué mide", body: "70 objetos (7 clases × 10, pose al azar). Cada uno se aparta y recibe la clase de su objeto más parecido entre los otros 69. Al azar: 1/7 ≈ 0.14.", size: 12 });
  card(s, 8.45, 3.85, 4.28, 1.45, { tone: "good", title: "Favorable", body: "Con 300 rotaciones CD llega a 0.99 y supera a los eigen-rasgos PCA (0.86).", size: 12.5 });
  card(s, 8.45, 5.45, 4.28, 1.4, { tone: "bad", title: "Desfavorable", body: "D2 es mejor: 1.00 limpio y 0.96 con oclusión, contra 0.66 de CD.", size: 12.5 });
  s.addNotes("La configuración a priori era 30 rotaciones. Un diagnóstico en la semilla de desarrollo 7 mostró que el promedio era ruidoso (30 → 0.94, 100 → 0.99, 300 → 1.00), así que se añadió la de 300 y se reportan ambas. Esto contradice al reporte previo (CD > D2).");
}

// =====================================================================================
// BLOQUE 09
divider(9, "Estado y siguientes pasos", [
  "¿Qué se construyó y qué evidencia respalda cada objetivo?",
  "¿Qué resultados del reporte previo se reprodujeron y cuáles no?",
  "¿Qué falta y qué limitación medida motiva cada pendiente?",
]);

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 09 · Estado", "Avance frente a los objetivos");
  const ok = { text: "Implementado y verificado", options: { color: GOOD_TX, bold: true } };
  const pre = { text: "Evaluación preliminar", options: { color: GOLD_TXT, bold: true } };
  const exp = { text: "Exploratorio", options: { color: BAD_TX, bold: true } };
  const rows = [
    ["Objetivo", "Estado", "Qué se hizo y evidencia", "Qué falta"],
    ["1. Implementar CD; grado y regularización", ok, "ChristoffelDarboux + diagnóstico de grado; traza < 10⁻³; pruebas", "Comparar regularizaciones en el benchmark"],
    ["2. Datos sintéticos y perturbaciones", ok, "12 primitivas 2D, 7 superficies 3D, 6 escenarios, 3 perturbaciones; 77 pruebas", "Más niveles por perturbación"],
    ["3. Similitud, embeddings y agrupamiento con CD", pre, "3 etapas de CD; embedding espectral; 840 corridas", "Centro de devanado robusto; k sin etiquetas"],
    ["4. Métricas y robustez", pre, "ARI, NMI, T, C, 5-NN; 5 semillas; oráculo", "Curvas por nivel de perturbación"],
    ["5. Comparación con referencias", pre, "4 métodos de agrupamiento y 5 de embedding", "Repetir con datos reales"],
    ["6. Aplicación 3D tipo ShapeNet", exp, "Soporte 3D y descriptor invariante en primitivas", "ShapeNetPart / ModelNet reales"],
  ];
  table(s, rows, MX, 1.6, CW, [3.3, 2.2, 3.85, 2.78], { fontSize: 12, rowH: 0.58 });
  card(s, MX, 5.85, CW, 1.0, { tone: "neu", body: "Implementado y verificado = código + pruebas + resultados reproducidos. Evaluación preliminar = benchmark con varias semillas, faltan casos o una limitación por resolver. Exploratorio = corre de extremo a extremo, sin datos reales.", size: 12.5 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 09 · Verificación", "Resultados del reporte previo: cuáles se reprodujeron");
  const Y = { text: "Se reproduce", options: { color: GOOD_TX, bold: true } };
  const N = { text: "No se reproduce", options: { color: BAD_TX, bold: true } };
  const P = { text: "Parcial", options: { color: GOLD_TXT, bold: true } };
  const rows = [
    ["Afirmación previa", "Este corte", "Dictamen"],
    ["P_d separa soporte, hueco y atípicos en el anillo", "Hueco fuera del soporte desde d = 2; atípico lejano de 188× a 6·10⁷× la mediana", Y],
    ["CD local-espectral resuelve espirales (0.99–1.00 con k diagnosticado)", "k = 10: 0.17; mejor k con etiquetas: 0.64", N],
    ["Fase de Szegő: ARI 1.00 en espirales", "1.00 entrelazado y 3 brazos, 0.92 logarítmica; falla con oclusión", P],
    ["CD + Log-Euclidiana supera a D2 en clasificación de objetos", "D2 1.00 vs CD 0.99 (300 rot.); con oclusión 0.96 vs 0.66", N],
    ["Embedding CD comparable a PCA, por debajo de UMAP", "T = 0.86 (CD) vs 0.81 (PCA) vs 0.95 (UMAP)", Y],
  ];
  table(s, rows, MX, 1.6, CW, [4.9, 5.2, 2.03], { fontSize: 13, rowH: 0.62 });
  card(s, MX, 5.6, CW, 1.25, { tone: "neu", body: "Las diferencias pueden venir de otras geometrías, tamaños y parámetros del reporte previo; solo se presentan como resultados los recalculados aquí. benchmarks/verificar_cifras.py comprueba 68 cifras citadas.", size: 13.5 });
}

{
  const s = pres.addSlide({ masterName: "CONTENIDO" });
  header(s, "Bloque 09 · Plan", "Pendientes hacia la presentación final");
  const rows = [
    ["Pendiente", "Motivo (evidencia)", "Responsable", "Fecha prevista"],
    ["Estimar el centro de devanado de forma robusta", "Oclusión y densidad cierran la compuerta", "Por asignar", "Por definir"],
    ["Elegir k de vecindad sin etiquetas (CD local)", "Oráculo 0.64 vs 0.17 por defecto en espirales", "Por asignar", "Por definir"],
    ["Conteo automático de grupos (homología persistente)", "Hoy K-Means/Spectral/CD reciben k", "Por asignar", "Por definir"],
    ["Promedio de Reynolds exacto o más eficiente", "30 rotaciones limitan a 0.89; 300 dan 0.99", "Por asignar", "Por definir"],
    ["ShapeNetPart / ModelNet reales", "Objetivo 6; todo lo actual es sintético", "Por asignar", "Por definir"],
    ["Verificar redacción de objetivos", "Lista tomada del reporte consolidado", "Por asignar", "Por definir"],
  ];
  table(s, rows, MX, 1.6, CW, [4.5, 4.33, 1.65, 1.65], { fontSize: 12.5, rowH: 0.5 });
  card(s, MX, 5.4, CW, 1.45, { tone: "dark", title: "Entregables de este corte", titleSize: 15, body: "Repositorio point-field (librería, núcleo CD, benchmarks reproducibles, verificación de cifras), documento explicativo detallado y esta presentación con respaldo.", size: 13.5 });
}

// =====================================================================================
// RESPALDO
{
  const s = pres.addSlide({ masterName: "TITULO" });
  s.addText("DIAPOSITIVAS DE RESPALDO", { x: MX, y: 2.3, w: CW, h: 0.4, fontFace: BODY, fontSize: 14, bold: true, color: "E3B23C", charSpacing: 3, margin: 0, isTextBox: true });
  s.addText("Fórmulas, techo oráculo, compuerta por caso, reproducibilidad y rejilla completa", { x: MX, y: 2.8, w: 10, h: 1.4, fontFace: HEAD, fontSize: 34, bold: true, color: "FFFFFF", margin: 0, isTextBox: true });
}

{
  const s = pres.addSlide({ masterName: "RESPALDO" });
  header(s, "Respaldo", "Hoja de fórmulas de los métodos evaluados");
  const F = [
    ["Afinidad CD local (mejora 1)", "P_i(x_j) con M_d ajustada en los k+1 vecinos de x_i;  z_ij = (P_i(x_j) − med_i)/MAD_i;  A_ij = exp(−z_ij²/2),  A = max(A, Aᵀ)"],
    ["Con d = 1", "P_i(x) − 1 = distancia de Mahalanobis local: el grafo sigue la orientación local de los datos"],
    ["Fase de Szegő (mejora 2)", "ψ = θ − c·g(r),  g ∈ {r, log r},  c = 2π·vueltas / (g₉₉ − g₁)"],
    ["Compuerta", "ganancia = sil(ψ*) − sil(θ);  activar si ganancia ≥ 0.20"],
    ["Descriptor 3D", "M̄_d = E_R[M_d(RX)] (30 o 300 rotaciones Haar),  M̃ = M̄ / tr M̄,  d_LE = ‖log M̃₁ − log M̃₂‖_F"],
    ["Trustworthiness (k)", "T = 1 − 2/(Nk(2N−3k−1)) Σᵢ Σ_{j∈U_k(i)} (r(i,j) − k);  C: mismo cálculo con espacios intercambiados"],
  ];
  F.forEach(([a, b], i) => {
    const y = 1.6 + i * 0.86;
    card(s, MX, y, CW, 0.76, {});
    s.addText(a, { x: MX + 0.2, y: y + 0.08, w: 3.1, h: 0.6, fontFace: BODY, fontSize: 13, bold: true, color: INK, margin: 0, isTextBox: true, valign: "middle" });
    s.addText(b, { x: MX + 3.4, y: y + 0.08, w: CW - 3.6, h: 0.6, fontFace: HEAD, fontSize: 14, color: "33413F", margin: 0, isTextBox: true, valign: "middle" });
  });
}

{
  const s = pres.addSlide({ masterName: "RESPALDO" });
  header(s, "Respaldo", "Configuración por defecto vs techo oráculo (usa etiquetas)");
  const cases = ["compacto", "no_convexo", "entrelazado", "anidado", "espiral_3brazos", "espiral_log"];
  const ms = [["DBSCAN", "DBSCAN"], ["Spectral (kNN)", "Spectral"], ["CD local-espectral (mejora 1)", "CD local"]];
  const rows = [["Caso (base)"].concat(ms.flatMap(([, n]) => [n + " def.", n + " oráculo"])).concat(["CD + Szegő def."])];
  cases.forEach((c) => {
    const row = [c];
    ms.forEach(([m]) => {
      const r = DATA.oracle.find((o) => o.case === c && o.variant === "base" && o.method === m);
      row.push(ariCell(r.ARI_default), ariCell(r.ARI_oracle));
    });
    const sz = DATA.oracle.find((o) => o.case === c && o.variant === "base" && o.method === "CD + fase Szegő (mejora 2)");
    row.push(ariCell(sz.ARI_default));
    rows.push(row);
  });
  table(s, rows, MX, 1.6, CW, [2.2, 1.4, 1.45, 1.4, 1.45, 1.4, 1.45, 1.38], { fontSize: 12.5, rowH: 0.45 });
  card(s, MX, 5.0, CW, 1.85, { tone: "neu", title: "Cómo leerla", body: "Oráculo = mejor ARI en una rejilla (DBSCAN: 20 valores de eps; Spectral y CD local: k ∈ {5,…,30}) elegida con las etiquetas, por semilla. Es una cota de potencial, no un resultado. Perturbados y reservados: results/tabla_oracle.csv.", size: 13 });
}

{
  const s = pres.addSlide({ masterName: "RESPALDO" });
  header(s, "Respaldo", "Compuerta de Szegő: desarrollo (0–2) y evaluación (100–104)");
  const vs = [["base", "base"], ["ruido", "+ ruido"], ["oclusion", "+ oclusión"], ["densidad_variable", "+ dens. var."]];
  const cs = ["compacto", "no_convexo", "entrelazado", "anidado", "espiral_3brazos", "espiral_log"];
  const rows = [["Caso"].concat(vs.map((v) => v[1]))];
  cs.forEach((c) => {
    const row = [{ text: c + (c.startsWith("espiral_") ? " *" : ""), options: { bold: true } }];
    vs.forEach(([v]) => {
      const e = DATA.gate_eval.find((x) => x.case === c && x.variant === v);
      const d = DATA.gate_dev.find((x) => x.case === c && x.variant === v);
      const on = e.fase_activada;
      row.push({ text: `${f2(d.g)}  /  ${f2(e.ganancia_devanado)}   ${Math.round(on * 100)} %`, options: { align: "center", fill: { color: on >= 0.5 ? "CFE8DD" : (on > 0 ? "FFF1C9" : "FFFFFF") }, bold: on >= 0.5 } });
    });
    rows.push(row);
  });
  table(s, rows, MX, 1.6, CW, [2.5, 2.4, 2.4, 2.4, 2.43], { fontSize: 12.5, rowH: 0.55 });
  caption(s, "Celda: ganancia media en desarrollo (0–2) / en evaluación (100–104) y % de semillas de evaluación con la fase activada. Verde: activada en la mayoría. * reservado.", MX, 5.6, CW, 0.6);
}

{
  const s = pres.addSlide({ masterName: "RESPALDO" });
  header(s, "Respaldo", "Reproducibilidad: script, datos, semillas y salida");
  const rows = [
    ["Script (benchmarks/)", "Datos y parámetros", "Semillas", "Salida"],
    ["fig_cd_2d.py", "anillo Annulus(0.75,1), N = 300; lunas; d ∈ {1,2,4,6}; ε = 10⁻⁸", "0", "fig_cd_2d · cd_2d_degree_diagnostics.csv"],
    ["fig_libreria.py", "6 escenarios, 3 perturbaciones, 6 variaciones", "0", "fig_primitivas · fig_libreria · escenarios.json"],
    ["benchmark_clustering_2d.py", "6 casos × 4 variantes × 7 métodos; oráculo", "0–2 dev; 100–104", "clustering_2d_{dev,eval,oracle}.csv"],
    ["fig_benchmark_clustering.py", "lee los CSV; re-ejecuta la semilla 100 para figuras", "100", "mapas de calor, rejillas, tablas"],
    ["benchmark_embeddings_2d.py", "4 casos enrollados en 3D; 6 métodos; k = 10", "100–104", "embeddings_2d_eval.csv"],
    ["fig_mejoras.py · fig_explicativas.py", "figuras paso a paso de cada bloque", "0; 100", "fig_mejoras · fig_paso_*"],
    ["exploratorio_3d.py", "7 clases × 10, 400 pts; d ∈ {2,4}; 30/300 rot.", "0; 100–104", "fig_3d_* · clasificacion_3d.csv"],
    ["verificar_cifras.py · run_all.sh", "comprueba 68 cifras citadas · ejecuta todo en orden", "—", "salida en consola"],
  ];
  table(s, rows, MX, 1.6, CW, [3.2, 4.65, 1.6, 2.68], { fontSize: 11.5, rowH: 0.47 });
  card(s, MX, 6.05, CW, 0.6, { tone: "dark", body: "Python 3.13 · numpy 2.5.3 · scipy 1.18.1 · scikit-learn 1.9.1 · scikit-image 0.26 · umap-learn 0.5.12 · protocolo en benchmarks/BENCHMARKS.md", size: 13 });
}

{
  const s = pres.addSlide({ masterName: "RESPALDO" });
  header(s, "Respaldo", "Todos los métodos en los seis casos base (semilla 100)");
  img(s, "fig_grid_clustering.png", MX, 1.45, { h: 5.5, alt: "Particiones de siete métodos en seis casos base" });
  card(s, 7.3, 1.6, 5.43, 2.9, { title: "Cómo leer la rejilla", body: ["Filas: escenarios; columnas: métodos; color = grupo asignado, gris = ruido (DBSCAN/OPTICS).", "Esquina inferior derecha: ARI; inferior izquierda: tiempo con carga de fondo (indicativo).", "«fase ON/OFF»: decisión de la compuerta de Szegő.", "Las dos últimas filas son casos reservados."], size: 13, bullets: true, gap: 5 });
}

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote", OUT));
