const LOTS = [
  { id: 'A1', name: 'Левый верхний', size: '42 × 26 см', base: 8900, bid: 8900, owner: null },
  { id: 'A2', name: 'Центр, уровень глаз', size: '42 × 26 см', base: 12900, bid: 15500, owner: 'kofeynya_51' },
  { id: 'A3', name: 'Правый верхний', size: '42 × 26 см', base: 8900, bid: 9400, owner: 'app_taxi' },
  { id: 'B1', name: 'Левый нижний', size: '42 × 26 см', base: 6900, bid: 6900, owner: null },
  { id: 'B2', name: 'Центр нижний', size: '42 × 26 см', base: 9900, bid: 11200, owner: 'remont_pod_kluch' },
  { id: 'B3', name: 'Правый нижний', size: '42 × 26 см', base: 6900, bid: 6900, owner: null }
];

const FULL = { id: 'FULL', short: 'ВСЁ', name: 'Всё стекло целиком', size: '130 × 55 см', base: 39000, bid: 42000, owner: 'dostavka_v_ryadom' };

const STEP = 500;
const SWATCHES = ['transparent', '#c9f24a', '#4aa8f2', '#ffffff', '#ff6b6b', '#111418'];

const state = {
  mode: 'lots',
  selected: 'A1',
  items: {},
  history: []
};

const money = n => new Intl.NumberFormat('ru-RU').format(Math.round(n)) + ' ₽';
const byId = id => (id === 'FULL' ? FULL : LOTS.find(l => l.id === id));
const tag = lot => lot.short || lot.id;
const activeLots = () => (state.mode === 'full' ? [FULL] : LOTS);

const el = {
  grid: document.getElementById('lotsGrid'),
  lotsBox: document.getElementById('lotsGrid'),
  stage: document.getElementById('stage'),
  table: document.getElementById('lotsTable'),
  title: document.getElementById('panelTitle'),
  price: document.getElementById('panelPrice'),
  modeLots: document.getElementById('modeLots'),
  modeFull: document.getElementById('modeFull'),
  drop: document.getElementById('drop'),
  file: document.getElementById('file'),
  presets: document.getElementById('presets'),
  caption: document.getElementById('caption'),
  scale: document.getElementById('scale'),
  scaleVal: document.getElementById('scaleVal'),
  swatches: document.getElementById('swatches'),
  clearLot: document.getElementById('clearLot'),
  clearAll: document.getElementById('clearAll'),
  currentBid: document.getElementById('currentBid'),
  bidInput: document.getElementById('bidInput'),
  bidBtn: document.getElementById('bidBtn'),
  bidHint: document.getElementById('bidHint'),
  sumCount: document.getElementById('sumCount'),
  sumTotal: document.getElementById('sumTotal'),
  history: document.getElementById('historyList'),
  navTimer: document.getElementById('navTimer'),
  auctionDate: document.getElementById('auctionDate'),
  bars: document.getElementById('bars'),
  form: document.getElementById('form'),
  formOk: document.getElementById('formOk'),
  year: document.getElementById('year')
};

function item(id) {
  if (!state.items[id]) state.items[id] = { src: null, caption: '', scale: 100, bg: 'transparent' };
  return state.items[id];
}

function preset(kind) {
  const art = {
    logo: `<rect width="400" height="250" fill="#11151b"/><circle cx="200" cy="104" r="46" fill="#c9f24a"/>
      <text x="200" y="120" text-anchor="middle" font-size="46" font-weight="800" fill="#11151b" font-family="Manrope,sans-serif">B</text>
      <text x="200" y="192" text-anchor="middle" font-size="30" font-weight="800" letter-spacing="6" fill="#ffffff" font-family="Manrope,sans-serif">BRAND</text>`,
    promo: `<rect width="400" height="250" fill="#c9f24a"/>
      <text x="200" y="128" text-anchor="middle" font-size="86" font-weight="800" fill="#11151b" font-family="Manrope,sans-serif">-20%</text>
      <text x="200" y="176" text-anchor="middle" font-size="24" font-weight="700" letter-spacing="3" fill="#2c3a08" font-family="Manrope,sans-serif">ПО ПРОМОКОДУ</text>`,
    qr: `<rect width="400" height="250" fill="#ffffff"/>
      <g fill="#11151b"><rect x="146" y="26" width="108" height="108" rx="6" fill="none" stroke="#11151b" stroke-width="8"/>
      <rect x="158" y="38" width="26" height="26"/><rect x="216" y="38" width="26" height="26"/><rect x="158" y="96" width="26" height="26"/>
      <rect x="196" y="70" width="16" height="16"/><rect x="222" y="82" width="16" height="16"/><rect x="238" y="108" width="14" height="14"/><rect x="196" y="108" width="14" height="14"/></g>
      <text x="200" y="196" text-anchor="middle" font-size="28" font-weight="800" letter-spacing="2" fill="#11151b" font-family="Manrope,sans-serif">СКАНИРУЙ</text>`
  };
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="250" viewBox="0 0 400 250">${art[kind]}</svg>`;
  return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
}

function renderLots() {
  const lots = activeLots();
  el.grid.classList.toggle('is-full', state.mode === 'full');
  el.grid.innerHTML = '';

  lots.forEach(lot => {
    const data = item(lot.id);
    const node = document.createElement('button');
    node.type = 'button';
    node.className = 'lot';
    node.dataset.id = lot.id;
    node.setAttribute('aria-label', `${lot.id} — ${lot.name}`);
    if (state.mode === 'full') node.classList.add('is-full');
    if (state.selected === lot.id) node.classList.add('is-selected');
    if (lot.owner === 'вы') node.classList.add('is-mine');
    if (lot.owner) node.classList.add('is-taken');
    if (data.bg !== 'transparent') node.style.background = data.bg;

    if (data.src) {
      const img = document.createElement('img');
      img.className = 'lot__img';
      img.src = data.src;
      img.alt = '';
      img.style.transform = `scale(${data.scale / 100})`;
      node.appendChild(img);
    }
    if (data.caption) {
      const cap = document.createElement('span');
      cap.className = 'lot__caption';
      cap.textContent = data.caption;
      node.appendChild(cap);
    }

    const label = document.createElement('span');
    label.className = 'lot__label';
    label.innerHTML = `<span>${tag(lot)}</span><span>${money(lot.bid)}</span>`;
    node.appendChild(label);

    node.addEventListener('click', () => select(lot.id));
    el.grid.appendChild(node);
  });
}

function renderTable() {
  el.table.innerHTML = '';
  [...LOTS, FULL].forEach(lot => {
    const tr = document.createElement('tr');
    const status = !lot.owner
      ? '<span class="tag">Свободен</span>'
      : lot.owner === 'вы'
        ? '<span class="tag tag--mine">Ваша ставка</span>'
        : `<span class="tag tag--bid">${lot.owner}</span>`;
    tr.innerHTML = `
      <td><b>${tag(lot)}</b></td>
      <td>${lot.name}</td>
      <td>${lot.size}</td>
      <td>${money(lot.base)}</td>
      <td><b>${money(lot.bid)}</b></td>
      <td>${status}</td>`;
    el.table.appendChild(tr);
  });
}

function renderPanel() {
  const lot = byId(state.selected);
  const data = item(lot.id);

  el.title.textContent = `${tag(lot)} · ${lot.name}`;
  el.price.textContent = new Intl.NumberFormat('ru-RU').format(lot.base);
  el.currentBid.textContent = money(lot.bid);
  el.bidInput.value = lot.bid + STEP;
  el.bidInput.min = lot.bid + STEP;
  el.caption.value = data.caption;
  el.scale.value = data.scale;
  el.scaleVal.textContent = data.scale + '%';
  el.bidHint.classList.remove('is-err');
  el.bidHint.textContent = `Минимальная ставка — ${money(lot.bid + STEP)} · шаг ${money(STEP)}`;

  [...el.swatches.children].forEach(sw => sw.classList.toggle('is-on', sw.dataset.color === data.bg));
  el.modeLots.classList.toggle('is-on', state.mode === 'lots');
  el.modeFull.classList.toggle('is-on', state.mode === 'full');
}

function renderSummary() {
  const mine = [...LOTS, FULL].filter(l => l.owner === 'вы');
  el.sumCount.textContent = mine.length;
  el.sumTotal.textContent = money(mine.reduce((acc, l) => acc + l.bid, 0));
}

function renderHistory() {
  el.history.innerHTML = state.history.length
    ? ''
    : '<li><span>Ставок пока нет</span><b>—</b></li>';
  state.history.slice(0, 12).forEach(row => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${row.time} · ${row.lot} · ${row.who}</span><b>${money(row.value)}</b>`;
    el.history.appendChild(li);
  });
}

function render() {
  renderLots();
  renderTable();
  renderPanel();
  renderSummary();
  renderHistory();
}

function select(id) {
  state.selected = id;
  render();
}

function setMode(mode) {
  state.mode = mode;
  state.selected = mode === 'full' ? 'FULL' : 'A1';
  render();
}

function applyImage(src) {
  item(state.selected).src = src;
  render();
}

function readFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const reader = new FileReader();
  reader.onload = e => applyImage(e.target.result);
  reader.readAsDataURL(file);
}

function placeBid() {
  const lot = byId(state.selected);
  const value = Number(el.bidInput.value);
  const min = lot.bid + STEP;
  if (!Number.isFinite(value) || value < min) {
    el.bidHint.textContent = `Ставка должна быть не меньше ${money(min)}`;
    el.bidHint.classList.add('is-err');
    return;
  }
  lot.bid = value;
  lot.owner = 'вы';
  state.history.unshift({
    lot: tag(lot),
    who: 'вы',
    value,
    time: new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
  });
  render();
}

function initSwatches() {
  SWATCHES.forEach(color => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'swatch';
    b.dataset.color = color;
    b.title = color === 'transparent' ? 'Без фона' : color;
    b.style.background = color === 'transparent'
      ? 'repeating-conic-gradient(#2a323d 0 25%, #1a2029 0 50%) 0 0/12px 12px'
      : color;
    b.addEventListener('click', () => {
      item(state.selected).bg = color;
      render();
    });
    el.swatches.appendChild(b);
  });
}

function initBuilderEvents() {
  el.modeLots.addEventListener('click', () => setMode('lots'));
  el.modeFull.addEventListener('click', () => setMode('full'));
  el.drop.addEventListener('click', () => el.file.click());
  el.file.addEventListener('change', e => readFile(e.target.files[0]));

  el.presets.addEventListener('click', e => {
    const kind = e.target.dataset.preset;
    if (!kind) return;
    applyImage(preset(kind));
  });

  el.caption.addEventListener('input', e => {
    item(state.selected).caption = e.target.value;
    renderLots();
  });

  el.scale.addEventListener('input', e => {
    item(state.selected).scale = Number(e.target.value);
    el.scaleVal.textContent = e.target.value + '%';
    renderLots();
  });

  el.clearLot.addEventListener('click', () => {
    delete state.items[state.selected];
    render();
  });

  el.clearAll.addEventListener('click', () => {
    state.items = {};
    render();
  });

  el.bidBtn.addEventListener('click', placeBid);
  el.bidInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') placeBid();
  });

  ['dragenter', 'dragover'].forEach(type =>
    el.stage.addEventListener(type, e => {
      e.preventDefault();
      el.stage.classList.add('is-drag');
    })
  );
  ['dragleave', 'drop'].forEach(type =>
    el.stage.addEventListener(type, e => {
      e.preventDefault();
      el.stage.classList.remove('is-drag');
    })
  );
  el.stage.addEventListener('drop', e => readFile(e.dataTransfer.files[0]));
}

function initTimer() {
  const now = new Date();
  const end = new Date(now.getFullYear(), now.getMonth() + 1, 0, 21, 0, 0);
  if (end - now < 0) end.setMonth(end.getMonth() + 1);
  el.auctionDate.textContent = end.toLocaleDateString('ru-RU', { day: 'numeric', month: 'long' });

  const tick = () => {
    const diff = Math.max(0, end - new Date());
    const d = Math.floor(diff / 86400000);
    const h = String(Math.floor(diff / 3600000) % 24).padStart(2, '0');
    const m = String(Math.floor(diff / 60000) % 60).padStart(2, '0');
    const s = String(Math.floor(diff / 1000) % 60).padStart(2, '0');
    el.navTimer.textContent = `${d}д ${h}:${m}:${s}`;
  };
  tick();
  setInterval(tick, 1000);
}

function initBars() {
  const data = [
    { label: 'Апр', value: 2980 },
    { label: 'Май', value: 3410 },
    { label: 'Июн', value: 3620 },
    { label: 'Июл', value: 2740 },
    { label: 'Авг', value: 3180 },
    { label: 'Сен', value: 3290 }
  ];
  const max = Math.max(...data.map(d => d.value));
  data.forEach(d => {
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.innerHTML = `<span class="bar__val">${new Intl.NumberFormat('ru-RU').format(d.value)}</span><span class="bar__fill" data-h="${Math.round((d.value / max) * 78)}"></span><span class="bar__label">${d.label}</span>`;
    el.bars.appendChild(bar);
  });

  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      el.bars.querySelectorAll('.bar__fill').forEach((fill, i) => {
        setTimeout(() => {
          fill.style.height = fill.dataset.h + '%';
        }, i * 90);
      });
      io.disconnect();
    });
  }, { threshold: 0.35 });
  io.observe(el.bars);
}

function initCounters() {
  const nodes = document.querySelectorAll('[data-count]');
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const node = entry.target;
      const target = Number(node.dataset.count);
      const start = performance.now();
      const duration = 1100;
      const step = now => {
        const p = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(1 - p, 3);
        node.textContent = new Intl.NumberFormat('ru-RU').format(Math.round(target * eased));
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
      io.unobserve(node);
    });
  }, { threshold: 0.6 });
  nodes.forEach(n => io.observe(n));
}

function initReveal() {
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('is-in');
      io.unobserve(entry.target);
    });
  }, { threshold: 0.18 });
  document.querySelectorAll('.reveal').forEach(n => io.observe(n));
}

function initForm() {
  el.form.addEventListener('submit', e => {
    e.preventDefault();
    let valid = true;
    el.form.querySelectorAll('input[required]').forEach(input => {
      const ok = input.value.trim().length > 1;
      input.parentElement.classList.toggle('is-invalid', !ok);
      if (!ok) valid = false;
    });
    if (!valid) return;

    const mine = [...LOTS, FULL].filter(l => l.owner === 'вы');
    const tail = mine.length
      ? ` Ваши лоты: ${mine.map(tag).join(', ')} на ${money(mine.reduce((a, l) => a + l.bid, 0))} в месяц.`
      : '';
    el.formOk.hidden = false;
    el.formOk.textContent = `Заявка принята. Свяжемся в течение дня.${tail}`;
    el.form.reset();
  });
}

initSwatches();
initBuilderEvents();
initTimer();
initBars();
initCounters();
initReveal();
initForm();
render();
el.year.textContent = new Date().getFullYear();
