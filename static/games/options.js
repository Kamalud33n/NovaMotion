const GAME_CFG_PREFIX = 'novamotion_game_cfg_';
let gameSchema = [];
let gameCfg = {};
let gameCfgId = null;

function validGameValue(f, v) {
    return f.options.some(o => o.value === v) ? v : f.default;
}

function loadGameCfg(game) {
    gameCfgId = game.id;
    gameSchema = game.config_schema || [];
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(GAME_CFG_PREFIX + game.id) || '{}'); } catch (e) { saved = {}; }
    gameCfg = {};
    gameSchema.forEach(f => { gameCfg[f.key] = validGameValue(f, saved[f.key]); });
}

function saveGameCfg() {
    try { localStorage.setItem(GAME_CFG_PREFIX + gameCfgId, JSON.stringify(gameCfg)); } catch (e) { /* storage unavailable */ }
}

function gameConfigValues() { return { ...gameCfg }; }

function setGameChoice(f, value) {
    gameCfg[f.key] = value;
    saveGameCfg();
    renderGameOptions();
}

function mkEl(tag, cls, text) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text != null) el.textContent = text;
    return el;
}

function renderGameOptions() {
    const box = document.getElementById('gameOptions');
    if (!box) return;
    box.textContent = '';
    gameSchema.forEach(f => {
        const g = mkEl('div', 'form-group');
        g.appendChild(mkEl('label', '', f.label));
        const sel = mkEl('div', 'duration-selector');
        f.options.forEach(o => {
            const b = mkEl('button', 'dur-btn' + (gameCfg[f.key] === o.value ? ' active' : ''), o.label);
            b.type = 'button';
            b.onclick = () => setGameChoice(f, o.value);
            sel.appendChild(b);
        });
        g.appendChild(sel);
        box.appendChild(g);
    });
}