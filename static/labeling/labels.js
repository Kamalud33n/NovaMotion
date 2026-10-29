/* Label editor for the labeling viewer: label buttons, compensation tags, note, and the
   "Your labels" table. Talks to the label API; the timeline is wired in by viewer.js.

   LabelingLabels.init({
       onLabels: (labels) => tl.setLabels(labels),   // the user's labels changed
       setSelection: (sel) => tl.setSelection(sel),  // highlight a range on the timeline
       seek: (frame) => {...},                       // move the playhead (optional)
   });
   LabelingLabels.select(sel);   // timeline selection: { kind: 'rep', start, end, rep } | { kind: 'label', label } | null

   rep_index is stored 0-based: repetition 3 on the timeline is rep_index 2. */
(function () {
    'use strict';

    const $ = (id) => document.getElementById(id);
    const els = {
        hint: $('lvEditorHint'),
        body: $('lvEditorBody'),
        info: $('lvSelInfo'),
        choices: $('lvChoices'),
        tagsWrap: $('lvTagsWrap'),
        tags: $('lvTags'),
        note: $('lvNote'),
        save: $('lvSave'),
        del: $('lvDelete'),
        cancel: $('lvCancel'),
        msg: $('lvMsg'),
        count: $('lvLabelCount'),
        table: $('lvLabelBody'),
        suggest: $('lvSuggest'),
        suggestText: $('lvSuggestText'),
        suggestAccept: $('lvSuggestAccept'),
    };

    const MIN_SUGGEST_FRAMES = 5;

    let hooks = { onLabels: null, setSelection: null, seek: null };
    let rec = null;
    let defs = null;
    let labels = [];
    let suggestions = [];
    let target = null;
    let editing = null;
    let choice = null;
    let tags = new Set();
    let busy = false;
    let token = 0;

    async function request(method, url, body) {
        const options = { method, credentials: 'same-origin' };
        if (body) {
            options.headers = { 'Content-Type': 'application/json' };
            options.body = JSON.stringify(body);
        }
        const res = await fetch(url, options);
        if (res.status === 401) {
            window.location.href = '/login';
            throw new Error('Not signed in');
        }
        if (!res.ok) {
            let message = `Request failed (${res.status})`;
            try {
                const data = await res.json();
                if (typeof data.detail === 'string') message = data.detail;
                else if (Array.isArray(data.detail)) message = 'Some values are not valid. Check them and try again.';
            } catch (e) { /* keep default */ }
            throw new Error(message);
        }
        return res.json();
    }

    const base = () => `/api/recordings/${encodeURIComponent(rec.patientId)}/${encodeURIComponent(rec.recordingId)}/labels`;
    const suggestUrl = () => base().replace(/\/labels$/, '/label-suggestions');
    const specOf = (id) => (defs ? defs.labels.find((l) => l.id === id) : null);
    const labelName = (id) => (specOf(id) ? specOf(id).name : id);
    const tagName = (id) => {
        const t = defs ? defs.tags.find((x) => x.id === id) : null;
        return t ? t.name : id;
    };
    const range = (start, end) => `${start}\u2013${end}`;

    function setMsg(text, isError) {
        els.msg.textContent = text || '';
        els.msg.classList.toggle('is-error', !!isError);
    }

    function notify() {
        if (hooks.onLabels) hooks.onLabels(labels.slice());
    }

    function highlight(row) {
        if (!hooks.setSelection) return;
        hooks.setSelection(row ? { kind: 'label', label: row, start: row.start_frame, end: row.end_frame } : null);
    }

    function buildChoices() {
        els.choices.replaceChildren();
        defs.labels.forEach((spec) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'lv-choice';
            button.dataset.label = spec.id;
            button.textContent = spec.name;
            if (spec.description) button.title = spec.description;
            els.choices.appendChild(button);
        });
        els.tags.replaceChildren();
        defs.tags.forEach((tag) => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'lv-chip';
            chip.dataset.tag = tag.id;
            chip.textContent = tag.name;
            if (tag.description) chip.title = tag.description;
            els.tags.appendChild(chip);
        });
    }

    async function ensureDefs() {
        if (defs) return;
        defs = await request('GET', '/api/label-definitions');
        buildChoices();
    }

    function currentSuggestion() {
        if (!target || editing || target.blocked || !Number.isInteger(target.repIndex)) return null;
        const found = suggestions.find((s) => s.rep_index === target.repIndex);
        const spec = found ? specOf(found.suggestion) : null;
        return spec && !spec.allow_tags ? found : null;
    }

    function renderSuggestion() {
        const found = currentSuggestion();
        els.suggest.hidden = !found;
        if (!found) return;
        const pill = document.createElement('span');
        pill.className = `lv-pill lv-pill-${found.suggestion}`;
        pill.textContent = labelName(found.suggestion);
        const reason = document.createElement('span');
        reason.textContent = `${found.reason} (${found.confidence} confidence)`;
        els.suggestText.replaceChildren(pill, reason);
        els.suggestAccept.textContent = `Accept ${labelName(found.suggestion)}`;
        els.suggestAccept.disabled = busy;
    }

    function renderEditor() {
        const open = !!target;
        els.hint.hidden = open;
        els.body.hidden = !open;
        if (!open) {
            els.info.textContent = '';
            els.suggest.hidden = true;
            return;
        }
        const parts = [];
        if (target.repIndex !== null && target.repIndex !== undefined) parts.push(`Rep ${target.repIndex + 1}`);
        parts.push(`frames ${range(target.start, target.end)}`);
        els.info.textContent = parts.join('  \u00b7  ');

        const spec = choice ? specOf(choice) : null;
        const needsTags = !!(spec && spec.allow_tags);
        const locked = busy || !!target.blocked;
        els.choices.querySelectorAll('.lv-choice').forEach((b) => {
            b.classList.toggle('is-active', b.dataset.label === choice);
            b.disabled = locked;
        });
        els.tagsWrap.hidden = !needsTags;
        els.tags.querySelectorAll('.lv-chip').forEach((c) => {
            c.classList.toggle('is-active', tags.has(c.dataset.tag));
            c.disabled = locked;
        });
        els.save.hidden = !needsTags || !!target.blocked;
        els.save.disabled = busy;
        els.del.hidden = !editing;
        els.del.disabled = busy;
        els.note.disabled = locked;
        renderSuggestion();
    }

    function cell(text) {
        const td = document.createElement('td');
        td.textContent = text;
        return td;
    }

    function actionButton(action, icon, title, danger) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `lv-icon-btn${danger ? ' is-danger' : ''}`;
        button.dataset.action = action;
        button.title = title;
        button.setAttribute('aria-label', title);
        const i = document.createElement('i');
        i.className = `fas fa-${icon}`;
        button.appendChild(i);
        return button;
    }

    function renderTable(errorText) {
        els.count.textContent = String(labels.length);
        els.table.replaceChildren();
        if (!labels.length) {
            const tr = document.createElement('tr');
            const td = cell(errorText || 'No labels yet');
            td.colSpan = 6;
            td.className = 'lv-table-empty';
            tr.appendChild(td);
            els.table.appendChild(tr);
            return;
        }
        labels.forEach((row) => {
            const tr = document.createElement('tr');
            tr.dataset.id = String(row.id);
            if (editing && editing.id === row.id) tr.classList.add('is-active');
            tr.appendChild(cell(range(row.start_frame, row.end_frame)));
            tr.appendChild(cell(row.rep_index === null || row.rep_index === undefined ? '-' : String(row.rep_index + 1)));
            const pillCell = document.createElement('td');
            const pill = document.createElement('span');
            pill.className = `lv-pill lv-pill-${row.label}`;
            pill.textContent = labelName(row.label);
            pillCell.appendChild(pill);
            tr.appendChild(pillCell);
            tr.appendChild(cell((row.tags || []).map(tagName).join(', ') || '-'));
            const note = row.note || '';
            const noteCell = cell(note.length > 60 ? `${note.slice(0, 57)}...` : note || '-');
            if (note) noteCell.title = note;
            tr.appendChild(noteCell);
            const actions = document.createElement('td');
            actions.className = 'lv-actions';
            actions.append(actionButton('edit', 'pen', 'Edit label', false),
                           actionButton('delete', 'trash', 'Delete label', true));
            tr.appendChild(actions);
            els.table.appendChild(tr);
        });
    }

    function reset() {
        target = null;
        editing = null;
        choice = null;
        tags = new Set();
        busy = false;
        els.note.value = '';
        setMsg('');
        renderEditor();
        renderTable();
    }

    function openLabel(row) {
        target = { start: row.start_frame, end: row.end_frame, repIndex: row.rep_index, blocked: null };
        editing = row;
        choice = row.label;
        tags = new Set(row.tags || []);
        els.note.value = row.note || '';
        setMsg('');
        renderEditor();
        renderTable();
    }

    function select(sel) {
        if (!sel) {
            reset();
            return;
        }
        if (sel.kind === 'label') {
            openLabel(labels.find((l) => l.id === sel.label.id) || sel.label);
            return;
        }
        const same = labels.find((l) => l.start_frame === sel.start && l.end_frame === sel.end);
        if (same) {
            openLabel(same);
            return;
        }
        const clash = labels.find((l) => l.start_frame < sel.end && l.end_frame > sel.start);
        target = {
            start: sel.start,
            end: sel.end,
            repIndex: Number.isInteger(sel.rep) ? sel.rep - 1 : null,
            blocked: clash || null,
        };
        editing = null;
        choice = null;
        tags = new Set();
        els.note.value = '';
        setMsg(clash
            ? `This overlaps your label at frames ${range(clash.start_frame, clash.end_frame)}. Select that label to edit or delete it.`
            : '', !!clash);
        renderEditor();
        renderTable();
    }

    async function save() {
        if (busy || !target || !choice || target.blocked) return;
        const spec = specOf(choice);
        if (!spec) return;
        const list = spec.allow_tags ? Array.from(tags) : [];
        if (spec.require_tags && !list.length) {
            setMsg('Pick at least one compensation tag.', true);
            return;
        }
        const body = {
            start_frame: target.start,
            end_frame: target.end,
            label: choice,
            tags: list,
            note: els.note.value.trim() || null,
            rep_index: target.repIndex === undefined ? null : target.repIndex,
        };
        const mine = token;
        busy = true;
        setMsg('Saving...');
        renderEditor();
        try {
            const row = editing
                ? await request('PUT', `${base()}/${editing.id}`, body)
                : await request('POST', base(), body);
            if (mine !== token) return;
            const at = labels.findIndex((l) => l.id === row.id);
            if (at >= 0) labels[at] = row; else labels.push(row);
            labels.sort((a, b) => a.start_frame - b.start_frame || a.id - b.id);
            editing = row;
            tags = new Set(row.tags || []);
            setMsg('Saved.');
            notify();
            highlight(row);
        } catch (err) {
            if (mine === token) setMsg(err.message, true);
        } finally {
            if (mine === token) {
                busy = false;
                renderEditor();
                renderTable();
            }
        }
    }

    async function remove(row) {
        if (busy || !row) return;
        if (!window.confirm('Delete this label?')) return;
        const mine = token;
        busy = true;
        renderEditor();
        try {
            await request('DELETE', `${base()}/${row.id}`);
            if (mine !== token) return;
            labels = labels.filter((l) => l.id !== row.id);
            notify();
            if (editing && editing.id === row.id) {
                reset();
                highlight(null);
            }
        } catch (err) {
            if (mine === token) setMsg(err.message, true);
        } finally {
            if (mine === token) {
                busy = false;
                renderEditor();
                renderTable();
            }
        }
    }

    function pickLabel(id) {
        if (busy || !target || target.blocked) return;
        const spec = specOf(id);
        if (!spec) return;
        choice = id;
        if (!spec.allow_tags) tags = new Set();
        setMsg('');
        renderEditor();
        if (!spec.allow_tags) save();
    }

    function toggleTag(id) {
        if (busy || !target || target.blocked) return;
        if (tags.has(id)) tags.delete(id); else tags.add(id);
        setMsg('');
        renderEditor();
    }

    async function setRecording(detail) {
        const mine = ++token;
        suggestions = [];
        rec = detail ? { patientId: detail.patientId, recordingId: detail.recordingId } : null;
        labels = [];
        reset();
        notify();
        if (!rec) return;
        try {
            await ensureDefs();
            const data = await request('GET', base());
            if (mine !== token) return;
            labels = data.labels || [];
            renderTable();
            notify();
        } catch (err) {
            if (mine === token) renderTable(`Could not load labels: ${err.message}`);
            return;
        }
        try {
            const data = await request('GET', suggestUrl());
            if (mine !== token) return;
            suggestions = (data.items || []).filter((i) => i.suggestion && i.metrics && i.metrics.frames >= MIN_SUGGEST_FRAMES);
            renderEditor();
        } catch (err) {
            suggestions = [];
        }
    }

    els.choices.addEventListener('click', (event) => {
        const button = event.target.closest('.lv-choice');
        if (button) pickLabel(button.dataset.label);
    });
    els.tags.addEventListener('click', (event) => {
        const chip = event.target.closest('.lv-chip');
        if (chip) toggleTag(chip.dataset.tag);
    });
    els.suggestAccept.addEventListener('click', () => {
        const found = currentSuggestion();
        if (found) pickLabel(found.suggestion);
    });
    els.save.addEventListener('click', save);
    els.del.addEventListener('click', () => remove(editing));
    els.cancel.addEventListener('click', () => {
        reset();
        highlight(null);
    });
    els.note.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            const spec = choice ? specOf(choice) : null;
            if (spec) save();
        }
    });
    els.table.addEventListener('click', (event) => {
        const tr = event.target.closest('tr[data-id]');
        if (!tr) return;
        const row = labels.find((l) => String(l.id) === tr.dataset.id);
        if (!row) return;
        const action = event.target.closest('[data-action]');
        if (action && action.dataset.action === 'delete') {
            remove(row);
            return;
        }
        openLabel(row);
        highlight(row);
        if (hooks.seek) hooks.seek(row.start_frame);
    });

    document.addEventListener('labeling:recording', (event) => setRecording(event.detail));

    window.LabelingLabels = {
        init(next) {
            hooks = Object.assign({ onLabels: null, setSelection: null, seek: null }, next);
            const current = window.LabelingPage && window.LabelingPage.current();
            if (current && (!rec || rec.recordingId !== current.recordingId)) setRecording(current);
            else if (rec) notify();
        },
        select,
        labels: () => labels.slice(),
    };
})();