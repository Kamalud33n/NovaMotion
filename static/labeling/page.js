/* Labeling page shell: sidebar, logout, patient and recording pickers, recording info card.
   When a recording is chosen it fires `labeling:recording` on document with
   detail = { patientId, recordingId, meta } (detail = null when the choice is cleared).
   Playback and timeline scripts listen for that event; LabelingPage.current() returns the
   same detail for scripts that load after it fired. */
(function () {
    'use strict';

    const $ = (id) => document.getElementById(id);
    const els = {
        patient: $('lvPatient'),
        recording: $('lvRecording'),
        refresh: $('lvRefresh'),
        defsNote: $('lvDefsNote'),
        defsNoteText: $('lvDefsNoteText'),
        empty: $('lvEmpty'),
        emptyIcon: $('lvEmptyIcon'),
        emptyTitle: $('lvEmptyTitle'),
        emptyText: $('lvEmptyText'),
        workspace: $('lvWorkspace'),
        meta: $('lvMeta'),
        flags: $('lvFlags'),
    };

    let current = null;
    let listToken = 0;
    let metaToken = 0;

    async function getJSON(url) {
        const res = await fetch(url, { credentials: 'same-origin' });
        if (res.status === 401) {
            window.location.href = '/login';
            throw new Error('Not signed in');
        }
        if (!res.ok) {
            let message = `Request failed (${res.status})`;
            try { message = (await res.json()).detail || message; } catch (e) { /* keep default */ }
            throw new Error(message);
        }
        return res.json();
    }

    function fmtDate(iso) {
        const d = iso ? new Date(iso) : null;
        return d && !isNaN(d) ? d.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '-';
    }

    function fmtDuration(seconds) {
        if (typeof seconds !== 'number' || !isFinite(seconds)) return '-';
        const total = Math.round(seconds);
        return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
    }

    function showEmpty(icon, title, text) {
        els.emptyIcon.className = `fas fa-${icon}`;
        els.emptyTitle.textContent = title;
        els.emptyText.textContent = text;
        els.empty.hidden = false;
        els.workspace.hidden = true;
    }

    function showWorkspace() {
        els.empty.hidden = true;
        els.workspace.hidden = false;
    }

    function setOptions(select, placeholder, items, toValue, toText) {
        select.replaceChildren();
        select.add(new Option(placeholder, ''));
        items.forEach((item) => select.add(new Option(toText(item), toValue(item))));
    }

    function syncUrl() {
        const params = new URLSearchParams();
        if (els.patient.value) params.set('patient', els.patient.value);
        if (els.recording.value) params.set('recording', els.recording.value);
        const query = params.toString();
        history.replaceState(null, '', query ? `?${query}` : window.location.pathname);
    }

    function announce(detail) {
        current = detail;
        document.dispatchEvent(new CustomEvent('labeling:recording', { detail }));
    }

    function renderMeta(meta) {
        const kind = meta.kind === 'hands' ? 'Hands' : 'Body pose';
        const mode = (meta.extra && meta.extra.mode) || null;
        const rows = [
            ['Exercise / game', meta.label || '-'],
            ['Type', kind],
            ['Mode', mode ? mode.charAt(0).toUpperCase() + mode.slice(1) : '-'],
            ['Side', meta.side ? meta.side.charAt(0).toUpperCase() + meta.side.slice(1) : '-'],
            ['Recorded', fmtDate(meta.started_at)],
            ['Duration', fmtDuration(meta.duration_seconds)],
            ['Frames', meta.frame_count != null ? String(meta.frame_count) : '-'],
            ['Average FPS', meta.avg_fps != null ? String(meta.avg_fps) : '-'],
            ['Session', meta.session_id || 'Not linked to a saved session'],
        ];
        els.meta.replaceChildren();
        rows.forEach(([name, value]) => {
            const box = document.createElement('div');
            const dt = document.createElement('dt');
            const dd = document.createElement('dd');
            dt.textContent = name;
            dd.textContent = value;
            box.append(dt, dd);
            els.meta.appendChild(box);
        });

        const flags = [];
        if (meta.truncated) flags.push('Recording was cut short because it reached the size limit.');
        if (meta.dropped_frames) flags.push(`${meta.dropped_frames} frame(s) were dropped while recording.`);
        if (meta.end_reason && meta.end_reason !== 'saved') flags.push(`Ended: ${meta.end_reason}.`);
        els.flags.replaceChildren();
        flags.forEach((text) => {
            const flag = document.createElement('div');
            const icon = document.createElement('i');
            const span = document.createElement('span');
            flag.className = 'lv-flag';
            icon.className = 'fas fa-triangle-exclamation';
            span.textContent = text;
            flag.append(icon, span);
            els.flags.appendChild(flag);
        });
    }

    async function openRecording() {
        const patientId = els.patient.value;
        const recordingId = els.recording.value;
        syncUrl();
        if (!patientId || !recordingId) {
            announce(null);
            if (patientId && !els.recording.disabled) {
                showEmpty('film', 'Select a recording', 'Choose one of this patient\'s saved recordings above.');
            }
            return;
        }
        const token = ++metaToken;
        showEmpty('spinner fa-spin', 'Loading recording...', '');
        try {
            const meta = await getJSON(`/api/recordings/${encodeURIComponent(patientId)}/${encodeURIComponent(recordingId)}`);
            if (token !== metaToken) return;
            renderMeta(meta);
            showWorkspace();
            announce({ patientId, recordingId, meta });
        } catch (err) {
            if (token !== metaToken) return;
            announce(null);
            showEmpty('circle-exclamation', 'Could not open this recording', err.message);
        }
    }

    async function loadRecordings(preselect) {
        const patientId = els.patient.value;
        const token = ++listToken;
        ++metaToken;
        announce(null);
        setOptions(els.recording, 'Select recording', [], () => '', () => '');
        els.recording.disabled = true;
        if (!patientId) {
            showEmpty('user', 'Select a patient', 'Pick a patient above to see their saved recordings.');
            syncUrl();
            return;
        }
        showEmpty('spinner fa-spin', 'Loading recordings...', '');
        try {
            const data = await getJSON(`/api/recordings?patient_id=${encodeURIComponent(patientId)}&limit=500`);
            if (token !== listToken) return;
            const rows = data.recordings || [];
            if (!rows.length) {
                showEmpty('film', 'No saved recordings',
                    'Recordings are saved only for patients who gave keypoint consent, once a session has finished.');
                syncUrl();
                return;
            }
            setOptions(els.recording, 'Select recording', rows,
                (r) => r.recording_id,
                (r) => `${r.label || 'Recording'} - ${fmtDate(r.ended_at || r.started_at)} - ${fmtDuration(r.duration_seconds)}`);
            els.recording.disabled = false;
            if (preselect && rows.some((r) => r.recording_id === preselect)) {
                els.recording.value = preselect;
                openRecording();
            } else {
                showEmpty('film', 'Select a recording', 'Choose one of this patient\'s saved recordings above.');
                syncUrl();
            }
        } catch (err) {
            if (token !== listToken) return;
            showEmpty('circle-exclamation', 'Could not load recordings', err.message);
        }
    }

    async function loadPatients(preselectPatient, preselectRecording) {
        try {
            const patients = await getJSON('/api/patients');
            setOptions(els.patient, 'Select patient', patients, (p) => p.id, (p) => p.name);
            if (!patients.length) {
                showEmpty('users', 'No patients yet', 'Add a patient first; recordings belong to a patient.');
                return;
            }
            if (preselectPatient && patients.some((p) => p.id === preselectPatient)) {
                els.patient.value = preselectPatient;
                await loadRecordings(preselectRecording);
            } else {
                showEmpty('user', 'Select a patient', 'Pick a patient above to see their saved recordings.');
            }
        } catch (err) {
            showEmpty('circle-exclamation', 'Could not load patients', err.message);
        }
    }

    async function loadDefinitionsNote() {
        try {
            const defs = await getJSON('/api/label-definitions');
            const status = String(defs.status || '');
            if (/draft/i.test(status) || /draft/i.test(String(defs.version || ''))) {
                els.defsNoteText.textContent =
                    `Label definitions ${defs.version || ''} are a draft and still need physiotherapist review.`;
                els.defsNote.hidden = false;
            }
        } catch (err) {
            els.defsNote.hidden = true;
        }
    }

    function handleLogout() {
        if (!confirm('Are you sure you want to logout?')) return;
        fetch('/api/auth/logout', { method: 'POST', credentials: 'same-origin' })
            .catch(() => {})
            .finally(() => { window.location.href = '/login'; });
    }
    window.handleLogout = handleLogout;

    window.LabelingPage = { current: () => current };

    document.addEventListener('DOMContentLoaded', () => {
        $('menuToggle')?.addEventListener('click', () =>
            document.querySelector('.sidebar').classList.toggle('active'));
        els.patient.addEventListener('change', () => loadRecordings(''));
        els.recording.addEventListener('change', openRecording);
        els.refresh.addEventListener('click', () => loadRecordings(els.recording.value));

        const params = new URLSearchParams(window.location.search);
        loadDefinitionsNote();
        loadPatients(params.get('patient') || '', params.get('recording') || '');
    });
})();