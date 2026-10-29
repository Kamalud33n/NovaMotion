/* End-of-session rating (Good / Bad / Worst) on the result modal.
   session.js reports the save through SessionRating.saveStarted / saved(id) / saveFailed;
   the rating is sent once the session has an id. It is stored by PATCH /api/sessions/{id}/rating. */
(function () {
    'use strict';

    let state = 'idle';
    let sessionId = null;
    let chosen = null;
    let chain = Promise.resolve();

    const $ = (id) => document.getElementById(id);

    function message(text, isError) {
        const el = $('ratingMsg');
        if (!el) return;
        el.textContent = text || '';
        el.classList.toggle('is-error', !!isError);
    }

    function paint() {
        document.querySelectorAll('#resultRating .rating-btn').forEach((button) => {
            button.classList.toggle('is-active', button.dataset.rating === chosen);
        });
    }

    async function send(value) {
        const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/rating`, {
            method: 'PATCH',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rating: value }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
    }

    function flush() {
        if (!chosen) return;
        if (state === 'failed') {
            message('The session was not saved, so the rating could not be saved.', true);
            return;
        }
        if (state !== 'saved') {
            message('Waiting for the session to finish saving...');
            return;
        }
        const value = chosen;
        chain = chain.then(async () => {
            try {
                await send(value);
                if (chosen === value) message('Saved.');
            } catch (err) {
                if (chosen === value) message('Could not save the rating. Click it again to retry.', true);
            }
        });
    }

    function reset() {
        state = 'idle';
        sessionId = null;
        chosen = null;
        paint();
        message('');
    }

    document.addEventListener('click', (event) => {
        const button = event.target.closest('#resultRating .rating-btn');
        if (!button) return;
        chosen = button.dataset.rating;
        paint();
        flush();
    });

    window.SessionRating = {
        reset,
        saveStarted() { state = 'saving'; },
        saved(id) {
            sessionId = id || null;
            state = sessionId ? 'saved' : 'failed';
            flush();
        },
        saveFailed() {
            state = 'failed';
            flush();
        },
    };
}());