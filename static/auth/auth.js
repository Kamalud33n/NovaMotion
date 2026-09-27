/* Auth pages — login / register / forgot / reset.
   The server validates everything again; this is just for quick feedback. */
(function () {
    'use strict';

    const $ = (sel, root) => (root || document).querySelector(sel);

    async function send(url, body, method) {
        try {
            const res = await fetch(url, {
                method: method || 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'same-origin',
                body: JSON.stringify(body),
            });
            let data = {};
            try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
            return { ok: res.ok, status: res.status, data };
        } catch (_) {
            return { ok: false, status: 0, data: { detail: 'Cannot reach the server. Check the connection and try again.' } };
        }
    }

    function messageFrom(result, fallback) {
        const d = result.data && result.data.detail;
        if (typeof d === 'string') return d;
        if (Array.isArray(d)) return 'Please check the form and try again.';
        return fallback || 'Something went wrong. Please try again.';
    }

    function showAlert(type, text) {
        const el = $('#authAlert');
        if (!el) return;
        el.className = 'auth-alert show ' + type;
        el.textContent = text;
    }
    function clearAlert() {
        const el = $('#authAlert');
        if (el) { el.className = 'auth-alert'; el.textContent = ''; }
    }

    function busy(form, isBusy, idleLabel) {
        const btn = $('button[type="submit"]', form);
        if (!btn) return;
        btn.disabled = isBusy;
        btn.dataset.idle = btn.dataset.idle || btn.textContent;
        btn.textContent = isBusy ? 'Please wait…' : (idleLabel || btn.dataset.idle);
    }

    function formData(form) {
        const out = {};
        new FormData(form).forEach((v, k) => { out[k] = v; });
        return out;
    }

    // Show / hide password buttons
    document.querySelectorAll('.pw-toggle').forEach((btn) => {
        btn.addEventListener('click', () => {
            const input = btn.parentElement.querySelector('input');
            const show = input.type === 'password';
            input.type = show ? 'text' : 'password';
            btn.textContent = show ? 'Hide' : 'Show';
        });
    });

    const minLen = parseInt(document.body.dataset.pwMin || '8', 10);
    function passwordProblem(pw, confirm) {
        if (pw.length < minLen) return 'Password must be at least ' + minLen + ' characters.';
        if (!/[A-Za-z]/.test(pw) || !/[0-9]/.test(pw)) return 'Password must contain at least one letter and one number.';
        if (pw !== confirm) return 'Passwords do not match.';
        return null;
    }

    // ── Login / register (tabs) ─────────────────────────────────────────
    const loginForm = $('#loginForm');
    const registerForm = $('#registerForm');
    const otpForm = $('#otpForm');
    if (loginForm && registerForm) {
        const tabs = document.querySelectorAll('.auth-tab');
        function selectTab(name) {
            tabs.forEach((t) => {
                const on = t.dataset.tab === name;
                t.classList.toggle('active', on);
                t.setAttribute('aria-selected', on ? 'true' : 'false');
            });
            loginForm.hidden = name !== 'login';
            registerForm.hidden = name !== 'register';
            if (otpForm) otpForm.hidden = true;
            clearAlert();
            $('#authTitle').textContent = name === 'login' ? 'Welcome back' : 'Create your account';
            $('#authLead').textContent = name === 'login'
                ? 'Sign in to continue to your dashboard.'
                : 'New accounts are reviewed by an administrator before they can sign in.';
        }
        tabs.forEach((t) => t.addEventListener('click', () => selectTab(t.dataset.tab)));
        if (location.hash === '#register') selectTab('register');

        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            clearAlert();
            const body = formData(loginForm);
            if (!body.email || !body.password) return showAlert('error', 'Enter your email and password.');
            busy(loginForm, true);
            const r = await send('/api/auth/login', body);
            if (r.ok) {
                window.location.href = r.data.redirect || '/dashboard';
                return;
            }
            busy(loginForm, false);
            showAlert('error', messageFrom(r, 'Sign in failed.'));
        });

        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            clearAlert();
            const body = formData(registerForm);
            const problem = passwordProblem(body.password || '', body.confirm_password || '');
            if (problem) return showAlert('error', problem);
            busy(registerForm, true);
            const r = await send('/api/auth/register', body);
            busy(registerForm, false);
            if (r.ok) {
                registerForm.hidden = true;
                if (otpForm) {
                    otpForm.hidden = false;
                    otpForm.dataset.email = r.data.email || body.email;
                    $('#otpEmail').textContent = r.data.email || body.email;
                    $('#otpCode').value = '';
                }
                $('#authTitle').textContent = 'Check your email';
                showAlert('success', r.data.message || 'Enter the code we emailed you.');
            } else {
                showAlert('error', messageFrom(r, 'Registration failed.'));
            }
        });

        // ── Verify OTP ───────────────────────────────────────────────────
        if (otpForm) {
            otpForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                clearAlert();
                const email = otpForm.dataset.email;
                const otp = $('#otpCode').value.trim();
                if (!otp) return showAlert('error', 'Enter the code we emailed you.');
                busy(otpForm, true);
                const r = await send('/api/auth/register/verify-otp', { email, otp });
                busy(otpForm, false);
                if (r.ok) {
                    otpForm.hidden = true;
                    registerForm.reset();
                    selectTab('login');
                    showAlert('success', r.data.message || 'Registration submitted.');
                } else {
                    showAlert('error', messageFrom(r, 'Verification failed.'));
                }
            });

            const resendBtn = $('#otpResendBtn');
            if (resendBtn) {
                resendBtn.addEventListener('click', async () => {
                    const email = otpForm.dataset.email;
                    resendBtn.disabled = true;
                    const r = await send('/api/auth/register/resend-otp', { email });
                    resendBtn.disabled = false;
                    if (r.ok) showAlert('success', r.data.message || 'Code resent.');
                    else showAlert('error', messageFrom(r, 'Could not resend code.'));
                });
            }
        }
    }

    // ── Forgot password ─────────────────────────────────────────────────
    const forgotForm = $('#forgotForm');
    if (forgotForm) {
        forgotForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            clearAlert();
            const body = formData(forgotForm);
            if (!body.email) return showAlert('error', 'Enter your email address.');
            busy(forgotForm, true);
            const r = await send('/api/auth/forgot-password', body);
            if (r.ok) {
                forgotForm.hidden = true;
                $('#forgotDone').hidden = false;
                $('#forgotDoneText').textContent = r.data.message;
            } else {
                busy(forgotForm, false);
                showAlert('error', messageFrom(r));
            }
        });
    }

    // ── Reset password ──────────────────────────────────────────────────
    const resetForm = $('#resetForm');
    if (resetForm) {
        resetForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            clearAlert();
            const body = formData(resetForm);
            const problem = passwordProblem(body.password || '', body.confirm_password || '');
            if (problem) return showAlert('error', problem);
            busy(resetForm, true);
            const r = await send('/api/auth/reset-password', body);
            if (r.ok) {
                resetForm.hidden = true;
                $('#resetDone').hidden = false;
            } else {
                busy(resetForm, false);
                showAlert('error', messageFrom(r));
            }
        });
    }
})();