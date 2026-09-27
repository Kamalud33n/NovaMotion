/**
 * Admin Panel - shared helpers
 * Loaded after /static/app.js (which already provides showToast, apiRequest,
 * formatDate, formatDateTime, etc.) on every page under /admin/*.
 */

// ============================================
// Sidebar toggle (mobile) + logout — identical pattern to the
// clinical pages (dashboard.html / patients.html use the same inline code).
// ============================================
function initAdminChrome() {
    document.getElementById('menuToggle')?.addEventListener('click', () =>
        document.querySelector('.sidebar').classList.toggle('active'));
}

function handleLogout() {
    if (!confirm('Are you sure you want to logout?')) return;
    fetch('/api/auth/logout', { method: 'POST', credentials: 'same-origin' })
        .catch(() => {})
        .finally(() => { window.location.href = '/login'; });
}

// ============================================
// Small formatting/escaping helpers reused across admin pages
// ============================================
function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

function initials(name) {
    return (name || '?').split(' ').filter(Boolean).slice(0, 2)
        .map(w => w[0].toUpperCase()).join('');
}

function timeAgo(isoString) {
    if (!isoString) return '—';
    const d = new Date(isoString);
    const diffSec = Math.floor((Date.now() - d.getTime()) / 1000);
    if (diffSec < 60) return 'just now';
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
    if (diffSec < 2592000) return `${Math.floor(diffSec / 86400)}d ago`;
    return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

// Status / role label + badge-class helpers — one place so every page
// (users, approvals, tickets, dashboard) renders the same labels.
const STATUS_LABELS = {
    pending: 'Pending', approved: 'Approved', rejected: 'Rejected', suspended: 'Suspended',
    open: 'Open', in_progress: 'In Progress', resolved: 'Resolved', closed: 'Closed',
};

function statusBadge(status) {
    const label = STATUS_LABELS[status] || status;
    return `<span class="status-badge ${status}">${label}</span>`;
}

function roleBadge(role) {
    const icon = role === 'admin' ? 'fa-user-shield' : role === 'doctor' ? 'fa-user-md' : 'fa-user-nurse';
    return `<span class="role-badge ${role}"><i class="fas ${icon}"></i> ${role}</span>`;
}

document.addEventListener('DOMContentLoaded', initAdminChrome);
