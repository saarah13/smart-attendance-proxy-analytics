/**
 * main.js — SGBIT Attendance System
 * Global JS: clock, sidebar toggle, auto-dismiss flash messages
 */

// ── Live Clock ───────────────────────────────────────────────────────
function updateClock() {
  const el = document.getElementById('live-clock');
  if (!el) return;
  const now = new Date();
  el.textContent = now.toLocaleString('en-IN', {
    weekday: 'short',
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: true
  });
}
updateClock();
setInterval(updateClock, 1000);

// ── Sidebar Toggle ───────────────────────────────────────────────────
function toggleSidebar() {
  document.getElementById('sidebar')?.classList.toggle('open');
}
// Close sidebar on outside click (mobile)
document.addEventListener('click', function(e) {
  const sidebar = document.getElementById('sidebar');
  const toggle  = document.getElementById('sidebar-toggle');
  if (sidebar && sidebar.classList.contains('open') &&
      !sidebar.contains(e.target) && e.target !== toggle) {
    sidebar.classList.remove('open');
  }
});

// ── Auto-dismiss Flash Messages ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.flash').forEach(function(el) {
    setTimeout(() => {
      el.style.transition = 'opacity 0.5s';
      el.style.opacity    = '0';
      setTimeout(() => el.remove(), 500);
    }, 5000);
  });
});

// ── Confirm Dangerous Actions ────────────────────────────────────────
document.querySelectorAll('[data-confirm]').forEach(function(el) {
  el.addEventListener('click', function(e) {
    if (!confirm(el.dataset.confirm)) e.preventDefault();
  });
});
