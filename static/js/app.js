// Shared behaviour: colour theme, toasts, tooltips and popovers.
(function () {
    'use strict';

    var media = window.matchMedia('(prefers-color-scheme: dark)');

    function savedTheme() {
        try { return localStorage.getItem('sb-theme') || 'auto'; } catch (e) { return 'auto'; }
    }

    function applyTheme(choice) {
        var dark = choice === 'dark' || (choice === 'auto' && media.matches);
        document.documentElement.setAttribute('data-bs-theme', dark ? 'dark' : 'light');
        document.querySelectorAll('[data-theme-value]').forEach(function (button) {
            var active = button.getAttribute('data-theme-value') === choice;
            button.classList.toggle('active', active);
            button.setAttribute('aria-pressed', active ? 'true' : 'false');
        });
    }

    document.querySelectorAll('[data-theme-value]').forEach(function (button) {
        button.addEventListener('click', function () {
            var choice = button.getAttribute('data-theme-value');
            try { localStorage.setItem('sb-theme', choice); } catch (e) {}
            applyTheme(choice);
        });
    });
    media.addEventListener('change', function () { if (savedTheme() === 'auto') applyTheme('auto'); });
    applyTheme(savedTheme());

    document.querySelectorAll('.toast').forEach(function (element) {
        bootstrap.Toast.getOrCreateInstance(element).show();
    });
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (element) {
        new bootstrap.Tooltip(element);
    });
    document.querySelectorAll('[data-bs-toggle="popover"]').forEach(function (element) {
        new bootstrap.Popover(element, { trigger: 'focus', html: false });
    });
})();
