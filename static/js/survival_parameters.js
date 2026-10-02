// Survival parameters page: horizons, columns and models; live summary and checks.
(function () {
    'use strict';

    var form = document.getElementById('params-form');
    var models = Array.prototype.slice.call(form.querySelectorAll('input[type=checkbox][data-group]'));
    var columns = Array.prototype.slice.call(form.querySelectorAll('input[data-column-index]'));
    var horizons = document.getElementById('horizons');
    var feedback = document.getElementById('horizons-feedback');
    var hint = feedback.textContent;
    var kFolds = document.getElementById('k_folds');
    var summary = document.getElementById('run-summary');
    var timeMax = parseFloat(form.getAttribute('data-time-max'));

    function selected() { return models.filter(function (m) { return m.checked; }); }

    function horizonProblem() {
        var parts = horizons.value.split(/[,;]/).map(function (v) { return v.trim(); }).filter(Boolean);
        if (!parts.length) return null;   // empty: the defaults
        if (parts.length > 5) return 'At most 5 horizons.';
        for (var i = 0; i < parts.length; i++) {
            var v = Number(parts[i]);
            if (isNaN(v)) return '"' + parts[i] + '" is not a number.';
            if (v <= 0 || v >= timeMax) return 'Horizons must be between 0 and ' + timeMax + ' (the longest follow-up).';
        }
        return null;
    }

    function update() {
        columns.forEach(function (c) {
            document.getElementById('ignore_' + c.getAttribute('data-column-index')).value = c.checked ? 'false' : 'true';
        });
        var kept = columns.filter(function (c) { return c.checked; }).length;
        document.getElementById('features-error').classList.toggle('d-none', kept > 0);
        var n = selected().length;
        document.getElementById('models-error').classList.toggle('d-none', n > 0);
        var problem = horizonProblem();
        horizons.classList.toggle('is-invalid', !!problem);
        feedback.classList.toggle('text-danger', !!problem);
        feedback.textContent = problem || hint;
        summary.textContent = [n + ' model' + (n === 1 ? '' : 's') + ' + Kaplan-Meier', kept + ' feature' + (kept === 1 ? '' : 's'),
                               (kFolds.value || '?') + '-fold validation', 'horizons ' + (horizons.value || 'default')].join(' · ');
    }

    form.querySelectorAll('[data-select-group]').forEach(function (button) {
        button.addEventListener('click', function () {
            var group = button.getAttribute('data-select-group');
            var on = button.getAttribute('data-value') === '1';
            models.forEach(function (m) { if (m.getAttribute('data-group') === group) m.checked = on; });
            update();
        });
    });
    form.addEventListener('input', update);
    form.addEventListener('change', update);
    form.addEventListener('submit', function (e) {
        update();
        var kept = columns.some(function (c) { return c.checked; });
        var invalid = !selected().length ? models[0] : horizonProblem() ? horizons : !kept ? columns[0] : null;
        if (invalid) {
            e.preventDefault();
            invalid.focus();
            invalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
            return;
        }
        var button = document.getElementById('run-btn');
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" aria-hidden="true"></span>Starting…';
    });
    update();
})();
