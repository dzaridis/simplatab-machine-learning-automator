// Clustering parameters page: number of clusters, columns, validation; live summary and checks.
(function () {
    'use strict';

    var form = document.getElementById('params-form');
    var models = Array.prototype.slice.call(form.querySelectorAll('input[type=checkbox][data-group]'));
    var columns = Array.prototype.slice.call(form.querySelectorAll('input[data-column-index]'));
    var kMin = document.getElementById('k_min');
    var kMax = document.getElementById('k_max');
    var kFixed = document.getElementById('n_clusters');
    var kFeedback = document.getElementById('k-feedback');
    var kHint = kFeedback.textContent;
    var kFolds = document.getElementById('k_folds');
    var validation = document.getElementById('validation');
    var metric = document.getElementById('selection_metric');
    var summary = document.getElementById('run-summary');
    var maxK = parseInt(form.getAttribute('data-max-k'), 10);

    function selected() { return models.filter(function (m) { return m.checked; }); }
    function mode() { return form.querySelector('input[name=n_clusters_mode]:checked').value; }
    function value(input) { return parseInt(input.value, 10); }

    function kProblem() {
        if (mode() === 'auto') {
            if (isNaN(value(kMin)) || value(kMin) < 2) return [kMin, 'Start at k = 2 or more.'];
            if (isNaN(value(kMax)) || value(kMax) < value(kMin)) return [kMax, 'The range must end at or after its start.'];
            if (value(kMax) > maxK) return [kMax, 'At most ' + maxK + ' clusters for this data.'];
        }
        if (mode() === 'fixed' && (isNaN(value(kFixed)) || value(kFixed) < 2 || value(kFixed) > maxK)) {
            return [kFixed, 'Choose between 2 and ' + maxK + ' clusters.'];
        }
        return null;
    }

    function update() {
        var m = mode();
        document.getElementById('k-auto-settings').classList.toggle('d-none', m !== 'auto');
        document.getElementById('k-fixed-settings').classList.toggle('d-none', m !== 'fixed');
        document.getElementById('pca_variance').disabled = !document.getElementById('reduction').checked;
        kFolds.disabled = !validation.checked;
        document.getElementById('validation-value').value = validation.checked ? 'kfold' : 'none';
        var stability = metric.querySelector('[data-needs-validation]');
        stability.disabled = !validation.checked;
        if (!validation.checked && metric.value === stability.value) metric.selectedIndex = 0;
        columns.forEach(function (c) {
            document.getElementById('ignore_' + c.getAttribute('data-column-index')).value = c.checked ? 'false' : 'true';
        });
        var kept = columns.filter(function (c) { return c.checked; }).length;
        document.getElementById('features-error').classList.toggle('d-none', kept > 0);
        var n = selected().length;
        document.getElementById('models-error').classList.toggle('d-none', n > 0);
        var deep = selected().some(function (m) { return m.getAttribute('data-deep') === '1'; });
        document.getElementById('deep-settings').classList.toggle('opacity-50', !deep);

        [kMin, kMax, kFixed].forEach(function (input) { input.classList.remove('is-invalid'); });
        var problem = kProblem();
        if (problem) problem[0].classList.add('is-invalid');
        kFeedback.classList.toggle('text-danger', !!problem && m === 'auto');
        kFeedback.textContent = problem && m === 'auto' ? problem[1] : kHint;

        var k = m === 'classes' ? 'k = number of classes' : m === 'fixed' ? 'k = ' + (kFixed.value || '?') : 'k from ' + (kMin.value || '?') + ' to ' + (kMax.value || '?');
        summary.textContent = [n + ' algorithm' + (n === 1 ? '' : 's'), k, kept + ' column' + (kept === 1 ? '' : 's'),
                               validation.checked ? (kFolds.value || '?') + '-fold validation' : 'no validation'].join(' · ');
    }

    form.querySelectorAll('[data-select-group]').forEach(function (button) {
        button.addEventListener('click', function () {
            var group = button.getAttribute('data-select-group');
            var on = button.getAttribute('data-value') === '1';
            models.forEach(function (m) { if (m.getAttribute('data-group') === group && !m.disabled) m.checked = on; });
            update();
        });
    });
    form.addEventListener('input', update);
    form.addEventListener('change', update);

    form.addEventListener('submit', function (e) {
        update();
        var problem = kProblem();
        var kept = columns.some(function (c) { return c.checked; });
        var invalid = !selected().length ? models[0] : problem ? problem[0] : !kept ? columns[0] : null;
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
