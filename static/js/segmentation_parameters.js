// Segmentation parameters page: validation mode, series, nnU-Net settings, live summary and validation.
(function () {
    'use strict';

    var form = document.getElementById('params-form');
    var networks = Array.prototype.slice.call(form.querySelectorAll('input[type=checkbox][data-group]'));
    var series = Array.prototype.slice.call(form.querySelectorAll('input[data-series]'));
    var reference = document.getElementById('reference');
    var seriesError = document.getElementById('series-error');
    var kFolds = document.getElementById('k_folds');
    var summary = document.getElementById('run-summary');
    var modelsError = document.getElementById('models-error');
    var runButton = document.getElementById('run-btn');
    var foldsFeedback = document.getElementById('k-folds-feedback');
    var foldsHint = foldsFeedback.textContent;
    var maxFolds = parseInt(form.getAttribute('data-max-folds'), 10);

    function selected() { return networks.filter(function (n) { return n.checked; }); }
    function chosenSeries() { return series.filter(function (s) { return s.checked; }).map(function (s) { return s.value; }); }
    function mode() { return form.querySelector('input[name=validation]:checked').value; }

    function foldsProblem() {
        if (mode() !== 'kfold') return null;
        var k = parseInt(kFolds.value, 10);
        if (isNaN(k) || k < 2) return 'Use at least 2 folds.';
        if (k > maxFolds) return 'At most ' + maxFolds + ' folds: one case (or patient) per fold at least.';
        return null;
    }

    function update() {
        var kfold = mode() === 'kfold';
        document.getElementById('kfold-settings').classList.toggle('d-none', !kfold);
        document.getElementById('holdout-settings').classList.toggle('d-none', kfold);
        var nnunet = selected().some(function (n) { return n.getAttribute('data-group') === 'nnunet'; });
        document.getElementById('nnunet-settings').classList.toggle('d-none', !nnunet);
        if (reference) {
            var chosen = chosenSeries();
            Array.prototype.forEach.call(reference.options, function (option) {
                option.disabled = chosen.length > 0 && chosen.indexOf(option.value) < 0;
            });
            if (chosen.length && chosen.indexOf(reference.value) < 0) reference.value = chosen[0];
            seriesError.classList.toggle('d-none', chosen.length > 0);
        }
        var n = selected().length;
        modelsError.classList.toggle('d-none', n > 0);
        var trainings = n * (kfold ? (parseInt(kFolds.value, 10) || 0) + 1 : 1);
        var parts = [n + ' network' + (n === 1 ? '' : 's'),
                     kfold ? (kFolds.value || '?') + '-fold cross-validation' : 'hold-out validation',
                     trainings + ' training' + (trainings === 1 ? '' : 's')];
        if (series.length) parts.splice(1, 0, chosenSeries().length + ' series');
        summary.textContent = parts.join(' · ');
        var problem = foldsProblem();
        kFolds.classList.toggle('is-invalid', !!problem);
        foldsFeedback.classList.toggle('text-danger', !!problem);
        foldsFeedback.textContent = problem || foldsHint;
    }

    form.querySelectorAll('[data-select-group]').forEach(function (button) {
        button.addEventListener('click', function () {
            var group = button.getAttribute('data-select-group');
            var value = button.getAttribute('data-value') === '1';
            networks.forEach(function (n) { if (n.getAttribute('data-group') === group) n.checked = value; });
            update();
        });
    });
    form.addEventListener('input', update);
    form.addEventListener('change', update);

    form.addEventListener('submit', function (e) {
        update();
        var invalid = series.length && !chosenSeries().length ? series[0]
            : (!selected().length ? networks[0] : (foldsProblem() ? kFolds : null));
        if (invalid) {
            e.preventDefault();
            invalid.focus();
            invalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
            return;
        }
        runButton.disabled = true;
        runButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" aria-hidden="true"></span>Starting…';
    });

    update();
})();
