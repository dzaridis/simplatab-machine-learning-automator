// 3D image parameters page: series and reference, dependent controls, live summary and validation.
(function () {
    'use strict';

    var form = document.getElementById('params-form');
    var networks = Array.prototype.slice.call(form.querySelectorAll('input[type=checkbox][data-group]'));
    var series = Array.prototype.slice.call(form.querySelectorAll('input[data-series]'));
    var reference = document.getElementById('reference');
    var seriesError = document.getElementById('series-error');
    var kFolds = document.getElementById('k_folds');
    var metric = document.getElementById('optimization_metric');
    var metricHelp = document.getElementById('metric-help');
    var finetuneSettings = document.getElementById('finetune-settings');
    var summary = document.getElementById('run-summary');
    var modelsError = document.getElementById('models-error');
    var runButton = document.getElementById('run-btn');
    var foldsFeedback = document.getElementById('k-folds-feedback');
    var foldsHint = foldsFeedback.textContent;

    function selected() { return networks.filter(function (n) { return n.checked; }); }
    function chosenSeries() { return series.filter(function (s) { return s.checked; }).map(function (s) { return s.value; }); }
    function mode() { return form.querySelector('input[name=mode]:checked').value; }

    function foldsProblem() {
        var k = parseInt(kFolds.value, 10);
        var minClass = parseInt(kFolds.getAttribute('data-min-class'), 10) || 0;
        if (isNaN(k) || k < 2) return 'Use at least 2 folds.';
        if (k > 20) return 'Use at most 20 folds.';
        if (minClass && k > minClass) return 'The smallest class has ' + minClass + ' training patients: use at most ' + minClass + ' folds.';
        return null;
    }

    function updateReference() {
        if (!reference) return;
        var chosen = chosenSeries();
        // Only chosen series can be the reference; keep the current one when still chosen
        Array.prototype.forEach.call(reference.options, function (option) {
            option.disabled = chosen.length > 0 && chosen.indexOf(option.value) < 0;
        });
        if (chosen.length && chosen.indexOf(reference.value) < 0) reference.value = chosen[0];
    }

    function update() {
        var finetune = mode() === 'finetune';
        finetuneSettings.classList.toggle('d-none', !finetune);
        finetuneSettings.querySelectorAll('input, select').forEach(function (el) { el.disabled = !finetune; });
        metricHelp.textContent = metric.disabled ? '' : metric.selectedOptions[0].getAttribute('data-description');
        updateReference();

        var n = selected().length;
        modelsError.classList.toggle('d-none', n > 0);
        var channels = series.length ? chosenSeries().length : 1;
        if (seriesError) seriesError.classList.toggle('d-none', channels > 0);
        var shape = document.getElementById('shape').selectedOptions[0].textContent.replace(' (default)', '');
        var parts = [n + ' network' + (n === 1 ? '' : 's'), channels + ' series', shape,
                     finetune ? 'fine-tuning' : 'feature extraction', (kFolds.value || '?') + '-fold cross-validation'];
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
