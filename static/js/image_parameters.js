// Image parameters page: dependent controls, live summary and validation.
(function () {
    'use strict';

    var form = document.getElementById('params-form');
    var networks = Array.prototype.slice.call(form.querySelectorAll('input[type=checkbox][data-group]'));
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
    function mode() { return form.querySelector('input[name=mode]:checked').value; }

    function foldsProblem() {
        var k = parseInt(kFolds.value, 10);
        var minClass = parseInt(kFolds.getAttribute('data-min-class'), 10) || 0;
        if (isNaN(k) || k < 2) return 'Use at least 2 folds.';
        if (k > 20) return 'Use at most 20 folds.';
        if (minClass && k > minClass) return 'The smallest class has ' + minClass + ' training images: use at most ' + minClass + ' folds.';
        return null;
    }

    function update() {
        var finetune = mode() === 'finetune';
        finetuneSettings.classList.toggle('d-none', !finetune);
        finetuneSettings.querySelectorAll('input, select').forEach(function (el) { el.disabled = !finetune; });
        metricHelp.textContent = metric.disabled ? '' : metric.selectedOptions[0].getAttribute('data-description');

        var n = selected().length;
        modelsError.classList.toggle('d-none', n > 0);
        var parts = [n + ' network' + (n === 1 ? '' : 's'), finetune ? 'fine-tuning' : 'feature extraction',
                     (kFolds.value || '?') + '-fold cross-validation'];
        if (finetune) parts.push('up to ' + (document.getElementById('epochs').value || '?') + ' epochs');
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
        var invalid = !selected().length ? networks[0] : (foldsProblem() ? kFolds : null);
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
