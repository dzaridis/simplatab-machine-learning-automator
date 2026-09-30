// Parameters page: dependent controls, live summary and validation.
(function () {
    'use strict';

    var form = document.getElementById('params-form');
    var models = Array.prototype.slice.call(form.querySelectorAll('input[type=checkbox][data-group]'));
    var kFolds = document.getElementById('k_folds');
    var gridSearch = document.getElementById('grid_search');
    var gsRadios = form.querySelectorAll('input[name=grid_search_type]');
    var gsHelp = document.getElementById('gs-help');
    var corr = document.getElementById('correlation_limit');
    var corrValue = document.getElementById('corr-value');
    var metric = document.getElementById('optimization_metric');
    var metricHelp = document.getElementById('metric-help');
    var bias = document.getElementById('bias_assessment');
    var feature = document.getElementById('feature');
    var summary = document.getElementById('run-summary');
    var modelsError = document.getElementById('models-error');
    var runButton = document.getElementById('run-btn');
    var foldsFeedback = document.getElementById('k-folds-feedback');
    var foldsHint = foldsFeedback.textContent;

    function selectedModels() { return models.filter(function (m) { return m.checked; }); }

    function foldsProblem() {
        var k = parseInt(kFolds.value, 10);
        var minClass = parseInt(kFolds.getAttribute('data-min-class'), 10) || 0;
        if (isNaN(k) || k < 2) return 'Use at least 2 folds.';
        if (k > 20) return 'Use at most 20 folds.';
        if (minClass && k > minClass) return 'The smallest class has ' + minClass + ' samples: use at most ' + minClass + ' folds.';
        return null;
    }

    function update() {
        var gsOn = gridSearch.checked;
        gsRadios.forEach(function (r) { r.disabled = !gsOn; });
        var exhaustive = document.getElementById('gs-exhaustive').checked;
        gsHelp.textContent = !gsOn ? 'Default hyperparameters are used.'
            : exhaustive ? 'Tries every combination: thorough, but can take hours.'
            : 'Tries 40 random combinations per model.';
        corrValue.textContent = Number(corr.value).toFixed(2);
        metricHelp.textContent = metric.selectedOptions[0].getAttribute('data-description');
        feature.disabled = !bias.checked;

        var n = selectedModels().length;
        var dl = selectedModels().filter(function (m) { return m.getAttribute('data-group') === 'deep_learning'; }).length;
        modelsError.classList.toggle('d-none', n > 0);
        var parts = [n + ' model' + (n === 1 ? '' : 's') + (dl ? ' (' + dl + ' deep learning)' : ''),
                     (kFolds.value || '?') + '-fold cross-validation',
                     gsOn ? (exhaustive ? 'exhaustive search' : 'randomized search') : 'default hyperparameters'];
        if (bias.checked) parts.push('bias assessment');
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
            models.forEach(function (m) { if (m.getAttribute('data-group') === group) m.checked = value; });
            update();
        });
    });
    form.addEventListener('input', update);
    form.addEventListener('change', update);

    form.addEventListener('submit', function (e) {
        update();
        var invalid = null;
        if (!selectedModels().length) invalid = models[0];
        else if (foldsProblem()) invalid = kFolds;
        else if (bias.checked && !feature.value) {
            feature.classList.add('is-invalid');
            invalid = feature;
        }
        if (invalid) {
            e.preventDefault();
            invalid.focus();
            invalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
            return;
        }
        feature.classList.remove('is-invalid');
        runButton.disabled = true;
        runButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" aria-hidden="true"></span>Starting…';
    });
    feature.addEventListener('change', function () { feature.classList.remove('is-invalid'); });

    update();
})();
