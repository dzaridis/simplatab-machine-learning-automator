// Detection parameters page: validation mode, live summary and validation of the form.
(function () {
    'use strict';

    var form = document.getElementById('params-form');
    var networks = Array.prototype.slice.call(form.querySelectorAll('input[type=checkbox][data-group]'));
    var kFolds = document.getElementById('k_folds');
    var summary = document.getElementById('run-summary');
    var modelsError = document.getElementById('models-error');
    var runButton = document.getElementById('run-btn');
    var foldsFeedback = document.getElementById('k-folds-feedback');
    var foldsHint = foldsFeedback.textContent;
    var maxFolds = parseInt(form.getAttribute('data-max-folds'), 10);

    function selected() { return networks.filter(function (n) { return n.checked; }); }
    function mode() { return form.querySelector('input[name=validation]:checked').value; }

    function foldsProblem() {
        if (mode() !== 'kfold') return null;
        var k = parseInt(kFolds.value, 10);
        if (isNaN(k) || k < 2) return 'Use at least 2 folds.';
        if (k > maxFolds) return 'At most ' + maxFolds + ' folds: one image (or patient) per fold at least.';
        return null;
    }

    function update() {
        var kfold = mode() === 'kfold';
        document.getElementById('kfold-settings').classList.toggle('d-none', !kfold);
        document.getElementById('holdout-settings').classList.toggle('d-none', kfold);
        var n = selected().length;
        modelsError.classList.toggle('d-none', n > 0);
        var trainings = n * (kfold ? (parseInt(kFolds.value, 10) || 0) + 1 : 1);
        summary.textContent = [n + ' network' + (n === 1 ? '' : 's'),
                               kfold ? (kFolds.value || '?') + '-fold cross-validation' : 'hold-out validation',
                               document.getElementById('image_size').value + ' px',
                               trainings + ' fine-tuning' + (trainings === 1 ? '' : 's')].join(' · ');
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
