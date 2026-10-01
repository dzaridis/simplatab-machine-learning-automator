// Forecasting parameters page: horizon and window limits, live summary and validation.
(function () {
    'use strict';

    var form = document.getElementById('params-form');
    var models = Array.prototype.slice.call(form.querySelectorAll('input[type=checkbox][data-group]'));
    var horizon = document.getElementById('horizon');
    var kFolds = document.getElementById('k_folds');
    var lookback = document.getElementById('lookback');
    var trials = document.getElementById('trials');
    var summary = document.getElementById('run-summary');
    var modelsError = document.getElementById('models-error');
    var runButton = document.getElementById('run-btn');
    var horizonFeedback = document.getElementById('horizon-feedback');
    var foldsFeedback = document.getElementById('k-folds-feedback');
    var horizonHint = horizonFeedback.textContent;
    var lengthMin = parseInt(form.getAttribute('data-length-min'), 10);
    var maxHorizon = parseInt(form.getAttribute('data-max-horizon'), 10);

    function selected() { return models.filter(function (m) { return m.checked; }); }
    function h() { return parseInt(horizon.value, 10); }
    // Every training series keeps at least one horizon before its first validation window
    function maxFolds() { return Math.min(10, Math.floor(lengthMin / h()) - 1); }

    function horizonProblem() {
        if (isNaN(h()) || h() < 1) return 'Use a horizon of at least 1.';
        if (h() > maxHorizon) return 'At most ' + maxHorizon + ': Test.csv gives ' + maxHorizon + ' points for its shortest series.';
        if (maxFolds() < 1) return 'The shortest training series (' + lengthMin + ' points) needs at least two horizons: use at most ' + Math.floor(lengthMin / 2) + '.';
        return null;
    }

    function foldsProblem() {
        var k = parseInt(kFolds.value, 10);
        if (isNaN(k) || k < 1) return 'Use at least 1 window.';
        if (!horizonProblem() && k > maxFolds()) return 'At most ' + maxFolds() + ' with this horizon: the shortest training series has ' + lengthMin + ' points.';
        return null;
    }

    function update() {
        lookback.disabled = !document.getElementById('lookback-fixed').checked;
        var n = selected().length;
        modelsError.classList.toggle('d-none', n > 0);

        var hp = horizonProblem();
        horizon.classList.toggle('is-invalid', !!hp);
        horizonFeedback.classList.toggle('text-danger', !!hp);
        horizonFeedback.textContent = hp || horizonHint;

        var fp = foldsProblem();
        kFolds.classList.toggle('is-invalid', !!fp);
        foldsFeedback.classList.toggle('text-danger', !!fp);
        foldsFeedback.textContent = fp || (hp ? '' : 'At most ' + maxFolds() + ' with a horizon of ' + h() + '.');

        var configs = 1 + (parseInt(trials.value, 10) || 0);
        var trainings = n * (configs * (parseInt(kFolds.value, 10) || 0) + 1);
        summary.textContent = [n + ' model' + (n === 1 ? '' : 's'), 'horizon ' + (horizon.value || '?'),
                               (kFolds.value || '?') + ' validation window' + (kFolds.value === '1' ? '' : 's'),
                               trainings + ' trainings in total'].join(' · ');
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
        var invalid = !selected().length ? models[0] : horizonProblem() ? horizon : foldsProblem() ? kFolds : null;
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
