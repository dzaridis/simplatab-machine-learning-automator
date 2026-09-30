// Run page: polls the pipeline status and renders it.
(function () {
    'use strict';

    var root = document.getElementById('run-root');
    var url = root.getAttribute('data-status-url');
    var STATUS = {
        pending: ['bi-circle', 'Waiting'],
        running: ['bi-arrow-repeat spin', 'In progress'],
        done: ['bi-check-circle-fill', 'Done'],
        skipped: ['bi-x-circle-fill', 'Skipped']
    };
    var timer = null;

    function formatElapsed(seconds) {
        var h = Math.floor(seconds / 3600), m = Math.floor(seconds % 3600 / 60), s = seconds % 60;
        var mm = (h ? String(m).padStart(2, '0') : String(m)), ss = String(s).padStart(2, '0');
        return (h ? h + ':' : '') + mm + ':' + ss;
    }

    function statusCell(value) {
        var cell = document.createElement('span');
        var spec = STATUS[value] || STATUS.pending;
        cell.className = 'status-cell ' + value;
        cell.innerHTML = '<i class="bi ' + spec[0] + '" aria-hidden="true"></i>';
        cell.appendChild(document.createTextNode(spec[1]));
        return cell;
    }

    function render(s) {
        document.getElementById('elapsed').textContent = formatElapsed(s.elapsed);
        document.getElementById('progress-label').textContent = s.progress + '%';
        var bar = document.getElementById('progress');
        bar.setAttribute('aria-valuenow', s.progress);
        bar.firstElementChild.style.width = s.progress + '%';

        var pill = document.getElementById('state-pill');
        var labels = { running: ['running', 'bi-arrow-repeat spin', 'Running'], done: ['available', 'bi-check-circle-fill', 'Finished'],
                       error: ['critical', 'bi-x-octagon-fill', 'Failed'], idle: ['soon', 'bi-pause-circle', 'Idle'] }[s.state];
        pill.className = 'status-pill ' + labels[0];
        pill.innerHTML = '<i class="bi ' + labels[1] + '" aria-hidden="true"></i><span>' + labels[2] + '</span>';

        var order = s.phases.map(function (p) { return p[0]; });
        var current = order.indexOf(s.phase);
        document.querySelectorAll('#phases li').forEach(function (li) {
            var index = order.indexOf(li.getAttribute('data-phase'));
            var done = s.state === 'done' || index < current;
            li.className = done ? 'done' : (index === current && s.state === 'running' ? 'active' : '');
            li.setAttribute('aria-current', li.className === 'active' ? 'step' : 'false');
            var icon = li.querySelector('i');
            if (icon) icon.remove();
            if (done) li.insertAdjacentHTML('afterbegin', '<i class="bi bi-check2" aria-hidden="true"></i>');
            else if (li.className === 'active') li.insertAdjacentHTML('afterbegin', '<i class="bi bi-arrow-repeat spin" aria-hidden="true"></i>');
        });

        var body = document.getElementById('models-body');
        body.innerHTML = '';
        s.models.forEach(function (m) {
            var tr = body.insertRow();
            var name = tr.insertCell();
            name.className = 'fw-semibold';
            name.textContent = m.name;
            if (m.note) {
                var note = document.createElement('div');
                note.className = 'small-muted fw-normal';
                note.textContent = m.note;
                name.appendChild(note);
            }
            tr.insertCell().appendChild(statusCell(m.kfold));
            tr.insertCell().appendChild(statusCell(m.test));
        });

        var log = document.getElementById('log');
        var atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
        log.textContent = s.log.join('\n');
        if (atBottom) log.scrollTop = log.scrollHeight;

        document.getElementById('done-panel').classList.toggle('d-none', s.state !== 'done');
        document.getElementById('error-panel').classList.toggle('d-none', s.state !== 'error');
        document.getElementById('error-message').textContent = s.message;
        if (s.state === 'error') document.getElementById('log-details').open = true;
        if (s.state !== 'running' && timer) { clearInterval(timer); timer = null; }
    }

    function poll() {
        fetch(url, { cache: 'no-store' })
            .then(function (r) { return r.json(); })
            .then(render)
            .catch(function () { /* transient: retried on the next tick */ });
    }

    var initial = JSON.parse(document.getElementById('initial-status').textContent);
    render(initial);
    if (initial.state === 'running') timer = setInterval(poll, 2000);
})();
