// Clustering upload page: CSV parsing and checks in the browser, preview. Test.csv is optional.
(function () {
    'use strict';

    var MAX_BYTES = 50 * 1024 * 1024;
    var MIN_ROWS = 10;
    var MISSING = new Set(['', 'NA', 'N/A', 'NaN', 'nan', '-NaN', '-nan', 'null', 'NULL', 'None', '#N/A', '#NA', 'n/a', '<NA>']);
    var ID_COLUMNS = ['ID', 'patient_id'];

    var state = { train: null, test: null };
    var form = document.getElementById('upload-form');
    var submit = document.getElementById('submit-btn');
    var status = document.getElementById('form-status');

    // RFC 4180 parser (quoted fields, escaped quotes, CRLF)
    function parseCSV(text) {
        var rows = [], row = [], field = '', i = 0, quoted = false;
        if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
        while (i < text.length) {
            var c = text[i];
            if (quoted) {
                if (c === '"') {
                    if (text[i + 1] === '"') { field += '"'; i++; } else { quoted = false; }
                } else { field += c; }
            } else if (c === '"') { quoted = true; }
            else if (c === ',') { row.push(field); field = ''; }
            else if (c === '\n' || c === '\r') {
                if (c === '\r' && text[i + 1] === '\n') i++;
                row.push(field); field = '';
                if (row.length > 1 || row[0] !== '') rows.push(row);
                row = [];
            } else { field += c; }
            i++;
        }
        if (field !== '' || row.length) { row.push(field); rows.push(row); }
        return rows;
    }

    function isNumber(value) { return value.trim() !== '' && !isNaN(Number(value)); }

    function analyse(file, text) {
        var rows = parseCSV(text);
        var header = rows.length ? rows[0].map(function (h) { return h.trim(); }) : [];
        var body = rows.slice(1);
        var info = { file: file, header: header, rows: body, nRows: body.length, targetIndex: header.indexOf('Target'),
                     idIndex: -1, features: [], numeric: 0, categorical: 0, missing: 0, classes: new Set(), targetNumeric: true };
        ID_COLUMNS.some(function (name) { info.idIndex = header.indexOf(name); return info.idIndex >= 0; });
        header.forEach(function (column, j) {
            if (j === info.idIndex || j === info.targetIndex) return;
            info.features.push(column);
            var numeric = true;
            body.forEach(function (r) {
                var v = (r[j] || '').trim();
                if (MISSING.has(v)) info.missing++;
                else if (!isNumber(v)) numeric = false;
            });
            if (numeric) info.numeric++; else info.categorical++;
        });
        if (info.targetIndex >= 0) {
            body.forEach(function (r) {
                var v = (r[info.targetIndex] || '').trim();
                if (MISSING.has(v)) return;
                info.classes.add(v);
                if (!isNumber(v)) info.targetNumeric = false;
            });
        }
        return info;
    }

    function check(level, text) { return { level: level, text: text }; }

    function fileChecks(kind, info) {
        var checks = [];
        if (info.file.size > MAX_BYTES) checks.push(check('bad', 'File is larger than 50 MB.'));
        if (!info.header.length || !info.nRows) { checks.push(check('bad', 'No data rows found.')); return checks; }
        if (kind === 'train' && info.nRows < MIN_ROWS) checks.push(check('bad', 'Clustering needs at least ' + MIN_ROWS + ' rows.'));
        if (!info.features.length) { checks.push(check('bad', 'No feature column (besides ID and Target).')); return checks; }
        checks.push(check('ok', info.nRows.toLocaleString() + ' samples · ' + info.numeric + ' numeric and ' + info.categorical + ' categorical feature' + (info.categorical === 1 ? '' : 's')));
        if (info.missing) checks.push(check('warn', info.missing + ' missing value(s): imputed (median, or a "missing" category).'));
        if (kind === 'train') {
            if (info.targetIndex < 0) checks.push(check('ok', 'No Target: unsupervised clustering (internal metrics)'));
            else if (info.targetNumeric && info.classes.size > 50) checks.push(check('warn', 'Target has ' + info.classes.size + ' distinct values: if they are measurements, not classes, it will be left out.'));
            else checks.push(check('ok', 'Target: ' + info.classes.size + ' classes, used only to evaluate the clusters'));
            if (info.idIndex < 0) checks.push(check('warn', 'No ID column: samples are identified by their row.'));
        }
        return checks;
    }

    function crossChecks() {
        var train = state.train, test = state.test, checks = [];
        if (!train || !test) return checks;
        var missing = train.features.filter(function (c) { return test.header.indexOf(c) < 0; });
        if (missing.length) checks.push(check('bad', 'Feature columns of Train.csv missing from Test.csv: ' + missing.join(', ')));
        else checks.push(check('ok', 'Same feature columns as Train.csv'));
        if (train.targetIndex >= 0 && test.targetIndex < 0) checks.push(check('warn', 'No Target: the test clusters are evaluated without labels.'));
        return checks;
    }

    function renderChecks(list, checks) {
        list.innerHTML = '';
        checks.forEach(function (c) {
            var li = document.createElement('li');
            var icon = { ok: 'bi-check-circle-fill ok', warn: 'bi-exclamation-triangle-fill warn', bad: 'bi-x-circle-fill bad' }[c.level];
            var label = { ok: 'OK', warn: 'Warning', bad: 'Error' }[c.level];
            li.innerHTML = '<i class="bi ' + icon + '" aria-hidden="true"></i><span><span class="visually-hidden">' + label + ': </span></span>';
            li.lastChild.appendChild(document.createTextNode(c.text));
            list.appendChild(li);
        });
    }

    function renderPreview(kind, info) {
        var container = document.querySelector('#preview-' + kind + ' .table-responsive');
        if (!info) {
            container.innerHTML = '<p class="small-muted p-3 mb-0">No Test.csv: the clusters are found and validated on Train.csv.</p>';
            return;
        }
        var table = document.createElement('table');
        table.className = 'table table-sm table-hover preview-table mb-0';
        var thead = table.createTHead().insertRow();
        info.header.forEach(function (h, j) {
            var th = document.createElement('th');
            th.textContent = h;
            if (j === info.targetIndex) th.className = 'target-col';
            thead.appendChild(th);
        });
        var tbody = table.createTBody();
        info.rows.slice(0, 8).forEach(function (r) {
            var tr = tbody.insertRow();
            info.header.forEach(function (_, j) {
                var td = tr.insertCell();
                td.textContent = r[j] === undefined ? '' : r[j];
                if (j === info.targetIndex) td.className = 'target-col';
            });
        });
        container.innerHTML = '';
        container.appendChild(table);
    }

    function refresh() {
        var blocking = false, warnings = 0;
        ['train', 'test'].forEach(function (kind) {
            var zone = document.querySelector('[data-dropzone="' + kind + '"]');
            var info = state[kind];
            if (!info) return;
            var checks = fileChecks(kind, info);
            if (kind === 'test') checks = checks.concat(crossChecks());
            renderChecks(zone.querySelector('[data-field="checks"]'), checks);
            var bad = checks.some(function (c) { return c.level === 'bad'; });
            zone.classList.toggle('has-error', bad);
            blocking = blocking || bad;
            warnings += checks.filter(function (c) { return c.level === 'warn'; }).length;
        });
        submit.disabled = !(state.train && !blocking);
        if (!state.train) status.textContent = 'Add Train.csv to continue.';
        else if (blocking) status.textContent = 'Fix the errors above to continue.';
        else status.textContent = (warnings ? 'Ready, with ' + warnings + ' warning(s) to review.' : 'The data looks good.')
            + (state.test ? '' : ' Test.csv is optional.');
        document.getElementById('preview-card').classList.toggle('d-none', !(state.train || state.test));
    }

    function reset(kind) {
        var zone = document.querySelector('[data-dropzone="' + kind + '"]');
        state[kind] = null;
        zone.querySelector('input[type=file]').value = '';
        zone.classList.remove('has-file', 'has-error');
        zone.querySelector('.dz-empty').classList.remove('d-none');
        zone.querySelector('.dz-file').classList.add('d-none');
        renderPreview(kind, null);
        refresh();
    }

    function load(kind, file) {
        var zone = document.querySelector('[data-dropzone="' + kind + '"]');
        if (!file) return;
        zone.querySelector('[data-field="name"]').textContent = file.name;
        zone.querySelector('[data-field="meta"]').textContent = 'Reading…';
        zone.classList.add('has-file');
        zone.querySelector('.dz-empty').classList.add('d-none');
        zone.querySelector('.dz-file').classList.remove('d-none');
        file.text().then(function (text) {
            state[kind] = analyse(file, text);
            var size = file.size > 1048576 ? (file.size / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(file.size / 1024)) + ' KB';
            zone.querySelector('[data-field="meta"]').textContent = size;
            renderPreview(kind, state[kind]);
            refresh();
        });
    }

    document.querySelectorAll('[data-dropzone]').forEach(function (zone) {
        var kind = zone.getAttribute('data-dropzone');
        var input = zone.querySelector('input[type=file]');
        input.addEventListener('change', function () { if (input.files[0]) load(kind, input.files[0]); });
        zone.querySelector('[data-action="replace"]').addEventListener('click', function () { input.click(); });
        var remove = zone.querySelector('[data-action="remove"]');
        if (remove) remove.addEventListener('click', function () { reset(kind); });
        ['dragenter', 'dragover'].forEach(function (type) {
            zone.addEventListener(type, function (e) { e.preventDefault(); zone.classList.add('dragover'); });
        });
        ['dragleave', 'drop'].forEach(function (type) {
            zone.addEventListener(type, function (e) { e.preventDefault(); zone.classList.remove('dragover'); });
        });
        zone.addEventListener('drop', function (e) {
            if (!e.dataTransfer.files.length) return;
            input.files = e.dataTransfer.files;
            load(kind, input.files[0]);
        });
    });

    form.addEventListener('submit', function (e) {
        if (submit.disabled) { e.preventDefault(); return; }
        submit.disabled = true;
        submit.querySelector('.btn-label').classList.add('d-none');
        submit.querySelector('.btn-busy').classList.remove('d-none');
    });
})();
