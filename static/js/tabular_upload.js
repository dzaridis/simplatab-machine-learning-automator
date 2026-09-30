// Upload page: drag and drop, CSV parsing and checks in the browser, preview.
(function () {
    'use strict';

    var MAX_BYTES = 50 * 1024 * 1024;
    // Values pandas reads as missing by default
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
        var info = { file: file, header: header, rows: body, nRows: body.length, missingRows: 0,
                     targetIndex: header.indexOf('Target'), categorical: {}, labels: null };
        body.forEach(function (r) {
            if (r.some(function (v) { return MISSING.has(v.trim()); })) info.missingRows++;
        });
        header.forEach(function (name, j) {
            if (j === info.targetIndex || ID_COLUMNS.indexOf(name) >= 0) return;
            var values = body.map(function (r) { return (r[j] || '').trim(); }).filter(function (v) { return !MISSING.has(v); });
            if (values.length && !values.every(isNumber)) info.categorical[name] = new Set(values);
        });
        if (info.targetIndex >= 0) {
            var values = body.map(function (r) { return (r[info.targetIndex] || '').trim(); }).filter(function (v) { return !MISSING.has(v); });
            info.targetNumeric = values.length > 0 && values.every(isNumber);
            if (info.targetNumeric) {
                var counts = {};
                values.forEach(function (v) { var k = Number(v); counts[k] = (counts[k] || 0) + 1; });
                info.labels = counts;
            }
        }
        return info;
    }

    function check(level, text) { return { level: level, text: text }; }

    function fileChecks(kind, info) {
        var checks = [];
        if (info.file.size > MAX_BYTES) checks.push(check('bad', 'File is larger than 50 MB.'));
        if (!info.header.length || !info.nRows) { checks.push(check('bad', 'No data rows found.')); return checks; }
        checks.push(check('ok', info.nRows.toLocaleString() + ' rows, ' + info.header.length + ' columns'));
        if (info.targetIndex < 0) {
            checks.push(check('bad', 'No column named "Target".'));
        } else if (!info.targetNumeric) {
            checks.push(check('bad', 'The Target column must contain numbers.'));
        } else {
            var labels = Object.keys(info.labels).map(Number).sort(function (a, b) { return a - b; });
            if (kind === 'train') {
                if (labels.length < 2) checks.push(check('bad', 'Target has a single class: at least two are needed.'));
                else {
                    var consecutive = labels.every(function (l, i) { return l === i; });
                    var kindLabel = labels.length > 2 ? 'Multiclass (' + labels.length + ' classes)' : 'Binary classification';
                    checks.push(check(consecutive ? 'ok' : 'warn', consecutive ? kindLabel :
                        kindLabel + ': classes should be numbered ' + labels.map(function (_, i) { return i; }).join(', ') +
                        ' (found ' + labels.join(', ') + '). The reported metrics assume this numbering.'));
                }
            } else {
                checks.push(check('ok', 'Target column found'));
            }
        }
        if (info.missingRows) {
            checks.push(check('warn', info.missingRows.toLocaleString() + ' row(s) with missing values will be removed.'));
        }
        return checks;
    }

    function crossChecks() {
        var train = state.train, test = state.test, checks = [];
        if (!train || !test) return checks;
        var trainCols = train.header.filter(function (c) { return ID_COLUMNS.indexOf(c) < 0; });
        var testCols = test.header.filter(function (c) { return ID_COLUMNS.indexOf(c) < 0; });
        var missing = trainCols.filter(function (c) { return testCols.indexOf(c) < 0; });
        var extra = testCols.filter(function (c) { return trainCols.indexOf(c) < 0; });
        if (missing.length) checks.push(check('bad', 'Columns of Train.csv missing from Test.csv: ' + missing.join(', ')));
        else checks.push(check('ok', 'Same columns as Train.csv'));
        if (extra.length) checks.push(check('warn', 'Columns only in Test.csv (ignored): ' + extra.join(', ')));
        if (train.labels && test.labels) {
            var unknown = Object.keys(test.labels).filter(function (l) { return !(l in train.labels); });
            if (unknown.length) checks.push(check('warn', 'Target classes not in Train.csv: ' + unknown.join(', ')));
        }
        var dropped = Object.keys(train.categorical).filter(function (c) {
            var other = test.categorical[c];
            if (!other) return false;
            var a = train.categorical[c];
            return a.size !== other.size || Array.from(a).some(function (v) { return !other.has(v); });
        });
        if (dropped.length) checks.push(check('warn', 'Categorical columns with different values in the two files will be dropped: ' + dropped.join(', ')));
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
        var table = document.createElement('table');
        table.className = 'table table-sm table-hover preview-table mb-0';
        var thead = table.createTHead().insertRow();
        // Target first: it is often the last column, out of view
        var order = info.header.map(function (_, j) { return j; });
        if (info.targetIndex > 0) { order.splice(info.targetIndex, 1); order.unshift(info.targetIndex); }
        order.forEach(function (j) {
            var h = info.header[j];
            var th = document.createElement('th');
            th.textContent = h;
            if (j === info.targetIndex) th.className = 'target-col';
            thead.appendChild(th);
        });
        var tbody = table.createTBody();
        info.rows.slice(0, 8).forEach(function (r) {
            var tr = tbody.insertRow();
            order.forEach(function (j) {
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
        var ready = state.train && state.test && !blocking;
        submit.disabled = !ready;
        if (!state.train || !state.test) status.textContent = 'Add both files to continue.';
        else if (blocking) status.textContent = 'Fix the errors above to continue.';
        else status.textContent = warnings ? 'Ready, with ' + warnings + ' warning(s) to review.' : 'Both files look good.';
        document.getElementById('preview-card').classList.toggle('d-none', !(state.train || state.test));
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
            var info = analyse(file, text);
            state[kind] = info;
            var size = file.size > 1048576 ? (file.size / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(file.size / 1024)) + ' KB';
            zone.querySelector('[data-field="meta"]').textContent = size;
            renderPreview(kind, info);
            refresh();
        });
    }

    document.querySelectorAll('[data-dropzone]').forEach(function (zone) {
        var kind = zone.getAttribute('data-dropzone');
        var input = zone.querySelector('input[type=file]');
        input.addEventListener('change', function () { load(kind, input.files[0]); });
        zone.querySelector('[data-action="replace"]').addEventListener('click', function () { input.click(); });
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
