// Forecasting upload page: CSV parsing and checks in the browser (long format), preview.
(function () {
    'use strict';

    var MAX_BYTES = 50 * 1024 * 1024;
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
        var idIndex = -1;
        ID_COLUMNS.some(function (name) { idIndex = header.indexOf(name); return idIndex >= 0; });
        var info = { file: file, header: header, rows: body, nRows: body.length, idIndex: idIndex,
                     timeIndex: header.indexOf('Time'), targetIndex: header.indexOf('Target'),
                     ids: new Set(), duplicates: 0, targetMissing: 0, targetBad: 0 };
        var seen = new Set();
        body.forEach(function (r) {
            var id = idIndex >= 0 ? (r[idIndex] || '').trim() : '';
            info.ids.add(id);
            if (info.timeIndex >= 0) {
                var key = id + '\u0000' + (r[info.timeIndex] || '').trim();
                if (seen.has(key)) info.duplicates++;
                seen.add(key);
            }
            if (info.targetIndex >= 0) {
                var v = (r[info.targetIndex] || '').trim();
                if (MISSING.has(v)) info.targetMissing++;
                else if (!isNumber(v)) info.targetBad++;
            }
        });
        return info;
    }

    function check(level, text) { return { level: level, text: text }; }

    function fileChecks(kind, info) {
        var checks = [];
        if (info.file.size > MAX_BYTES) checks.push(check('bad', 'File is larger than 50 MB.'));
        if (!info.header.length || !info.nRows) { checks.push(check('bad', 'No data rows found.')); return checks; }
        var missing = [];
        if (info.idIndex < 0) missing.push('ID');
        if (info.timeIndex < 0) missing.push('Time');
        if (info.targetIndex < 0) missing.push('Target');
        if (missing.length) {
            checks.push(check('bad', 'Missing column' + (missing.length > 1 ? 's' : '') + ': ' + missing.join(', ') + '.'));
            return checks;
        }
        checks.push(check('ok', info.ids.size.toLocaleString() + ' series, ' + info.nRows.toLocaleString() + ' rows'));
        if (info.targetBad) checks.push(check('bad', 'The Target column must contain numbers (' + info.targetBad + ' value(s) are not).'));
        if (info.duplicates) checks.push(check('bad', info.duplicates + ' row(s) repeat the ID and Time of another row.'));
        if (info.targetMissing) checks.push(check('warn', info.targetMissing + ' missing Target value(s): interpolated within each series.'));
        var covariates = info.header.filter(function (c, j) { return j !== info.idIndex && j !== info.timeIndex && j !== info.targetIndex; });
        if (kind === 'train') {
            checks.push(check('ok', covariates.length ? 'Covariates: ' + covariates.join(', ') : 'No covariates: the target history only'));
        }
        return checks;
    }

    function crossChecks() {
        var train = state.train, test = state.test, checks = [];
        if (!train || !test || train.idIndex < 0 || test.idIndex < 0) return checks;
        var core = function (c) { return ID_COLUMNS.indexOf(c) < 0; };
        var missing = train.header.filter(core).filter(function (c) { return test.header.indexOf(c) < 0; });
        if (missing.length) checks.push(check('bad', 'Columns of Train.csv missing from Test.csv: ' + missing.join(', ')));
        else checks.push(check('ok', 'Same columns as Train.csv'));
        var continued = 0;
        test.ids.forEach(function (id) { if (train.ids.has(id)) continued++; });
        var fresh = test.ids.size - continued;
        var parts = [];
        if (continued) parts.push(continued + ' continue Train.csv series');
        if (fresh) parts.push(fresh + ' new series');
        checks.push(check('ok', 'Test series: ' + parts.join(', ')));
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
        submit.disabled = !(state.train && state.test && !blocking);
        if (!state.train || !state.test) status.textContent = 'Add both files to continue.';
        else if (blocking) status.textContent = 'Fix the errors above to continue.';
        else status.textContent = warnings ? 'Ready, with ' + warnings + ' warning(s) to review.' : 'Both files look good. The time steps are checked on upload.';
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
