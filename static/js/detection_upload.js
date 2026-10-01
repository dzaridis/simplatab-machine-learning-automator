// Detection upload page: lists the files of each zip in the browser (from the zip's central
// directory, without reading the whole file) to recognise the images and the annotation
// format, then uploads with a progress bar. The annotations are parsed on the server.
(function () {
    'use strict';

    var form = document.getElementById('upload-form');
    var submit = document.getElementById('submit-btn');
    var status = document.getElementById('form-status');
    var MAX_BYTES = Number(form.getAttribute('data-max-bytes'));
    var IMAGE_EXT = /\.(png|jpe?g|bmp|tiff?|dcm|dicom|dic|nii|nii\.gz)$/i;
    var MASK_FOLDERS = ['masks', 'mask', 'labels', 'label', 'segmentations', 'annotations'];
    var state = { train: null, test: null };

    // ---- Zip central directory ------------------------------------------------------
    function readSlice(file, start, end) {
        return file.slice(start, end).arrayBuffer().then(function (b) { return new DataView(b); });
    }

    function u64(view, offset) {
        return view.getUint32(offset, true) + view.getUint32(offset + 4, true) * 4294967296;
    }

    function listZip(file) {
        var tail = Math.min(file.size, 65557 + 20);
        return readSlice(file, file.size - tail, file.size).then(function (view) {
            var eocd = -1;
            for (var i = view.byteLength - 22; i >= 0; i--) {
                if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
            }
            if (eocd < 0) throw new Error('not a zip file');
            var entries = view.getUint16(eocd + 10, true);
            var size = view.getUint32(eocd + 12, true);
            var offset = view.getUint32(eocd + 16, true);
            if (entries === 0xffff || size === 0xffffffff || offset === 0xffffffff) {
                // ZIP64: the locator just before the end record points to the ZIP64 end record
                if (eocd < 20 || view.getUint32(eocd - 20, true) !== 0x07064b50) throw new Error('unsupported zip');
                var recordOffset = u64(view, eocd - 12);
                return readSlice(file, recordOffset, recordOffset + 56).then(function (record) {
                    if (record.getUint32(0, true) !== 0x06064b50) throw new Error('unsupported zip');
                    return { entries: u64(record, 32), size: u64(record, 40), offset: u64(record, 48) };
                });
            }
            return { entries: entries, size: size, offset: offset };
        }).then(function (dir) {
            if (dir.size > 128 * 1024 * 1024) throw new Error('zip index too large to preview');
            return readSlice(file, dir.offset, dir.offset + dir.size);
        }).then(function (view) {
            var names = [], p = 0, utf8 = new TextDecoder('utf-8');
            while (p + 46 <= view.byteLength && view.getUint32(p, true) === 0x02014b50) {
                var nameLength = view.getUint16(p + 28, true), extra = view.getUint16(p + 30, true), comment = view.getUint16(p + 32, true);
                names.push(utf8.decode(new Uint8Array(view.buffer, view.byteOffset + p + 46, nameLength)));
                p += 46 + nameLength + extra + comment;
            }
            return names;
        });
    }

    function analyze(names) {
        // Files only, without macOS metadata and hidden files
        var files = names.filter(function (n) { return !/\/$/.test(n); }).map(function (n) {
            return n.replace(/\\/g, '/').split('/').filter(function (p) { return p && p !== '.'; });
        }).filter(function (parts) {
            return parts.length && parts[0] !== '__MACOSX' && !parts.some(function (p) { return p.charAt(0) === '.'; });
        });
        var info = { images: 0, volumes: 0, json: 0, xml: 0, txt: 0, csv: 0, masks: 0, series: new Set() };
        files.forEach(function (parts) {
            var name = parts[parts.length - 1].toLowerCase();
            var inMasks = parts.slice(0, -1).some(function (p) { return MASK_FOLDERS.indexOf(p.toLowerCase()) >= 0; });
            if (/\.(png|jpe?g|bmp|tiff?|nii|nii\.gz)$/.test(name) && inMasks) { info.masks++; return; }
            if (/\.nii(\.gz)?$/.test(name)) { info.volumes++; return; }
            if (/\.json$/.test(name)) { info.json++; return; }
            if (/\.xml$/.test(name)) { info.xml++; return; }
            if (/\.csv$/.test(name)) { info.csv++; return; }
            if (/\.(txt|names|ya?ml)$/.test(name)) { if (!/^(classes|obj|labels)\.(txt|names)$|\.ya?ml$/.test(name)) info.txt++; return; }
            if (IMAGE_EXT.test(name) || name.indexOf('.') < 0 || /^[\d.]+$/.test(name)) {
                info.images++;
                if (/\.(dcm|dicom|dic)$/.test(name) || name.indexOf('.') < 0) info.series.add(parts.slice(0, -1).join('/'));
            }
        });
        var formats = [];
        if (info.json) formats.push('COCO JSON');
        if (info.csv) formats.push('CSV');
        if (info.xml) formats.push('Pascal VOC');
        if (info.txt) formats.push('YOLO');
        if (info.masks) formats.push('masks');
        info.formats = formats;
        return info;
    }

    // ---- Checks and rendering ---------------------------------------------------------
    function check(level, text) { return { level: level, text: text }; }

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

    function fileChecks(kind, info) {
        var checks = [];
        if (info.file.size > MAX_BYTES) checks.push(check('bad', 'Larger than ' + Math.round(MAX_BYTES / 1073741824) + ' GB.'));
        if (info.error) { checks.push(check('warn', 'Contents could not be previewed (' + info.error + '): they will be checked after the upload.')); return checks; }
        var parts = [];
        if (info.images) parts.push(info.images.toLocaleString() + ' image file(s)');
        if (info.volumes) parts.push(info.volumes.toLocaleString() + ' NIfTI volume(s)');
        if (!parts.length) checks.push(check('bad', 'No images or volumes found.'));
        else checks.push(check('ok', parts.join(', ')));
        if (!info.formats.length) checks.push(check('bad', 'No annotations found (COCO JSON, YOLO .txt, VOC .xml, CSV or a masks folder).'));
        else if (info.formats.length > 1) checks.push(check('warn', 'Several possible annotation files (' + info.formats.join(', ') + '): keep one format per zip.'));
        else checks.push(check('ok', 'Annotations: ' + info.formats[0] + ' (checked after the upload)'));
        return checks;
    }

    function crossChecks() {
        var checks = [];
        if (!state.train || !state.test || state.train.error || state.test.error) return checks;
        var train3d = state.train.volumes > 0, test3d = state.test.volumes > 0;
        if (train3d !== test3d) checks.push(check('warn', 'One zip holds NIfTI volumes and the other does not: both must be 2D or both 3D.'));
        return checks;
    }

    function refresh() {
        var blocking = false, warnings = 0;
        ['train', 'test'].forEach(function (kind) {
            var info = state[kind];
            if (!info) return;
            var zone = document.querySelector('[data-dropzone="' + kind + '"]');
            var checks = fileChecks(kind, info);
            if (kind === 'test') checks = checks.concat(crossChecks());
            renderChecks(zone.querySelector('[data-field="checks"]'), checks);
            var bad = checks.some(function (c) { return c.level === 'bad'; });
            zone.classList.toggle('has-error', bad);
            blocking = blocking || bad;
            warnings += checks.filter(function (c) { return c.level === 'warn'; }).length;
        });
        submit.disabled = !(state.train && state.test) || blocking;
        if (!state.train || !state.test) status.textContent = 'Add both zip files to continue.';
        else if (blocking) status.textContent = 'Fix the errors above to continue.';
        else status.textContent = warnings ? 'Ready, with ' + warnings + ' warning(s) to review.' : 'Both files look good.';
    }

    function formatBytes(bytes) {
        if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(2) + ' GB';
        if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + ' MB';
        return Math.max(1, Math.round(bytes / 1024)) + ' KB';
    }

    function load(kind, file) {
        if (!file) return;
        var zone = document.querySelector('[data-dropzone="' + kind + '"]');
        zone.querySelector('[data-field="name"]').textContent = file.name;
        zone.querySelector('[data-field="meta"]').textContent = formatBytes(file.size) + ' · reading…';
        zone.classList.add('has-file');
        zone.querySelector('.dz-empty').classList.add('d-none');
        zone.querySelector('.dz-file').classList.remove('d-none');
        listZip(file).then(function (names) {
            state[kind] = analyze(names);
            state[kind].file = file;
        }).catch(function (e) {
            state[kind] = { file: file, error: e.message, formats: [] };
            if (!/\.zip$/i.test(file.name)) state[kind].error = 'not a zip file';
        }).then(function () {
            zone.querySelector('[data-field="meta"]').textContent = formatBytes(file.size);
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

    // ---- Upload with progress ---------------------------------------------------------
    form.addEventListener('submit', function (e) {
        e.preventDefault();
        if (submit.disabled) return;
        var panel = document.getElementById('upload-progress'), bar = document.getElementById('upload-bar');
        var label = document.getElementById('upload-label'), detail = document.getElementById('upload-detail');
        var error = document.getElementById('upload-error');
        error.classList.add('d-none');
        panel.classList.remove('d-none');
        submit.disabled = true;
        submit.querySelector('.btn-label').classList.add('d-none');
        submit.querySelector('.btn-busy').classList.remove('d-none');
        document.querySelectorAll('[data-action="replace"]').forEach(function (b) { b.disabled = true; });

        var xhr = new XMLHttpRequest(), started = Date.now();
        xhr.open('POST', form.action);
        xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
        xhr.upload.onprogress = function (event) {
            if (!event.lengthComputable) return;
            var percent = Math.floor(100 * event.loaded / event.total);
            var speed = event.loaded / Math.max(1, (Date.now() - started) / 1000);
            var remaining = (event.total - event.loaded) / Math.max(speed, 1);
            bar.setAttribute('aria-valuenow', percent);
            bar.firstElementChild.style.width = percent + '%';
            label.textContent = 'Uploading… ' + percent + '%';
            detail.textContent = formatBytes(event.loaded) + ' of ' + formatBytes(event.total) + ' · ' + formatBytes(speed) + '/s' +
                (percent < 100 ? ' · about ' + Math.ceil(remaining / 60) + ' min left' : '');
        };
        xhr.upload.onload = function () {
            label.textContent = 'Extracting the images and reading the annotations…';
            detail.textContent = 'This can take a few minutes for large datasets.';
        };
        function fail(message) {
            panel.classList.add('d-none');
            error.textContent = message;
            error.classList.remove('d-none');
            submit.disabled = false;
            submit.querySelector('.btn-label').classList.remove('d-none');
            submit.querySelector('.btn-busy').classList.add('d-none');
            document.querySelectorAll('[data-action="replace"]').forEach(function (b) { b.disabled = false; });
        }
        xhr.onload = function () {
            var body = {};
            try { body = JSON.parse(xhr.responseText); } catch (err) { /* not JSON */ }
            if (xhr.status === 200 && body.redirect) window.location.href = body.redirect;
            else if (xhr.status === 413) fail('The upload is too large.');
            else fail(body.error || 'The upload failed (HTTP ' + xhr.status + ').');
        };
        xhr.onerror = function () { fail('The upload was interrupted. Check the connection and try again.'); };
        xhr.send(new FormData(form));
    });
})();
