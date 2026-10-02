import os
import glob
import json
import logging
import shutil
import tempfile
import traceback
import zipfile

# The pipeline runs in a background thread: use a non-interactive matplotlib backend
# (GUI backends, e.g. macOS', cannot draw outside the main thread).
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import yaml
from flask import (Flask, render_template, request, redirect, url_for, flash, send_from_directory,
                   send_file, jsonify, abort)
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from Helpers.pipelines_main import train_k_fold, external_test, read_yaml
from Helpers.data_checks import DataChecker
from Helpers import DBDM
from Helpers.standalone import requirements as model_requirements
from Helpers.image import dataset as image_dataset
from Helpers.image.io import CT_WINDOWS
from Helpers.image.models import BACKBONES, BY_KEY as BACKBONES_BY_KEY
from Helpers.image3d import dataset as image3d_dataset
from Helpers.image3d.models import NETWORKS as NETWORKS_3D, BY_KEY as NETWORKS_3D_BY_KEY
from Helpers.image3d.volumes import SHAPES as VOLUME_SHAPES, CROPS as VOLUME_CROPS, SINGLE as SINGLE_SERIES
from Helpers.forecasting import data as forecast_data
from Helpers.forecasting.models import MODELS as FORECAST_MODELS
from Helpers.detection import dataset as detection_dataset
from Helpers.segmentation import data as segmentation_data
from Helpers.segmentation.models import NETWORKS as SEG_NETWORKS, BY_KEY as SEG_BY_KEY
from Helpers.detection.models import DETECTORS, BY_KEY as DETECTORS_BY_KEY
from Helpers.detection.metrics import MAIN_IOU
from web.catalog import AUTOMATORS, MODELS, THRESHOLD_METRICS, get_automator
from web.jobs import PipelineJob, PHASES, IMAGE_PHASES, FORECAST_PHASES, DETECTION_PHASES, SEGMENTATION_PHASES

# Set in the Docker image by the release CI (same as the image and release tags)
APP_VERSION = os.environ.get("SIMPLATAB_VERSION", "dev")

app = Flask(__name__, template_folder='templates')
# Required by flash() to show the upload errors
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(24)

root_app = Flask(__name__)
@root_app.route('/')
def root_redirect():
    return redirect('/automl/')

app.wsgi_app = ProxyFix(app.wsgi_app)
application = DispatcherMiddleware(root_app, {
    '/automl': app.wsgi_app
})

# Create temporary directories for input and output
TEMP_INPUT_FOLDER = os.path.join(tempfile.gettempdir(), 'ml_app_input')
TEMP_OUTPUT_FOLDER = os.path.join(tempfile.gettempdir(), 'ml_app_output')
os.makedirs(TEMP_INPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_OUTPUT_FOLDER, exist_ok=True)

# Image automator: uploaded zips, extracted class folders and the preprocessed image cache
IMAGE_INPUT_FOLDER = os.path.join(tempfile.gettempdir(), 'ml_app_images')
# Detection automator: uploaded zips, extracted images and the cache of 8-bit images and volumes
DETECTION_INPUT_FOLDER = os.path.join(tempfile.gettempdir(), 'ml_app_detection')
# Segmentation automator: uploaded zips, extracted images and masks, the cache and nnU-Net's folders
SEGMENTATION_INPUT_FOLDER = os.path.join(tempfile.gettempdir(), 'ml_app_segmentation')
# Forecasting automator: Train.csv, Test.csv, their summary and the run parameters
FORECAST_INPUT_FOLDER = os.path.join(tempfile.gettempdir(), 'ml_app_forecasting')
FORECAST_EXAMPLE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Examples", "time-series-forecasting")

# Configure upload settings
ALLOWED_EXTENSIONS = {'csv'}
TABULAR_MAX_BYTES = 50 * 1024 * 1024  # 50 MB per CSV upload
# Largest request: the two zips of the image automator (5 GB each)
app.config['MAX_CONTENT_LENGTH'] = 2 * image_dataset.MAX_ZIP_BYTES + 16 * 1024 * 1024

job = PipelineJob()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def materials_dir():
    """Folder where the pipeline writes its outputs (relative to the working directory:
    /app/Materials in the Docker image)."""
    return os.path.abspath("Materials")


def has_results():
    root = materials_dir()
    return any(files for _, _, files in os.walk(root)) if os.path.isdir(root) else False


def clear_materials():
    """Deletes all the outputs of the previous run."""
    root = materials_dir()
    # The pipeline logs errors to Materials/error_log.log: release the file, otherwise the
    # next run would keep writing to the deleted file (logging.basicConfig reopens it).
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler) and handler.baseFilename.startswith(root + os.sep):
            root_logger.removeHandler(handler)
            handler.close()

    # Check if directory exists
    if os.path.exists(root):
        # Remove all files in the Materials directory and its subdirectories
        for dirpath, dirs, files in os.walk(root, topdown=False):
            for file in files:
                file_path = os.path.join(dirpath, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
            
            # Remove empty subdirectories except the Materials directory itself
            if dirpath != root:
                try:
                    os.rmdir(dirpath)
                except Exception as e:
                    print(f"Error removing directory {dirpath}: {e}")
    
    # Recreate any necessary subdirectories
    os.makedirs(os.path.join(root, "Models"), exist_ok=True)


@app.context_processor
def inject_globals():
    return {"automators": AUTOMATORS, "app_version": APP_VERSION, "job_running": job.running}


# ---------------------------------------------------------------------------
# Landing page and automators
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('landing.html')


@app.route('/automators/<slug>')
def automator(slug):
    item = get_automator(slug)
    if item is None:
        abort(404)
    if item.available:
        return redirect(url_for(item.endpoint))
    return render_template('automator_placeholder.html', automator=item)


# ---------------------------------------------------------------------------
# Tabular automator: upload -> parameters -> run -> results
# ---------------------------------------------------------------------------

@app.route('/tabular')
def tabular():
    return render_template('tabular/upload.html', automator=get_automator("tabular"))


@app.route('/upload', methods=['POST'])
def upload_files():
    if job.running:
        flash('A pipeline is already running. Wait for it to finish before uploading new data.', 'warning')
        return redirect(url_for('run'))

    # Clear temporary folders
    for folder in [TEMP_INPUT_FOLDER, TEMP_OUTPUT_FOLDER]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    if (request.content_length or 0) > TABULAR_MAX_BYTES:
        flash('The files are too large: at most 50 MB per upload.', 'danger')
        return redirect(url_for('tabular'))

    # Check if files were uploaded
    if 'train_file' not in request.files or 'test_file' not in request.files:
        flash('Missing required files', 'danger')
        return redirect(url_for('tabular'))

    train_file = request.files['train_file']
    test_file = request.files['test_file']

    # Check if filenames are valid
    if train_file.filename == '' or test_file.filename == '':
        flash('No selected files', 'danger')
        return redirect(url_for('tabular'))

    if not (allowed_file(train_file.filename) and allowed_file(test_file.filename)):
        flash('Invalid file type. Only CSV files are allowed.', 'danger')
        return redirect(url_for('tabular'))

    # Save files
    train_file.save(os.path.join(TEMP_INPUT_FOLDER, 'Train.csv'))
    test_file.save(os.path.join(TEMP_INPUT_FOLDER, 'Test.csv'))

    # Redirect to parameters page
    return redirect(url_for('parameters'))


def dataset_summary():
    """Summary of the uploaded Train.csv / Test.csv for the parameters page (None if missing)."""
    train_path = os.path.join(TEMP_INPUT_FOLDER, "Train.csv")
    test_path = os.path.join(TEMP_INPUT_FOLDER, "Test.csv")
    if not os.path.exists(train_path):
        return None
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    summary = {
        "train_rows": len(train_df),
        "test_rows": len(test_df) if test_df is not None else 0,
        "n_features": len([c for c in train_df.columns if c not in ("Target", "ID", "patient_id")]),
        "rows_with_missing": int(train_df.isna().any(axis=1).sum()),
        "has_target": 'Target' in train_df.columns,
        "is_multiclass": False,
        "num_classes": 0,
        "class_distribution": [],
        "min_class_count": 0,
        "target_warning": None,
        # Candidate facets for the bias assessment: categorical or low-cardinality columns
        "bias_features": [c for c in train_df.columns
                          if c not in ("Target", "ID", "patient_id")
                          and (train_df[c].dtype == object or train_df[c].nunique() <= 10)],
    }
    if summary["has_target"]:
        counts = train_df['Target'].value_counts().sort_index()
        summary.update(
            num_classes=len(counts),
            is_multiclass=len(counts) > 2,
            min_class_count=int(counts.min()),
            class_distribution=[
                {"label": label, "count": int(count), "percentage": round(100 * count / len(train_df), 1)}
                for label, count in counts.items()
            ],
            target_warning=DataChecker.target_label_issue(train_df, test_df),
        )
    return summary


@app.route('/parameters', methods=['GET', 'POST'])
def parameters():
    if job.running:
        return redirect(url_for('run'))
    if not os.path.exists(os.path.join(TEMP_INPUT_FOLDER, "Train.csv")):
        flash('Upload Train.csv and Test.csv first.', 'warning')
        return redirect(url_for('tabular'))

    if request.method == 'POST':
        # Get parameters from form
        params = {}
        params["BiasAssessment"] = request.form.get('bias_assessment') == 'true'
        params["Feature"] = request.form.get('feature') or 'None'
        params["number_of_k_folds"] = int(request.form.get('k_folds'))

        params["apply_grid_search"] = {}
        params["apply_grid_search"]["enabled"] = request.form.get('grid_search') == 'true'
        params["apply_grid_search"]["type"] = {}
        randomized = request.form.get('grid_search_type', 'randomized') == 'randomized'
        params["apply_grid_search"]["type"]["Randomized"] = randomized
        params["apply_grid_search"]["type"]["Exhaustive"] = not randomized

        params["Correlation Limit"] = float(request.form.get('correlation_limit'))
        params["Metric For Threshold Optimization"] = request.form.get('optimization_metric')

        params["Machine Learning Models"] = {
            model.name: request.form.get(model.field) == 'true' for model in MODELS
        }
        selected = [name for name, enabled in params["Machine Learning Models"].items() if enabled]
        if not selected:
            flash('Select at least one model.', 'danger')
            return redirect(url_for('parameters'))

        # Save YAML file
        yaml_path = os.path.join(TEMP_INPUT_FOLDER, "machine_learning_parameters.yaml")
        with open(yaml_path, 'w') as file:
            yaml.dump(params, file)

        # Run the machine learning pipeline in the background and follow it on the run page.
        # The results of the previous run are replaced (the parameters page says so).
        clear_materials()
        if not job.start(lambda: run_pipeline(TEMP_INPUT_FOLDER, TEMP_OUTPUT_FOLDER, params), selected):
            flash('A pipeline is already running.', 'warning')
        return redirect(url_for('run'))

    summary = dataset_summary()
    return render_template(
        'tabular/parameters.html',
        summary=summary,
        previous_results=has_results(),
        target_warning=summary["target_warning"],
        models=MODELS,
        threshold_metrics=THRESHOLD_METRICS,
    )


@app.route('/run')
def run():
    if job.state == "idle":
        return redirect(url_for('index'))
    return render_template('run.html', status=job.snapshot(), automator=get_automator(job.automator))


@app.route('/api/status')
def status():
    return jsonify(job.snapshot())


# ---------------------------------------------------------------------------
# Image classification automator: upload -> parameters -> run -> results
# ---------------------------------------------------------------------------

IMAGE_SUMMARY = "summary.json"
WINDOW_LABELS = [("auto", "Automatic (DICOM header)")] + [
    (key, f"CT {key.replace('_', ' ')} ({center} / {width} HU)") for key, (center, width) in CT_WINDOWS.items()]
VOLUME_LABELS = [("middle", "Middle slice"), ("mip", "Maximum intensity projection")]


def image_summary():
    path = os.path.join(IMAGE_INPUT_FOLDER, IMAGE_SUMMARY)
    return image_dataset.load_json(path) if os.path.exists(path) else None


def _wants_json():
    return request.headers.get('X-Requested-With') == 'XMLHttpRequest'


def _image_upload_error(message):
    if _wants_json():
        return jsonify({"error": message}), 400
    flash(message, 'danger')
    return redirect(url_for('image'))


@app.route('/image')
def image():
    return render_template('image/upload.html', automator=get_automator("image-classification"),
                           max_gb=image_dataset.MAX_ZIP_BYTES // 1024 ** 3)


IMAGE_EXAMPLE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Examples", "image-classification-3d")


@app.route('/image/example/<name>')
def image_example(name):
    if name not in ("Train3D.zip", "Test3D.zip"):
        return redirect(url_for('image'))
    return send_from_directory(IMAGE_EXAMPLE_FOLDER, name, as_attachment=True)


@app.route('/image/upload', methods=['POST'])
def image_upload():
    if job.running:
        return _image_upload_error('A pipeline is already running. Wait for it to finish before uploading new data.')
    files = {split: request.files.get(f'{split}_zip') for split in ("train", "test")}
    if not all(f and f.filename for f in files.values()):
        return _image_upload_error('Add both Train.zip and Test.zip.')
    if not all(f.filename.lower().endswith('.zip') for f in files.values()):
        return _image_upload_error('Invalid file type: upload two .zip files.')

    shutil.rmtree(IMAGE_INPUT_FOLDER, ignore_errors=True)
    os.makedirs(IMAGE_INPUT_FOLDER)
    try:
        for split, upload in files.items():
            archive = os.path.join(IMAGE_INPUT_FOLDER, f"{split}.zip")
            upload.save(archive)
            if os.path.getsize(archive) > image_dataset.MAX_ZIP_BYTES:
                raise image_dataset.DatasetError(
                    f"{upload.filename} is larger than {image_dataset.MAX_ZIP_BYTES // 1024 ** 3} GB.")
            image_dataset.extract_zip(archive, os.path.join(IMAGE_INPUT_FOLDER, split))
            os.remove(archive)  # keep only the extracted images
        summary = image_dataset.summarize(os.path.join(IMAGE_INPUT_FOLDER, "train"),
                                          os.path.join(IMAGE_INPUT_FOLDER, "test"))
        summary["volume3d"] = volume_summary(IMAGE_INPUT_FOLDER, summary)
        summary["dim"] = 3 if summary["volume3d"] else 2
    except image_dataset.DatasetError as e:
        shutil.rmtree(IMAGE_INPUT_FOLDER, ignore_errors=True)
        return _image_upload_error(str(e))
    if summary["errors"] and not summary["volume3d"]:
        shutil.rmtree(IMAGE_INPUT_FOLDER, ignore_errors=True)
        return _image_upload_error(" ".join(summary["errors"]))
    image_dataset.save_json(summary, os.path.join(IMAGE_INPUT_FOLDER, IMAGE_SUMMARY))
    if _wants_json():
        return jsonify({"redirect": url_for('image_parameters')})
    return redirect(url_for('image_parameters'))


def volume_summary(folder, summary):
    """The 3D view of an upload made of volumes (DICOM series, NIfTI, multi-frame DICOM) usable for
    3D classification, else None. Uploads with PNG/JPEG/TIFF images are classified in 2D."""
    if summary["kinds"].get("raster") or not (summary["kinds"].get("dicom") or summary["kinds"].get("nifti")):
        return None
    try:
        summary = image3d_dataset.summarize(os.path.join(folder, "train"), os.path.join(folder, "test"))
    except Exception:
        logging.error(traceback.format_exc())
        return None
    return summary if summary["volumetric"] and not summary["errors"] else None


def _bounded(form, name, cast, low, high, default):
    try:
        value = cast(form.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid value for {name}.")
    if not low <= value <= high:
        raise ValueError(f"{name.replace('_', ' ').capitalize()} must be between {low} and {high}.")
    return value


def image_params_from_form(form, summary):
    selected = [b.key for b in BACKBONES if form.get(b.key) == 'true']
    if not selected:
        raise ValueError('Select at least one network.')
    max_folds = min(20, summary["min_class_count"])
    positive = form.get('positive_class')
    return {
        "models": selected,
        "mode": "finetune" if form.get('mode') == 'finetune' else "features",
        "k_folds": _bounded(form, 'k_folds', int, 2, max(2, max_folds), 5),
        "metric": form.get('optimization_metric') if form.get('optimization_metric') in dict(THRESHOLD_METRICS) else "Balanced Accuracy",
        "classes": summary["classes"],
        "positive_class": positive if positive in summary["classes"] else summary["positive_class"],
        "window": form.get('window') if form.get('window') in dict(WINDOW_LABELS) else "auto",
        "volume": form.get('volume') if form.get('volume') in dict(VOLUME_LABELS) else "middle",
        "augmentation": {key: form.get(key) == 'true' for key in ("horizontal_flip", "vertical_flip", "rotation", "intensity")},
        "epochs": _bounded(form, 'epochs', int, 1, 200, 20),
        "learning_rate": _bounded(form, 'learning_rate', float, 1e-6, 1e-2, 1e-4),
        "patience": _bounded(form, 'patience', int, 1, 50, 5),
        "batch_size": _bounded(form, 'batch_size', int, 1, 256, 32),
    }


def image3d_params_from_form(form, summary):
    selected = [n.key for n in NETWORKS_3D if form.get(n.key) == 'true']
    if not selected:
        raise ValueError('Select at least one network.')
    names = {s["name"] for s in summary["series"]}
    if summary["single_series"]:
        channels = [SINGLE_SERIES]
    else:
        channels = [name for name in form.getlist('channels') if name in names]
        reference = form.get('reference')
        if reference in channels:  # the reference series comes first: the others are aligned on its grid
            channels = [reference] + [c for c in channels if c != reference]
        if not channels:
            raise ValueError('Select at least one series.')
    shapes = {"x".join(map(str, shape)): list(shape) for shape in VOLUME_SHAPES}
    crop = _bounded(form, 'crop', float, 0.5, 1.0, 1.0)
    max_folds = min(20, summary["min_class_patients"])
    positive = form.get('positive_class')
    return {
        "models": selected,
        "mode": "finetune" if form.get('mode') == 'finetune' else "features",
        "k_folds": _bounded(form, 'k_folds', int, 2, max(2, max_folds), 5),
        "metric": form.get('optimization_metric') if form.get('optimization_metric') in dict(THRESHOLD_METRICS) else "Balanced Accuracy",
        "classes": summary["classes"],
        "positive_class": positive if positive in summary["classes"] else summary["positive_class"],
        "channels": channels,
        "shape": shapes.get(form.get('shape'), list(VOLUME_SHAPES[0])),
        "crop": crop if crop in VOLUME_CROPS else 1.0,
        "window": form.get('window') if form.get('window') in dict(WINDOW_LABELS) else "auto",
        "augmentation": {key: form.get(key) == 'true' for key in ("horizontal_flip", "vertical_flip", "rotation", "intensity")},
        "epochs": _bounded(form, 'epochs', int, 1, 200, 30),
        "learning_rate": _bounded(form, 'learning_rate', float, 1e-6, 1e-2, 1e-4),
        "patience": _bounded(form, 'patience', int, 1, 50, 8),
        "batch_size": _bounded(form, 'batch_size', int, 1, 64, 4),
        "pretrained": os.environ.get("SIMPLATAB_PRETRAINED", "1") != "0",
    }


@app.route('/image/parameters', methods=['GET', 'POST'])
def image_parameters():
    if job.running:
        return redirect(url_for('run'))
    summary = image_summary()
    if summary is None:
        flash('Upload Train.zip and Test.zip first.', 'warning')
        return redirect(url_for('image'))
    volumes3d = summary.get("volume3d")
    requested = request.form.get('dim') if request.method == 'POST' else request.args.get('dim')
    dim = 3 if volumes3d and requested != '2' else 2
    if dim == 2 and summary["errors"]:  # only the 3D view of the upload is usable
        return redirect(url_for('image_parameters', dim=3))

    if request.method == 'POST' and dim == 3:
        try:
            params = image3d_params_from_form(request.form, volumes3d)
        except ValueError as e:
            flash(str(e), 'danger')
            return redirect(url_for('image_parameters'))
        image_dataset.save_json(params, os.path.join(IMAGE_INPUT_FOLDER, "params.json"))
        names = [NETWORKS_3D_BY_KEY[key].name for key in params["models"]]
        clear_materials()
        from Helpers.image3d.pipeline import run_image3d_pipeline
        if not job.start(lambda: run_image3d_pipeline(IMAGE_INPUT_FOLDER, params), names,
                         automator="image-classification", phases=IMAGE_PHASES, initial_phase="prep"):
            flash('A pipeline is already running.', 'warning')
        return redirect(url_for('run'))

    if request.method == 'GET' and dim == 3:
        import torch
        return render_template(
            'image/parameters3d.html',
            automator=get_automator("image-classification"),
            summary=volumes3d,
            files_summary=summary,
            previous_results=has_results(),
            networks=NETWORKS_3D,
            shapes=VOLUME_SHAPES,
            crops=VOLUME_CROPS,
            threshold_metrics=THRESHOLD_METRICS,
            window_labels=WINDOW_LABELS,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            cpu_count=os.cpu_count(),
        )

    if request.method == 'POST':
        try:
            params = image_params_from_form(request.form, summary)
        except ValueError as e:
            flash(str(e), 'danger')
            return redirect(url_for('image_parameters'))
        image_dataset.save_json(params, os.path.join(IMAGE_INPUT_FOLDER, "params.json"))
        names = [BACKBONES_BY_KEY[key].name for key in params["models"]]
        # The results of the previous run are replaced (the parameters page says so)
        clear_materials()
        from Helpers.image.pipeline import run_image_pipeline
        if not job.start(lambda: run_image_pipeline(IMAGE_INPUT_FOLDER, params), names,
                         automator="image-classification", phases=IMAGE_PHASES, initial_phase="prep"):
            flash('A pipeline is already running.', 'warning')
        return redirect(url_for('run'))

    import torch
    return render_template(
        'image/parameters.html',
        automator=get_automator("image-classification"),
        summary=summary,
        previous_results=has_results(),
        backbones=BACKBONES,
        threshold_metrics=THRESHOLD_METRICS,
        window_labels=WINDOW_LABELS,
        volume_labels=VOLUME_LABELS,
        volumes3d=bool(volumes3d),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cpu_count=os.cpu_count(),
    )


# ---------------------------------------------------------------------------
# Object detection automator: upload -> parameters -> run -> results
# ---------------------------------------------------------------------------

DETECTION_SIZES = [320, 512, 640, 800, 1024]


def detection_summary():
    path = os.path.join(DETECTION_INPUT_FOLDER, "summary.json")
    return detection_dataset.load_json(path) if os.path.exists(path) else None


def _detection_upload_error(message):
    if _wants_json():
        return jsonify({"error": message}), 400
    flash(message, 'danger')
    return redirect(url_for('detection'))


@app.route('/detection')
def detection():
    return render_template('detection/upload.html', automator=get_automator("object-detection"),
                           max_gb=image_dataset.MAX_ZIP_BYTES // 1024 ** 3)


DETECTION_EXAMPLES = ("Train.zip", "Test.zip", "Train3D.zip", "Test3D.zip")
DETECTION_EXAMPLE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Examples", "object-detection")


@app.route('/detection/example/<name>')
def detection_example(name):
    if name not in DETECTION_EXAMPLES:
        abort(404)
    return send_from_directory(DETECTION_EXAMPLE_FOLDER, name, as_attachment=True)


@app.route('/detection/upload', methods=['POST'])
def detection_upload():
    if job.running:
        return _detection_upload_error('A pipeline is already running. Wait for it to finish before uploading new data.')
    files = {split: request.files.get(f'{split}_zip') for split in ("train", "test")}
    if not all(f and f.filename for f in files.values()):
        return _detection_upload_error('Add both Train.zip and Test.zip.')
    if not all(f.filename.lower().endswith('.zip') for f in files.values()):
        return _detection_upload_error('Invalid file type: upload two .zip files.')
    shutil.rmtree(DETECTION_INPUT_FOLDER, ignore_errors=True)
    os.makedirs(DETECTION_INPUT_FOLDER)
    try:
        for split, upload in files.items():
            archive = os.path.join(DETECTION_INPUT_FOLDER, f"{split}.zip")
            upload.save(archive)
            if os.path.getsize(archive) > image_dataset.MAX_ZIP_BYTES:
                raise image_dataset.DatasetError(
                    f"{upload.filename} is larger than {image_dataset.MAX_ZIP_BYTES // 1024 ** 3} GB.")
            image_dataset.extract_zip(archive, os.path.join(DETECTION_INPUT_FOLDER, split))
            os.remove(archive)
        summary = detection_dataset.summarize(os.path.join(DETECTION_INPUT_FOLDER, "train"),
                                              os.path.join(DETECTION_INPUT_FOLDER, "test"))
    except image_dataset.DatasetError as e:
        shutil.rmtree(DETECTION_INPUT_FOLDER, ignore_errors=True)
        return _detection_upload_error(str(e))
    if summary["errors"]:
        shutil.rmtree(DETECTION_INPUT_FOLDER, ignore_errors=True)
        return _detection_upload_error(" ".join(summary["errors"]))
    detection_dataset.save_json(summary, os.path.join(DETECTION_INPUT_FOLDER, "summary.json"))
    if _wants_json():
        return jsonify({"redirect": url_for('detection_parameters')})
    return redirect(url_for('detection_parameters'))


def detection_params_from_form(form, summary):
    selected = [d.key for d in DETECTORS if form.get(d.key) == 'true']
    if not selected:
        raise ValueError('Select at least one network.')
    validation = "holdout" if form.get('validation') == 'holdout' else "kfold"
    size = _bounded(form, 'image_size', int, 320, 1024, 640)
    return {
        "models": selected,
        "validation": validation,
        "k_folds": _bounded(form, 'k_folds', int, 2, max(2, summary["max_folds"]), min(5, summary["max_folds"])),
        "holdout_fraction": _bounded(form, 'holdout_percent', int, 10, 40, 20) / 100,
        "epochs": _bounded(form, 'epochs', int, 1, 300, 30),
        "patience": _bounded(form, 'patience', int, 1, 50, 5),
        "batch_size": _bounded(form, 'batch_size', int, 1, 64, 4),
        "image_size": size if size in DETECTION_SIZES else 640,
        "lr_scale": _bounded(form, 'lr_scale', float, 0.1, 10, 1.0),
        "augmentation": {key: form.get(key) == 'true' for key in ("horizontal_flip", "vertical_flip", "intensity")},
        "window": form.get('window') if form.get('window') in dict(WINDOW_LABELS) else "auto",
        "negative_ratio": _bounded(form, 'negative_ratio', float, 0, 5, 1.0),
        "drise_images": _bounded(form, 'drise_images', int, 0, 12, 4),
        "drise_masks": _bounded(form, 'drise_masks', int, 50, 2000, 300),
        # SIMPLATAB_PRETRAINED=0: networks without pretrained weights (offline use, tests)
        "pretrained": os.environ.get("SIMPLATAB_PRETRAINED", "1") != "0",
    }


@app.route('/detection/parameters', methods=['GET', 'POST'])
def detection_parameters():
    if job.running:
        return redirect(url_for('run'))
    summary = detection_summary()
    if summary is None:
        flash('Upload Train.zip and Test.zip first.', 'warning')
        return redirect(url_for('detection'))

    if request.method == 'POST':
        try:
            params = detection_params_from_form(request.form, summary)
        except ValueError as e:
            flash(str(e), 'danger')
            return redirect(url_for('detection_parameters'))
        detection_dataset.save_json(params, os.path.join(DETECTION_INPUT_FOLDER, "params.json"))
        names = [DETECTORS_BY_KEY[key].name for key in params["models"]]
        clear_materials()
        from Helpers.detection.pipeline import run_detection_pipeline
        if not job.start(lambda: run_detection_pipeline(DETECTION_INPUT_FOLDER, params), names,
                         automator="object-detection", phases=DETECTION_PHASES, initial_phase="prep"):
            flash('A pipeline is already running.', 'warning')
        return redirect(url_for('run'))

    import torch
    return render_template(
        'detection/parameters.html',
        automator=get_automator("object-detection"),
        summary=summary,
        previous_results=has_results(),
        detectors=DETECTORS,
        sizes=DETECTION_SIZES,
        window_labels=WINDOW_LABELS,
        main_iou=MAIN_IOU[summary["dim"]],
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cpu_count=os.cpu_count(),
    )


# ---------------------------------------------------------------------------
# Segmentation automator: upload -> parameters -> run -> results
# ---------------------------------------------------------------------------

SEGMENTATION_EXAMPLES = ("Train.zip", "Test.zip", "Train3D.zip", "Test3D.zip")
SEGMENTATION_EXAMPLE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Examples", "image-segmentation")


def segmentation_summary():
    path = os.path.join(SEGMENTATION_INPUT_FOLDER, "summary.json")
    return segmentation_data.load_json(path) if os.path.exists(path) else None


def _segmentation_upload_error(message):
    if _wants_json():
        return jsonify({"error": message}), 400
    flash(message, 'danger')
    return redirect(url_for('segmentation'))


@app.route('/segmentation')
def segmentation():
    return render_template('segmentation/upload.html', automator=get_automator("image-segmentation"),
                           max_gb=image_dataset.MAX_ZIP_BYTES // 1024 ** 3)


@app.route('/segmentation/example/<name>')
def segmentation_example(name):
    if name not in SEGMENTATION_EXAMPLES:
        abort(404)
    return send_from_directory(SEGMENTATION_EXAMPLE_FOLDER, name, as_attachment=True)


@app.route('/segmentation/upload', methods=['POST'])
def segmentation_upload():
    if job.running:
        return _segmentation_upload_error('A pipeline is already running. Wait for it to finish before uploading new data.')
    files = {split: request.files.get(f'{split}_zip') for split in ("train", "test")}
    if not all(f and f.filename for f in files.values()):
        return _segmentation_upload_error('Add both Train.zip and Test.zip.')
    if not all(f.filename.lower().endswith('.zip') for f in files.values()):
        return _segmentation_upload_error('Invalid file type: upload two .zip files.')
    shutil.rmtree(SEGMENTATION_INPUT_FOLDER, ignore_errors=True)
    os.makedirs(SEGMENTATION_INPUT_FOLDER)
    try:
        for split, upload in files.items():
            archive = os.path.join(SEGMENTATION_INPUT_FOLDER, f"{split}.zip")
            upload.save(archive)
            if os.path.getsize(archive) > image_dataset.MAX_ZIP_BYTES:
                raise image_dataset.DatasetError(
                    f"{upload.filename} is larger than {image_dataset.MAX_ZIP_BYTES // 1024 ** 3} GB.")
            image_dataset.extract_zip(archive, os.path.join(SEGMENTATION_INPUT_FOLDER, split))
            os.remove(archive)
        summary = segmentation_data.summarize(os.path.join(SEGMENTATION_INPUT_FOLDER, "train"),
                                              os.path.join(SEGMENTATION_INPUT_FOLDER, "test"))
    except (image_dataset.DatasetError, segmentation_data.SegmentationDataError) as e:
        shutil.rmtree(SEGMENTATION_INPUT_FOLDER, ignore_errors=True)
        return _segmentation_upload_error(str(e))
    if summary["errors"]:
        shutil.rmtree(SEGMENTATION_INPUT_FOLDER, ignore_errors=True)
        return _segmentation_upload_error(" ".join(summary["errors"]))
    segmentation_data.save_json(summary, os.path.join(SEGMENTATION_INPUT_FOLDER, "summary.json"))
    if _wants_json():
        return jsonify({"redirect": url_for('segmentation_parameters')})
    return redirect(url_for('segmentation_parameters'))


def segmentation_params_from_form(form, summary):
    available = [n for n in SEG_NETWORKS if n.dim == summary["dim"]]
    selected = [n.key for n in available if form.get(n.key) == 'true']
    if not selected:
        raise ValueError('Select at least one network.')
    channels = None
    if len(summary.get("series", [])) > 1:
        names = {s["name"] for s in summary["series"]}
        channels = [name for name in form.getlist('channels') if name in names]
        reference = form.get('reference')
        if reference in channels:  # the reference series first: the others and the mask are put on its grid
            channels = [reference] + [c for c in channels if c != reference]
        if not channels:
            raise ValueError('Select at least one series.')
    elif summary.get("channels"):
        channels = summary["channels"]
    validation = "holdout" if form.get('validation') == 'holdout' else "kfold"
    return {
        "models": selected,
        "dim": summary["dim"],
        "mapping": summary["mapping"],
        "channels": channels,
        "validation": validation,
        "k_folds": _bounded(form, 'k_folds', int, 2, max(2, summary["max_folds"]), min(5, summary["max_folds"])),
        "holdout_fraction": _bounded(form, 'holdout_percent', int, 10, 40, 20) / 100,
        "epochs": _bounded(form, 'epochs', int, 1, 1000, 20),
        "iterations": _bounded(form, 'iterations', int, 1, 1000, 50),
        "batch_size": _bounded(form, 'batch_size', int, 1, 64, 2 if summary["dim"] == 3 else 8),
        "learning_rate": _bounded(form, 'learning_rate', float, 1e-5, 1e-2, 1e-3),
        "augmentation": {key: form.get(key) == 'true'
                         for key in ("rotation", "intensity", "horizontal_flip", "vertical_flip", "depth_flip")},
        "normalisation": form.get('normalisation') if form.get('normalisation') in ("auto", "ct", "zscore") else "auto",
        "tta": form.get('tta') == 'true',
        "nnunet_epochs": _bounded(form, 'nnunet_epochs', int, 1, 1000, 100),
        "nnunet_iterations": _bounded(form, 'nnunet_iterations', int, 1, 250, 250),
        # SIMPLATAB_PRETRAINED=0: networks without pretrained weights (offline use, tests)
        "pretrained": os.environ.get("SIMPLATAB_PRETRAINED", "1") != "0",
    }


@app.route('/segmentation/parameters', methods=['GET', 'POST'])
def segmentation_parameters():
    if job.running:
        return redirect(url_for('run'))
    summary = segmentation_summary()
    if summary is None:
        flash('Upload Train.zip and Test.zip first.', 'warning')
        return redirect(url_for('segmentation'))

    if request.method == 'POST':
        try:
            params = segmentation_params_from_form(request.form, summary)
        except ValueError as e:
            flash(str(e), 'danger')
            return redirect(url_for('segmentation_parameters'))
        segmentation_data.save_json(params, os.path.join(SEGMENTATION_INPUT_FOLDER, "params.json"))
        names = [SEG_BY_KEY[key].name for key in params["models"]]
        clear_materials()
        from Helpers.segmentation.pipeline import run_segmentation_pipeline
        if not job.start(lambda: run_segmentation_pipeline(SEGMENTATION_INPUT_FOLDER, params), names,
                         automator="image-segmentation", phases=SEGMENTATION_PHASES, initial_phase="prep"):
            flash('A pipeline is already running.', 'warning')
        return redirect(url_for('run'))

    import torch
    return render_template(
        'segmentation/parameters.html',
        automator=get_automator("image-segmentation"),
        summary=summary,
        previous_results=has_results(),
        networks=[n for n in SEG_NETWORKS if n.dim == summary["dim"]],
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cpu_count=os.cpu_count(),
    )


# ---------------------------------------------------------------------------
# Time series forecasting automator: upload -> parameters -> run -> results
# ---------------------------------------------------------------------------

def forecast_summary():
    path = os.path.join(FORECAST_INPUT_FOLDER, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@app.route('/forecasting')
def forecasting():
    return render_template('forecasting/upload.html', automator=get_automator("time-series-forecasting"))


@app.route('/forecasting/example/<name>')
def forecasting_example(name):
    if name not in ("Train.csv", "Test.csv"):
        abort(404)
    return send_from_directory(FORECAST_EXAMPLE_FOLDER, name, as_attachment=True)


@app.route('/forecasting/upload', methods=['POST'])
def forecasting_upload():
    if job.running:
        flash('A pipeline is already running. Wait for it to finish before uploading new data.', 'warning')
        return redirect(url_for('run'))
    if (request.content_length or 0) > TABULAR_MAX_BYTES:
        flash('The files are too large: at most 50 MB per upload.', 'danger')
        return redirect(url_for('forecasting'))
    files = {name: request.files.get(field) for name, field in (("Train.csv", "train_file"), ("Test.csv", "test_file"))}
    if not all(f and f.filename for f in files.values()):
        flash('Add both Train.csv and Test.csv.', 'danger')
        return redirect(url_for('forecasting'))
    if not all(allowed_file(f.filename) for f in files.values()):
        flash('Invalid file type. Only CSV files are allowed.', 'danger')
        return redirect(url_for('forecasting'))
    shutil.rmtree(FORECAST_INPUT_FOLDER, ignore_errors=True)
    os.makedirs(FORECAST_INPUT_FOLDER)
    for name, upload in files.items():
        upload.save(os.path.join(FORECAST_INPUT_FOLDER, name))
    summary = forecast_data.summarize(os.path.join(FORECAST_INPUT_FOLDER, "Train.csv"),
                                      os.path.join(FORECAST_INPUT_FOLDER, "Test.csv"))
    if summary["errors"]:
        shutil.rmtree(FORECAST_INPUT_FOLDER, ignore_errors=True)
        flash(" ".join(summary["errors"]), 'danger')
        return redirect(url_for('forecasting'))
    with open(os.path.join(FORECAST_INPUT_FOLDER, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return redirect(url_for('forecasting_parameters'))


def forecast_params_from_form(form, summary):
    selected = [m.key for m in FORECAST_MODELS if form.get(m.key) == 'true']
    if not selected:
        raise ValueError('Select at least one model.')
    horizon = _bounded(form, 'horizon', int, 1, max(1, summary["max_horizon"]), summary["suggested_horizon"])
    folds_limit = min(10, forecast_data.max_folds(summary["length_min"], horizon))
    if folds_limit < 1:
        raise ValueError(f"The shortest training series ({summary['length_min']} points) is too short for a horizon of "
                         f"{horizon}: it needs at least two horizons of points. Choose a shorter horizon.")
    lookback = 0
    if form.get('lookback_mode') == 'fixed':
        lookback = _bounded(form, 'lookback', int, 1, 10 * summary["length_max"], 2 * horizon)
    return {
        "models": selected,
        "horizon": horizon,
        "k_folds": _bounded(form, 'k_folds', int, 1, folds_limit, min(3, folds_limit)),
        "lookback": lookback,
        "trials": _bounded(form, 'trials', int, 0, 50, 0),
        "max_steps": _bounded(form, 'max_steps', int, 50, 10000, 500),
        "season": _bounded(form, 'season', int, 1, 1000, summary["season"]),
        "future_columns": [c for i, c in enumerate(summary["dynamic"]) if form.get(f'role_{i}') == 'future'],
    }


@app.route('/forecasting/parameters', methods=['GET', 'POST'])
def forecasting_parameters():
    if job.running:
        return redirect(url_for('run'))
    summary = forecast_summary()
    if summary is None:
        flash('Upload Train.csv and Test.csv first.', 'warning')
        return redirect(url_for('forecasting'))

    if request.method == 'POST':
        try:
            params = forecast_params_from_form(request.form, summary)
        except ValueError as e:
            flash(str(e), 'danger')
            return redirect(url_for('forecasting_parameters'))
        with open(os.path.join(FORECAST_INPUT_FOLDER, "params.json"), "w") as f:
            json.dump(params, f, indent=2)
        # The results of the previous run are replaced (the parameters page says so)
        clear_materials()
        from Helpers.forecasting.pipeline import run_forecasting_pipeline
        if not job.start(lambda: run_forecasting_pipeline(FORECAST_INPUT_FOLDER, params), params["models"],
                         automator="time-series-forecasting", phases=FORECAST_PHASES, initial_phase="data"):
            flash('A pipeline is already running.', 'warning')
        return redirect(url_for('run'))

    import torch
    return render_template(
        'forecasting/parameters.html',
        automator=get_automator("time-series-forecasting"),
        summary=summary,
        previous_results=has_results(),
        models=FORECAST_MODELS,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cpu_count=os.cpu_count(),
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

METRICS = ["AUC", "Balanced Accuracy", "F-score", "Accuracy", "Sensitivity", "Specificity"]


def _relative(path, root):
    return os.path.relpath(path, root).replace(os.sep, "/")


def collect_results(root):
    """Everything the results page shows, read from the Materials folder."""
    results = {"test": None, "kfold": None, "kfold_name": None, "best": None, "curves": [],
               "class_curves": [], "confusion": {}, "shap": {}, "gradcam": {}, "models": [], "files": [],
               "predictions": [], "classes": [], "skipped": [], "notes": [], "info": {},
               "automator": get_automator("tabular")}
    if not os.path.isdir(root):
        return results
    # Classes and settings of the run (older tabular runs have none)
    info_path = os.path.join(root, "run_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            results["info"] = json.load(f)
    results["automator"] = get_automator(results["info"].get("automator", "tabular"))

    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            results["files"].append({"path": _relative(path, root), "size": os.path.getsize(path)})
    results["files"].sort(key=lambda f: f["path"])

    test_path = os.path.join(root, "test_results.xlsx")
    if os.path.exists(test_path):
        test = pd.read_excel(test_path, index_col=0)
        metrics = [m for m in METRICS if m in test.columns]
        best = {m: test[m].max() for m in metrics}
        results["test"] = {
            "metrics": metrics,
            "rows": [{"model": model, "values": {m: float(row[m]) for m in metrics},
                      "best": {m: bool(row[m] == best[m]) for m in metrics}}
                     for model, row in test.iterrows()],
        }
        if "AUC" in metrics and len(test):
            top = test["AUC"].idxmax()
            results["best"] = {"model": top, "values": {m: float(test.loc[top, m]) for m in metrics}}

    kfold_paths = sorted(glob.glob(os.path.join(root, "*_fold_results.xlsx")), key=os.path.getmtime)
    if kfold_paths:
        kfold = pd.read_excel(kfold_paths[-1], index_col=0)
        results["kfold_name"] = os.path.basename(kfold_paths[-1]).split("_")[0]
        metrics = [m for m in METRICS if m in kfold.columns]
        results["kfold"] = {"metrics": metrics,
                            "rows": [{"model": model, "values": {m: str(row[m]) for m in metrics}}
                                     for model, row in kfold.iterrows()]}

    for name in ("ROC_CURVES.png", "PR_CURVES.png"):
        path = os.path.join(root, "ROC_Curves", name)
        if os.path.exists(path):
            results["curves"].append({"path": _relative(path, root),
                                      "title": "ROC curves" if name.startswith("ROC") else "Precision-recall curves"})
    for path in sorted(glob.glob(os.path.join(root, "ROC_Curves", "*_class_*_CURVES.png"))):
        name = os.path.basename(path)
        model, kind = name.split("_class_")[0], "ROC" if "_ROC_" in name else "Precision-recall"
        results["class_curves"].append({"path": _relative(path, root), "title": f"{model}: {kind} per class"})

    for path in sorted(glob.glob(os.path.join(root, "ConfusionMatrices", "*_confusion_matrix.png"))):
        name = os.path.basename(path)[:-len("_confusion_matrix.png")]
        if "_Internal_" in name:
            model, split = name.split("_Internal_")[0], "Internal K-fold (mean)"
        else:
            model, split = name.rsplit("_", 1)[0], "External test"
        results["confusion"].setdefault(model, []).append({"path": _relative(path, root), "title": split})

    for model_dir in sorted(glob.glob(os.path.join(root, "Shap_Features", "*"))):
        plots = sorted(glob.glob(os.path.join(model_dir, "*.png")))
        if plots:
            results["shap"][os.path.basename(model_dir)] = [
                {"path": _relative(p, root),
                 "title": os.path.basename(p)[:-4].replace(f"_{os.path.basename(model_dir)}", "").replace("_", " ")}
                for p in plots]

    for path in sorted(glob.glob(os.path.join(root, "Models", "*.pkl"))):
        name = os.path.basename(path)[:-len("_pipeline.pkl")]
        results["models"].append({"path": _relative(path, root), "name": name, "size": os.path.getsize(path),
                                  "requirements": model_requirements(name)})
    names = {"".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in b.name): b.name
             for b in list(BACKBONES) + list(NETWORKS_3D)}
    for path in sorted(glob.glob(os.path.join(root, "Models", "*.pt"))):
        stem = os.path.basename(path)[:-3]
        results["models"].append({"path": _relative(path, root), "name": names.get(stem, stem),
                                  "size": os.path.getsize(path)})

    for model_dir in sorted(glob.glob(os.path.join(root, "GradCAM", "*"))):
        plots = sorted(glob.glob(os.path.join(model_dir, "*.png")))
        if plots:
            model = names.get(os.path.basename(model_dir), os.path.basename(model_dir))
            results["gradcam"][model] = [
                {"path": _relative(p, root), "title": "Class " + os.path.basename(p)[:-4].split("_gradcam_", 1)[-1]}
                for p in plots]
    for path in sorted(glob.glob(os.path.join(root, "Predictions", "*.csv"))):
        stem = os.path.basename(path)[:-len("_test_predictions.csv")]
        results["predictions"].append({"path": _relative(path, root), "name": names.get(stem, stem),
                                       "size": os.path.getsize(path)})
    classes_path = os.path.join(root, "classes.csv")
    if os.path.exists(classes_path):
        results["classes"] = pd.read_csv(classes_path).to_dict("records")

    log_path = os.path.join(root, "error_log.log")
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as log:
            for line in log:
                if "failed and was skipped:" in line:
                    model, reason = line.split(":ERROR:", 1)[-1].split(" failed and was skipped:", 1)
                    results["skipped"].append({"model": model.strip(), "reason": reason.strip()})
    for name in ("Shap_error_log.txt", "Model_save_error_log.txt", "GradCAM_error_log.txt"):
        if os.path.exists(os.path.join(root, name)):
            results["notes"].append(name)
    return results


FORECAST_METRICS = ["MAE", "RMSE", "sMAPE", "MASE"]


def _figure(path, root, title):
    return {"path": _relative(path, root), "title": title} if os.path.exists(path) else None


def collect_forecast_results(root):
    """What the results page of the forecasting automator shows, read from the Materials folder."""
    with open(os.path.join(root, "run_info.json")) as f:
        info = json.load(f)
    results = {"info": info, "automator": get_automator("time-series-forecasting"), "test": None, "kfold": None,
               "best": None, "forecasts": {}, "explain": {}, "metric_plots": [], "models": [], "tables": [], "files": []}
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            results["files"].append({"path": _relative(path, root), "size": os.path.getsize(path)})
    results["files"].sort(key=lambda f: f["path"])

    test_path = os.path.join(root, "test_results.xlsx")
    if os.path.exists(test_path):
        test = pd.read_excel(test_path, index_col=0)
        best = {m: test[m].min() for m in FORECAST_METRICS}
        results["test"] = [{"model": model, "values": {m: float(row[m]) for m in FORECAST_METRICS},
                            "best": {m: bool(row[m] == best[m]) for m in FORECAST_METRICS}}
                           for model, row in test.iterrows()]
        top = info.get("best_model")
        if top in test.index:
            results["best"] = {"model": top, "values": {m: float(test.loc[top, m]) for m in FORECAST_METRICS}}
            baseline = "Seasonal naive"
            if baseline in test.index and test.loc[baseline, "MAE"] > 0:
                results["best"]["gain"] = 100 * (1 - test.loc[top, "MAE"] / test.loc[baseline, "MAE"])
    kfold_path = os.path.join(root, f"{info.get('k_folds')}_fold_results.xlsx")
    if os.path.exists(kfold_path):
        kfold = pd.read_excel(kfold_path, index_col=0)
        results["kfold"] = [{"model": model, "values": {m: str(row[m]) for m in FORECAST_METRICS}}
                            for model, row in kfold.iterrows()]

    for name, title in (("test_metrics.png", "Test errors by model"), ("error_by_horizon.png", "Test MAE at each horizon step")):
        item = _figure(os.path.join(root, "Metrics_Plots", name), root, title)
        if item:
            results["metric_plots"].append(item)
    order = [row["model"] for row in results["test"] or []]
    for model in order:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
        figures = [_figure(os.path.join(root, "Forecast_Plots", f"{safe}_test_forecasts.png"), root,
                           f"{model}: test forecasts vs. observed values"),
                   _figure(os.path.join(root, "Forecast_Plots", f"{safe}_future_forecasts.png"), root,
                           f"{model}: forecasts beyond the last observed point")]
        if any(figures):
            results["forecasts"][model] = [f for f in figures if f]
        item = _figure(os.path.join(root, "Explainability", f"{safe}_integrated_gradients.png"), root,
                       f"{model}: integrated gradients")
        if item:
            importance = os.path.join(root, "Explainability", f"{safe}_feature_importance.csv")
            item["importance"] = pd.read_csv(importance).head(8).to_dict("records") if os.path.exists(importance) else []
            results["explain"][model] = item
        path = os.path.join(root, "Models", f"{safe}.zip")
        if os.path.exists(path):
            results["models"].append({"path": _relative(path, root), "name": model, "size": os.path.getsize(path)})
    for name, title, description in (
            ("test_forecasts.csv", "Test forecasts", "Every test point: observed Target and the forecast of each model."),
            ("future_forecasts.csv", "Forecasts beyond the data", "The next horizon after the last point of every test series."),
            ("validation_windows.csv", "Validation windows", "The errors of every model on each rolling-origin window.")):
        path = os.path.join(root, "Forecasts", name)
        if os.path.exists(path):
            results["tables"].append({"path": _relative(path, root), "name": title, "description": description,
                                      "size": os.path.getsize(path)})
    return results


def collect_detection_results(root):
    """What the results page of the detection automator shows, read from the Materials folder."""
    with open(os.path.join(root, "run_info.json")) as f:
        info = json.load(f)
    metrics = info["metrics"]
    results = {"info": info, "automator": get_automator("object-detection"), "metrics": metrics, "test": None,
               "validation": None, "best": None, "curves": [], "detections": {}, "drise": {}, "models": [],
               "predictions": [], "files": []}
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            results["files"].append({"path": _relative(path, root), "size": os.path.getsize(path)})
    results["files"].sort(key=lambda f: f["path"])
    test_path = os.path.join(root, "test_results.xlsx")
    if os.path.exists(test_path):
        test = pd.read_excel(test_path, index_col=0)
        best = {m: test[m].max() for m in metrics}
        results["test"] = [{"model": model, "values": {m: float(row[m]) for m in metrics},
                            "best": {m: bool(row[m] == best[m]) for m in metrics}} for model, row in test.iterrows()]
        top = info.get("best_model")
        if top in test.index:
            results["best"] = {"model": top, "values": {m: float(test.loc[top, m]) for m in metrics}}
    validation_path = os.path.join(root, info.get("validation_file", ""))
    if os.path.isfile(validation_path):
        table = pd.read_excel(validation_path, index_col=0)
        results["validation"] = [{"model": model, "values": {m: (f"{row[m]:.3f}" if isinstance(row[m], float) else str(row[m]))
                                                             for m in metrics}} for model, row in table.iterrows()]
    for name, title in (("test_metrics.png", "Test metrics by network"), ("PR_CURVES.png", "Precision-recall curves"),
                        ("FROC_CURVES.png", "FROC curves"), ("ap_per_class.png", "AP per class")):
        item = _figure(os.path.join(root, "Detection_Curves", name), root, title)
        if item:
            results["curves"].append(item)
    for model in [row["model"] for row in results["test"] or []]:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
        for folder, key, prefix in (("Detections", "detections", ""), ("Explainability", "drise", "D-RISE · ")):
            paths = sorted(glob.glob(os.path.join(root, folder, f"{safe}_*.png")))
            if paths:
                results[key][model] = [{"path": _relative(p, root),
                                        "title": prefix + os.path.basename(p)[len(safe) + 1:-4]} for p in paths]
        exported = info.get("models", {}).get(model)
        if exported and os.path.exists(os.path.join(root, "Models", exported)):
            path = os.path.join(root, "Models", exported)
            results["models"].append({"path": _relative(path, root), "name": model, "size": os.path.getsize(path),
                                      "file": exported, "library": "torchvision" if exported.endswith(".pt") else "transformers"})
        path = os.path.join(root, "Predictions", f"{safe}_test_predictions.csv")
        if os.path.exists(path):
            results["predictions"].append({"path": _relative(path, root), "name": model, "size": os.path.getsize(path)})
    return results


def collect_segmentation_results(root):
    """What the results page of the segmentation automator shows, read from the Materials folder."""
    with open(os.path.join(root, "run_info.json")) as f:
        info = json.load(f)
    metrics = info["metrics"]
    lower = {"HD95", "ASSD"}  # distances: lower is better
    results = {"info": info, "automator": get_automator("image-segmentation"), "metrics": metrics, "lower": sorted(lower),
               "test": None, "validation": None, "best": None, "plots": [], "overlays": {}, "per_class": [],
               "models": [], "predictions": [], "tables": [], "files": []}
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            results["files"].append({"path": _relative(path, root), "size": os.path.getsize(path)})
    results["files"].sort(key=lambda f: f["path"])

    def number(value):
        return float(value) if value == value else float("nan")
    test_path = os.path.join(root, "test_results.xlsx")
    if os.path.exists(test_path):
        test = pd.read_excel(test_path, index_col=0)
        best = {m: (test[m].min() if m in lower else test[m].max()) for m in metrics}
        results["test"] = [{"model": model, "values": {m: number(row[m]) for m in metrics},
                            "best": {m: bool(row[m] == best[m]) for m in metrics}} for model, row in test.iterrows()]
        top = info.get("best_model")
        if top in test.index:
            results["best"] = {"model": top, "values": {m: number(test.loc[top, m]) for m in metrics}}
    validation_path = os.path.join(root, info.get("validation_file", ""))
    if os.path.isfile(validation_path):
        table = pd.read_excel(validation_path, index_col=0)
        results["validation"] = [{"model": model, "values": {m: (f"{row[m]:.3f}" if isinstance(row[m], float) else str(row[m]))
                                                             for m in metrics}} for model, row in table.iterrows()]
    per_class_path = os.path.join(root, "Segmentation_Metrics", "test_per_class.csv")
    if os.path.exists(per_class_path) and len(info.get("classes", [])) > 2:
        per_class = pd.read_csv(per_class_path)
        results["per_class"] = [{"model": model, "values": dict(zip(group["class"], group["Dice"].astype(float)))}
                                for model, group in per_class.groupby("model", sort=False)]
    for name, title in (("test_overlap_metrics.png", "Overlap metrics on the test set"),
                        ("test_distances.png", "Surface distances on the test set"),
                        ("test_dice_per_case.png", "Dice of each test case"),
                        ("test_dice_per_class.png", "Test Dice per class")):
        item = _figure(os.path.join(root, "Segmentation_Plots", name), root, title)
        if item:  # the four-panel overlap figure on a full row, the others two per row
            item["columns"] = "col-12" if name == "test_overlap_metrics.png" else "col-12 col-lg-6"
            results["plots"].append(item)
    for model in [row["model"] for row in results["test"] or []]:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
        paths = glob.glob(os.path.join(root, "Overlays", safe, f"{safe}_*.png"))
        order = {"worst": 0, "median": 1, "best": 2}
        paths.sort(key=lambda p: order.get(os.path.basename(p)[len(safe) + 1:].split("_", 1)[0], 3))
        if paths:
            items = []
            for p in paths:
                rank, case = (os.path.basename(p)[len(safe) + 1:-4].split("_", 1) + [""])[:2]
                items.append({"path": _relative(p, root), "title": f"{rank.capitalize()} test case · {case}"})
            results["overlays"][model] = items
        exported = info.get("models", {}).get(model)
        if exported and os.path.exists(os.path.join(root, "Models", exported["file"])):
            path = os.path.join(root, "Models", exported["file"])
            results["models"].append({"path": _relative(path, root), "name": model, "size": os.path.getsize(path),
                                      "file": exported["file"], "library": exported["library"],
                                      "configuration": exported.get("configuration"), "fold": exported.get("fold")})
        folder = os.path.join(root, "Predictions", safe)
        if os.path.isdir(folder):
            masks = [f for f in os.listdir(folder) if not f.startswith(".")]
            results["predictions"].append({"name": model, "count": len(masks), "folder": _relative(folder, root),
                                           "size": sum(os.path.getsize(os.path.join(folder, f)) for f in masks)})
    for name, title, description in (
            ("test_per_class.csv", "Test metrics per class", "Every network and class: Dice, IoU, HD95, ASSD, sensitivity and precision on Test.zip."),
            ("test_per_case.csv", "Test metrics per case", "Every network, test case and class.")):
        path = os.path.join(root, "Segmentation_Metrics", name)
        if os.path.exists(path):
            results["tables"].append({"path": _relative(path, root), "name": title, "description": description,
                                      "size": os.path.getsize(path)})
    return results


@app.route('/download_predictions/<path:model>')
def download_predictions(model):
    """The predicted test masks of one segmentation network, as a zip."""
    root = materials_dir()
    folder = os.path.realpath(os.path.join(root, "Predictions", model))
    if not folder.startswith(os.path.realpath(os.path.join(root, "Predictions")) + os.sep) or not os.path.isdir(folder):
        abort(404)
    archive = tempfile.TemporaryFile()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as zipf:
        for name in sorted(os.listdir(folder)):
            zipf.write(os.path.join(folder, name), name)
    archive.seek(0)
    return send_file(archive, mimetype='application/zip', as_attachment=True,
                     download_name=f'{os.path.basename(folder)}_test_masks.zip')


def _run_automator(root):
    path = os.path.join(root, "run_info.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("automator", "tabular")
    return "tabular"


@app.route('/results')
def results():
    root = materials_dir()
    if _run_automator(root) == "object-detection":
        return render_template('detection/results.html', results=collect_detection_results(root))
    if _run_automator(root) == "image-segmentation":
        return render_template('segmentation/results.html', results=collect_segmentation_results(root))
    if _run_automator(root) == "time-series-forecasting":
        return render_template('forecasting/results.html', results=collect_forecast_results(root),
                               metrics=FORECAST_METRICS)
    return render_template('results.html', results=collect_results(root),
                           metrics_order=METRICS)


@app.route('/materials/<path:filepath>')
def material(filepath):
    """Serves an output file inline (e.g. a plot shown on the results page)."""
    return send_from_directory(materials_dir(), filepath)


@app.route('/download/<path:filepath>')
def download_file(filepath):
    return send_from_directory(materials_dir(), filepath, as_attachment=True)


@app.route('/download_all')
def download_all():
    # Built in a temporary file rather than in memory: trained networks weigh up to ~100 MB each
    archive = tempfile.TemporaryFile()
    root = materials_dir()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for dirpath, dirs, files in os.walk(root):
            for file in files:
                file_path = os.path.join(dirpath, file)
                # Paths relative to the Materials directory; trained networks are already compressed
                compression = zipfile.ZIP_STORED if file.endswith(('.pt', '.png', '.zip')) else zipfile.ZIP_DEFLATED
                zipf.write(file_path, os.path.relpath(file_path, root), compress_type=compression)
    archive.seek(0)
    return send_file(archive, mimetype='application/zip', as_attachment=True,
                     download_name='pipeline_results.zip')


@app.route('/clear_files', methods=['POST'])
def clear_files():
    if job.running:
        flash('A pipeline is running: its results cannot be deleted now.', 'warning')
        return redirect(url_for('results'))
    clear_materials()
    flash('All result files were deleted.', 'success')
    return redirect(url_for('results'))


def run_pipeline(input_folder, output_folder, params):
    try:
        read_yaml(input_folder)

        # Perform Bias Assessment
        if params["BiasAssessment"]:
            print("------------- \n", " Bias Detection Started \n", "-------------")
            try:
                print("------------- \n", " Bias Detection Started for Train.csv \n", "-------------")
                DBDM.bias_config(
                    file_path=os.path.join(input_folder, "Train.csv"),
                    subgroup_analysis=0,  # default is 0
                    facet=params["Feature"],
                    outcome='Target',
                    subgroup_col='',  # default is ''
                    label_value=1,  # default is 1
                )
                print("------------- \n", " Bias Detection Finished for Train.csv \n", "-------------")
            except Exception as e:
                print(f"Error in bias detection for Train.csv: {e}")
            
            try:
                print("------------- \n", " Bias Detection Started for Test.csv \n", "-------------")
                DBDM.bias_config(
                    file_path=os.path.join(input_folder, "Test.csv"), 
                    subgroup_analysis=0, # default is 0
                    facet=params["Feature"],
                    outcome='Target',
                    subgroup_col='',  # default is ''
                    label_value=1,  # default is 1
                )
                print("------------- \n", " Bias Detection Finished for Test.csv \n", "-------------")
            except Exception as e:
                print(f"Error in bias detection for Test.csv: {e}")

        # Load data
        print("------------- \n", "Loading Data \n", "-------------")
        data_checker = DataChecker(input_folder)

        # Process the data
        try:
            train, test = data_checker.process_data()
            print("Train and Test data processed successfully.")
        except FileNotFoundError as e:
            print(e)
            return f"Error: {e}"
        except ValueError as e:
            print(e)
            return f"Error: {e}"

        X_train = train.drop('Target', axis=1)
        y_train = train['Target']
        # For the code shown on the results page (using the saved models on new data)
        columns = pd.read_csv(os.path.join(input_folder, "Train.csv"), nrows=0).columns
        with open(os.path.join("Materials", "run_info.json"), "w") as f:
            json.dump({"automator": "tabular", "classes": sorted(int(c) for c in y_train.unique()),
                       "id_column": next((c for c in ("ID", "patient_id") if c in columns), None)}, f, indent=2)
        X_test = test.drop('Target', axis=1)
        y_test = test['Target']
        print("------------- \n", "Data Loaded successfully \n", "-------------")

        # Run the pipeline
        print("------------- \n", "Training on K-Fold cross validation \n", "-------------")
        params_dict, scores_storage, thresholds, _ = train_k_fold(X_train, y_train)
        print("------------- \n", "Training on K-Fold cross validation completed successfully \n", "-------------")

        print("------------- \n", "Evaluating algorithms on Test.csv \n", "-------------")
        external_test(X_train, y_train, X_test, y_test, params_dict, thresholds)
        print("Pipeline completed successfully.")
        return "Pipeline completed successfully"
    
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return f"Error: {e}"


if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 5000, application, use_debugger=True, threaded=True)
