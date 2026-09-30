import os
import glob
import io
import logging
import tempfile
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
from web.catalog import AUTOMATORS, MODELS, THRESHOLD_METRICS, get_automator
from web.jobs import PipelineJob, PHASES

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

# Configure upload settings
ALLOWED_EXTENSIONS = {'csv'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max

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
        return redirect(url_for('tabular'))
    return render_template('tabular/run.html', status=job.snapshot(), phases=PHASES)


@app.route('/api/status')
def status():
    return jsonify(job.snapshot())


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

METRICS = ["AUC", "Balanced Accuracy", "F-score", "Accuracy", "Sensitivity", "Specificity"]


def _relative(path, root):
    return os.path.relpath(path, root).replace(os.sep, "/")


def collect_results(root):
    """Everything the results page shows, read from the Materials folder."""
    results = {"test": None, "kfold": None, "kfold_name": None, "best": None, "curves": [],
               "class_curves": [], "confusion": {}, "shap": {}, "models": [], "files": [],
               "skipped": [], "notes": []}
    if not os.path.isdir(root):
        return results

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
        results["models"].append({"path": _relative(path, root),
                                  "name": os.path.basename(path)[:-len("_pipeline.pkl")],
                                  "size": os.path.getsize(path)})

    log_path = os.path.join(root, "error_log.log")
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as log:
            for line in log:
                if "failed and was skipped:" in line:
                    model, reason = line.split(":ERROR:", 1)[-1].split(" failed and was skipped:", 1)
                    results["skipped"].append({"model": model.strip(), "reason": reason.strip()})
    for name in ("Shap_error_log.txt", "Model_save_error_log.txt"):
        if os.path.exists(os.path.join(root, name)):
            results["notes"].append(name)
    return results


@app.route('/results')
def results():
    return render_template('tabular/results.html', results=collect_results(materials_dir()),
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
    # Create a BytesIO object to store the zip file
    memory_file = io.BytesIO()
    root = materials_dir()

    # Create a zip file
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all files in Materials directory
        for dirpath, dirs, files in os.walk(root):
            for file in files:
                file_path = os.path.join(dirpath, file)
                # Calculate path relative to Materials directory for the archive
                zipf.write(file_path, os.path.relpath(file_path, root))

    # Move the cursor to the beginning of the BytesIO object
    memory_file.seek(0)

    # Return the zip file as an attachment
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='pipeline_results.zip'
    )


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
