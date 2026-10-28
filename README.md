# VoxAssist — Responsive Speech Bot

VoxAssist is an end-to-end Deep Learning course project with two separate response modes: measurable intent classification and optional free-form generation through a local Ollama model. It accepts typed or spoken input and can read responses aloud in the browser.

## What it does

- **Accepts speech or text.** The microphone button uses the browser's Web Speech API when available. Typing remains available as a fallback.
- **Classifies intent with a neural network.** Text is converted into a multi-hot representation of word and two-word phrase features. A multilayer perceptron (MLP) predicts one of the intent categories.
- **Responds in text and optionally in speech.** The browser's SpeechSynthesis API reads the response aloud when Voice Replies is enabled.
- **Supports general-purpose responses.** Ollama can run `qwen3.5:2b` locally; classifier mode remains available for benchmark experiments.
- **Handles uncertainty.** When the highest predicted probability is below the configured threshold, the bot asks the user to rephrase rather than treating every prediction as certain.
- **Includes a training dataset and evaluation script.** The intent patterns and response text are editable in `intents.json`. Training writes a model, metadata, and metrics to `artifacts/`.
- **Adapts to desktop and mobile screens.** The frontend uses plain HTML, CSS, and JavaScript; no Node.js build step is needed.

> **Important speech/privacy note:** Neural-network inference runs in the local Flask process. Browser speech recognition may use an online service depending on the browser, so audio processing is not guaranteed to be offline. This demo does not save chat history or microphone recordings.

## Architecture

```text
Microphone ──> Browser Speech Recognition ──> Transcribed text ──┐
                                                                  │
Keyboard ─────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                      Flask POST /api/chat
                                  │
                                  ▼
                  Tokenize words and word bigrams
                                  │
                                  ▼
                     Multi-hot input feature vector
                                  │
                                  ▼
                  MLP: Dense(128) → Dense(64) → logits
                                  │
                                  ▼
                  Softmax + confidence threshold
                                  │
                                  ▼
                    Select response for predicted intent
                                  │
                         ┌────────┴────────┐
                         ▼                 ▼
                    Text response    Browser text-to-speech
```

The speech input and output components run in the browser. The intent classifier runs in Python using PyTorch. Flask serves both the frontend and the JSON API.

## Project structure

```text
voxassist/
├── app.py                         # Flask server and /api/chat endpoint
├── train.py                       # Trains, evaluates, and saves the neural model
├── model_utils.py                 # Shared tokenizer, vectorizer, and MLP definition
├── intents.json                   # Intent labels, example phrases, and responses
├── requirements.txt               # Python dependencies
├── frontend/
│   ├── templates/index.html       # Main responsive interface
│   └── static/
│       ├── css/styles.css         # Layout, components, animation, responsive rules
│       └── js/app.js              # Chat, microphone, fetch API, and speech synthesis
├── artifacts/
│   ├── model.pt                   # Trained PyTorch state dictionary
│   ├── metadata.json              # Vocabulary, labels, responses, configuration
│   └── metrics.json               # Held-out and training metrics from train.py
└── tests/
    └── test_model.py              # Basic model and artifact checks
```

## Training dataset (`intents.json`)

The training dataset is the editable file **`intents.json`** in the project root. It is a small, hand-written educational text dataset created for this demo; it is not a downloaded benchmark dataset and it does not contain audio recordings. The browser turns speech into text first, then the neural network classifies that text.

### Dataset summary

| Property | Value |
|---|---:|
| Intent categories | 22 |
| Example text patterns | 352 |
| Patterns per intent | 16 |
| Evaluation holdout | 66 patterns (about 20%) |
| Patterns used to fit the evaluation model | 286 |

Each intent contains a `tag`, a list of example `patterns`, and a list of possible `responses`. The `fallback_responses` list at the top level supplies a reply when the model's confidence is below its threshold. Example structure:

```json
{
  "intents": [
    {
      "tag": "deep_learning",
      "patterns": [
        "what is deep learning",
        "define deep learning",
        "explain deep learning simply"
      ],
      "responses": [
        "Deep learning uses neural networks with multiple layers to learn patterns from data."
      ]
    }
  ],
  "fallback_responses": [
    "I am not sure I understood that. Could you rephrase it?"
  ]
}
```

The example above is abbreviated for illustration. The actual file contains 22 intents and 16 patterns for each intent. The intent tags are:

| Intent tag | What it covers |
|---|---|
| `greeting` | Hello and other greetings |
| `goodbye` | Ending a conversation |
| `thanks` | Thank-you messages |
| `bot_identity` | The bot's name and identity |
| `capabilities` | What the bot can do |
| `project_overview` | Overview of VoxAssist |
| `deep_learning` | Deep learning concepts |
| `neural_network` | Neural-network concepts |
| `intent_classification` | How a message is assigned an intent |
| `speech_recognition` | Speech-to-text input |
| `text_to_speech` | Spoken bot responses |
| `model_training` | Training process and algorithm |
| `dataset` | Dataset-related questions |
| `study_tips` | General study advice |
| `exam_prep` | Exam preparation |
| `project_report` | Project report help |
| `microphone_troubleshooting` | Microphone issues |
| `browser_support` | Supported browser questions |
| `privacy` | Chat and audio privacy questions |
| `joke` | Jokes |
| `motivation` | Encouragement and motivation |
| `getting_started` | How to start using the bot |

### How the training data is used

1. `train.py` reads each non-empty string in every `patterns` list and assigns it the numeric class of its intent tag.
2. Text is converted into binary multi-hot features based on individual words (unigrams) and adjacent two-word phrases (bigrams).
3. A stratified, seeded 80/20 split is used to evaluate a model on held-out examples. With 352 patterns, this run evaluates on 66 examples and fits the evaluation model on the other 286.
4. After evaluation, the deployable model is trained again on all 352 labelled patterns and saved under `artifacts/`.
5. The model predicts one of the 22 intent classes. The app chooses a response associated with that tag, or returns a fallback when confidence is low.

### Editing and extending the dataset

To teach the bot a new way of asking an existing question, add a natural, varied sentence to that intent's `patterns` array. To introduce a new intent, add a new object with a unique `tag`, representative `patterns`, and suitable `responses`. If adding an intent, make sure the JSON remains valid and then retrain.

After changing the dataset, run from the project root:

```powershell
python train.py
```

The script regenerates `artifacts/model.pt`, `artifacts/metadata.json`, and `artifacts/metrics.json`. Avoid using near-identical patterns across different tags because that can make classes ambiguous. Add varied examples for every intent and examine per-intent errors before relying on an overall accuracy figure.

**Current limitation:** the included holdout result is about 50% accuracy, while the final model's training accuracy is 100%. That gap suggests overfitting and limited phrase diversity. The dataset is appropriate for demonstrating the end-to-end workflow, but is too small to claim robust real-world intent understanding.

## Requirements

- Windows 10/11, macOS, or Linux
- Python **3.11 or 3.12** recommended (the provided Windows script supports both)
- A modern browser; **Chrome or Microsoft Edge** is recommended for microphone support
- A working microphone for voice input (optional)
- Internet access may be needed to install dependencies and for browser speech recognition, depending on browser settings

## Run it on Windows 11

**Quick setup option:** after extracting the ZIP, double-click `setup_and_run_windows.bat`. It uses Python 3.11 if installed, otherwise Python 3.12; it creates a virtual environment, installs CPU PyTorch and Flask, trains the model if the artifacts are missing, and starts the web server. The first setup needs internet access. You can also follow the manual steps below for more control.

Open **Command Prompt** or **PowerShell** and follow these steps.

### 1. Extract the project

Extract the ZIP archive. Open a terminal in the extracted `voxassist` directory—the folder that contains `app.py` and `train.py`.

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your `python --version` output is Python 3.12.4, so the commands above will use that Python installation. If PowerShell blocks environment activation, you can run the environment's Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Alternatively, in Command Prompt you can activate it with `.venv\Scripts\activate.bat`. If you have multiple Python versions and need to choose explicitly, `py -0p` lists installed versions and `py -3.12 -m venv .venv` creates the environment with Python 3.12.

### 3. Install PyTorch CPU and Flask

For a regular CPU-only laptop, install the CPU build of PyTorch first, then the remaining dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

This project does not require a CUDA-capable GPU. Training is small enough for a typical laptop CPU, although runtime depends on the computer.

### 4. Train the intent model

```powershell
python train.py
```

The script reads `intents.json`, creates a reproducible stratified holdout evaluation, trains the final model on all labelled patterns, then writes these files under `artifacts/`:

- `model.pt` — neural-network weights
- `metadata.json` — feature vocabulary, intent labels, responses, and inference threshold
- `metrics.json` — dataset size, held-out accuracy, training accuracy, and loss information

The ZIP includes a trained model so the app can be tried quickly, but running `train.py` is recommended when you change the data or want to reproduce the current metrics.

### 5. Start the web app

```powershell
python app.py
```

You should see a message containing:

```text
Starting VoxAssist at http://127.0.0.1:5000
```

Keep this terminal open while using the app. Open **http://127.0.0.1:5000** in Chrome or Edge.

### 6. Try it

Try typing one of these prompts or selecting a quick-prompt chip:

- `Explain deep learning`
- `How is the model trained?`
- `What is speech recognition?`
- `Explain this project`
- `How do I fix my mic?`
- `Help me prepare for exams`

Select the microphone button and allow microphone access to use voice input. Use the **Voice replies** toggle to turn spoken answers on or off. If microphone input is unavailable, typing still works.

### 7. Stop the server

Return to the terminal and press **Ctrl+C**.

## Common troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**

Activate `.venv` in the same terminal and run the installation commands again. Ensure that `python` refers to the virtual environment (`where python` in Command Prompt or `Get-Command python` in PowerShell).

**`ModuleNotFoundError: No module named 'flask'`**

Run `python -m pip install -r requirements.txt` after activating the virtual environment.

**The page says the model is unavailable**

Run `python train.py` from the project root, wait for the final metrics, then restart `python app.py`. The backend intentionally shows a clear setup error when model artifacts are missing.

**The microphone button is disabled or doesn't work**

Use a recent Chrome or Edge browser, visit the app through `http://127.0.0.1:5000` (localhost is treated as a secure context by modern browsers), and allow microphone permission. Check Windows **Settings → Privacy & security → Microphone** and confirm the correct input device is selected. If voice recognition service is unavailable, type into the message box.

**The bot gives a fallback response**

This is intentional when the model's highest class probability is below the confidence threshold. Try a shorter, more specific phrase. To teach the bot a new phrasing, add the phrase to the appropriate `patterns` list in `intents.json` and run `python train.py` again.

**Port 5000 is already in use**

Stop the other app using that port, or change `port=5000` in the last lines of `app.py` and open the matching local URL. If changing ports, update any instructions you use for the browser address.

## Customize the bot

Every item in `intents.json` has:

- `tag`: the intent's internal label, such as `speech_recognition`
- `patterns`: sample utterances that should map to this intent
- `responses`: response variations the app can choose from

For best results, give each intent **many varied, representative patterns**. Avoid putting nearly identical phrases in different intents. If two intents are routinely confused, clarify their definitions or combine them. After editing the file, rerun `python train.py` to regenerate the artifacts.

The confidence threshold is stored in `artifacts/metadata.json` as `confidence_threshold`; adjust the value in `train.py` if you want a different default, then retrain. A higher threshold makes the system more cautious but produces more fallback replies.

## Model and evaluation details

- **Task:** supervised text intent classification
- **Input representation:** binary multi-hot vector of unigrams and adjacent word bigrams
- **Network:** fully connected layers with 128 and 64 hidden units, ReLU activations, and dropout
- **Output:** one logit per intent; softmax is used at inference to calculate class probabilities
- **Loss:** cross-entropy
- **Optimizer:** Adam with learning rate 0.003 and weight decay 0.0001
- **Random seed:** 42 for reproducible data split and training initialization
- **Current dataset:** 352 hand-written example utterances across 22 intent categories

The current small dataset produces a held-out accuracy of around **50%** in the included stratified holdout experiment (see the exact value in `artifacts/metrics.json`). The final model fits the training examples very well, so the gap between training and held-out performance is a sign of overfitting and limited phrase coverage. This is a useful and honest limitation to discuss in a course report. Improve it by collecting more varied labelled utterances, reviewing per-intent confusion, and comparing models or feature representations. Do not present the current result as production-grade accuracy.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serves the web interface |
| `GET` | `/api/health` | Reports whether the model loaded |
| `POST` | `/api/chat` | Accepts JSON `{ "message": "..." }` and returns the response and classification metadata |

Example request body:

```json
{"message": "Explain deep learning"}
```

Successful responses include `response`, `intent`, `intent_label`, `confidence`, and `recognized`.

## Tests

After installing the dependencies and training the model, run:

```powershell
python -m unittest discover -s tests -v
```

The suite includes the original model checks, CLINC150 split and artifact checks, Flask compatibility checks, and mocked Ollama provider, timeout, unavailable-service, retrieval-formatting, and response-cleaning tests. The current checkout passes 13 tests.

## CLINC150 benchmark

The optional benchmark uses the official CLINC150/OOS release from the [clinc/oos-eval repository](https://github.com/clinc/oos-eval), specifically `data/data_full.json`, saved as `data/clinc150/data_full.json`. The dataset is distributed under CC BY 4.0; cite Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction* (EMNLP 2019), and retain the attribution when redistributing it.

Download and validate the six official splits without repartitioning them:

```powershell
python download_clinc150.py
python train_benchmark.py
```

`train_benchmark.py` fits both models on `train` plus `oos_train`, selects the OOS cutoff using only `val` plus `oos_val`, and evaluates once on `test` plus `oos_test`. It writes `artifacts/comparison_metrics.json` and `artifacts/comparison_metrics.csv`, along with `clinc_mlp_model.pt`, `clinc_mlp_metadata.json`, `clinc_svm.joblib`, and `clinc_svm_metadata.json`. The MLP reuses the existing 128/64/ReLU/dropout architecture with training-only binary word and bigram features. The baseline is `TfidfVectorizer(ngram_range=(1, 2))` plus `LinearSVC`.

The recorded final test results from this checkout are:

| model | in-domain accuracy | overall accuracy | macro F1 | OOS precision / recall / F1 | OOS false rejection |
|---|---:|---:|---:|---:|---:|
| MLP | 86.53% | 77.67% | 0.825 | 0.737 / 0.378 / 0.500 | 3.00% |
| TF-IDF + LinearSVC | 90.38% | 83.09% | 0.867 | 0.813 / 0.503 / 0.621 | 2.58% |

The MLP took about 59.7 seconds to train and its saved benchmark artifacts total about 16.2 MB; SVM training took about 2.2 seconds and its artifacts total about 36.8 MB. These figures are CPU and environment dependent. The full confusion summaries and exact values are in the JSON report.

Select a benchmark model without changing the frontend or API contract:

```powershell
$env:VOXASSIST_MODEL = "mlp"  # or "svm"
python app.py
```

Unset `VOXASSIST_MODEL` to use the original `model.pt` classifier. MLP softmax confidence and SVM decision scores are model-specific scores, not directly comparable probabilities. CLINC intents without authored entries in `intents.json` receive a transparent generic response; the classifier does not generate factual answers.

## Ideas for future work

- Grow and balance the dataset using real, varied user utterances.
- Add a confusion matrix, per-class precision/recall/F1, and a separate final test set.
- Compare the MLP against an LSTM or a small transformer encoder.
- Support additional languages and adjustable speech-recognition locales.
- Add an optional local speech-recognition engine for offline transcription.
- Add session export only with an explicit user choice and clear privacy controls.

## Expanding the response domain

`intents.json` remains the small authored-response set used by the original demo. CLINC150 expands intent coverage for benchmarking, but it is still an intent dataset rather than a general knowledge source. For broader questions, a stronger next step is a pretrained text encoder such as a sentence-transformer: fine-tune it on labelled intents, or retrieve the closest approved answer from a larger domain-specific FAQ/document collection. Keep retrieval separate from classification, require a confidence or similarity threshold, and return a fallback when no source is sufficiently relevant. This avoids presenting an intent label as if it were factual knowledge. A production version should also add domain-specific training data and a held-out test set instead of relying only on generic public intents.

## General-purpose Ollama responses

Classifier mode remains the default and continues to use the original MLP, CLINC150 MLP, or CLINC150 SVM selected with `VOXASSIST_MODEL`. For free-form responses, install [Ollama](https://ollama.com/download), start it, and download the configured local model:

```powershell
ollama pull qwen3.5:2b
$env:VOXASSIST_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "qwen3.5:2b"
python app.py
```

The provider calls Ollama's local `/api/chat` endpoint with a timeout. If Ollama is stopped, the model is missing, or the request times out, `/api/chat` returns HTTP 503 with a setup error instead of pretending a classifier produced the answer. LLM responses use `intent: "llm_response"`, `provider: "ollama"`, and `confidence: null`; they are free-form generation and must not be interpreted as intent-classification scores.

Optional retrieval can ground responses in approved local `.txt` and `.md` files. Put those files under `knowledge/`, install the optional dependency, and enable it:

```powershell
python -m pip install -r requirements-llm.txt
$env:VOXASSIST_RETRIEVAL = "1"
python app.py
```

Retrieval uses `sentence-transformers/all-MiniLM-L6-v2` on a small in-memory passage list. It is separate from CLINC150 training and does not alter classifier metrics. The prompt tells the model to use supplied evidence, distinguish unsupported claims, and admit when evidence is insufficient. Set `VOXASSIST_RETRIEVAL=0` (or unset it) to use Ollama without retrieval. Local prompts and retrieved text are sent to the local Ollama service; browser speech recognition may still use a browser vendor's online service.

## Current interface behavior

The web page is intentionally minimal: it contains the conversation session, quick prompts, text composer, microphone button, send button, clear button, and Voice Replies toggle. Training metrics and model comparison results are not shown in the chat interface. The server removes Markdown punctuation such as `**`, `*`, backticks, and the word `asterisk` before returning responses and before browser speech synthesis, so generated formatting is not spoken as punctuation names.

## Complete Windows setup

Use Python 3.12 from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For local Ollama mode, install Ollama, restart PowerShell so `ollama` is on `PATH`, and run:

```powershell
ollama pull qwen3.5:2b
$env:VOXASSIST_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "qwen3.5:2b"
$env:OLLAMA_TIMEOUT_SECONDS = "120"
python app.py
```

Ollama mode returns `intent: "llm_response"` and `confidence: null`; those fields must not be interpreted as classifier metrics. If Ollama is unavailable, the API returns HTTP 503 with a setup message. The existing classifier and benchmark artifacts remain usable without Ollama.
