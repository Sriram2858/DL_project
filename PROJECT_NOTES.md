# VoxAssist — project presentation notes

## Problem statement
Many basic chat interfaces require users to type their requests. This prototype explores a responsive, speech-friendly interface in which a spoken request is transcribed, classified into an intent, and answered in text and optional audio.

## Objective
Build an end-to-end demo that combines a browser speech interface, a trainable neural-network intent classifier, response selection, and a responsive frontend.

## Main modules
1. **Frontend:** HTML/CSS/JavaScript conversation interface with responsive layout, quick prompts, microphone state, optional voice replies, and clear errors.
2. **Speech-to-text:** Browser Web Speech API. This is a platform capability, not a custom neural acoustic model trained by this project.
3. **Text representation:** Tokenization plus binary unigram and adjacent-bigram features.
4. **Deep-learning classifier:** PyTorch MLP with hidden dimensions 128 and 64, ReLU activations, dropout, cross-entropy loss, and Adam optimizer.
5. **Response selection:** A predicted intent selects one of the authored responses in `intents.json`. Low-confidence predictions trigger a fallback or, in optional hybrid mode, Ollama.
6. **Text-to-speech:** Browser SpeechSynthesis API, with a UI control to disable spoken replies.

## Dataset
`intents.json` contains hand-written patterns and response variants for 50 categories. It is an educational dataset created for this prototype, not a benchmark dataset. Patterns should be expanded and balanced before interpreting results as a general measure of language understanding.

## Evaluation
`train.py` makes a reproducible stratified holdout split, trains an evaluation model only on the training partition, and computes holdout accuracy. It then trains the deployable model on all patterns and stores pattern-specific response maps in metadata. The current 50-intent run achieves 36.2% holdout accuracy and 83.7% final training accuracy, indicating limited generalization; use this as a real limitation rather than reporting training accuracy as general performance.

## Limitations to explain
- The intent model can only recognize 50 predefined intent categories.
- Word-feature classification is not the same as open-ended language understanding or a large language model.
- Unseen phrasings can be misclassified; confidence thresholding reduces but does not eliminate this issue.
- Browser speech recognition support and privacy behavior depend on the browser. Voice transcription may require a network service.
- Speech synthesis uses browser-provided voices; voice quality and availability vary.

## Responsible future improvements
- Collect additional utterances from multiple speakers and writing styles.
- Use a separate untouched test set after tuning; include precision, recall, macro-F1, and a confusion matrix.
- Compare against an LSTM or transformer encoder using the same split.
- Consider a local ASR engine for offline use, additional languages, and stronger privacy.

## CLINC150 benchmark extension

The project includes `download_clinc150.py` and `train_benchmark.py`. The official CLINC150 `data_full.json` is stored under `data/clinc150/` and its `train`, `val`, `test`, `oos_train`, `oos_val`, and `oos_test` splits are preserved exactly. Training uses the official training and OOS-training records; validation records select the model's OOS rejection threshold; the official test records are reserved for final metrics. No test repartitioning or leakage is performed.

The existing PyTorch MLP is compared with a scikit-learn TF-IDF word unigram/bigram `LinearSVC` baseline on identical records. Metrics include in-domain and overall accuracy, macro precision/recall/F1, OOS precision/recall/F1, false rejection rate, top confusion pairs, training time, and artifact size. Results are saved in `artifacts/comparison_metrics.json` and `.csv`. Softmax values from the MLP and SVM decision scores are not probabilities and should not be compared numerically.

CLINC150 is the OOS evaluation dataset released by Larson et al., “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction” (EMNLP 2019), under CC BY 4.0. Keep this attribution and the repository citation with copies of the data. The classifier only identifies intent labels; it does not provide factual knowledge. Authored responses remain in `intents.json`, and benchmark-only intents use an explicit generic response.

## Free-form response provider

`VOXASSIST_PROVIDER=ollama` enables a separate response-generation path backed by Ollama's local HTTP chat API. The default model is `qwen3.5:2b`, configurable with `OLLAMA_MODEL`; the default provider remains `classifier` for compatibility. Classifier benchmark metrics describe intent prediction only. They are not measurements of LLM response quality. LLM responses are marked with `intent: "llm_response"`, a provider field, and no classifier confidence.

`VOXASSIST_PROVIDER=hybrid` is an optional router. It keeps authored responses for tags present in `intents.json`, sends unmatched CLINC150 intents and low-confidence cases to Ollama, and returns a classifier fallback if Ollama is unavailable. This allows wider conversations without pretending that a CLINC150 label is a factual answer.

Optional retrieval reads approved Markdown and text files from `knowledge/` and uses `sentence-transformers/all-MiniLM-L6-v2` to select a few relevant passages. This knowledge retrieval is not used to fit either benchmark classifier. The generated prompt instructs Ollama to ground claims in supplied passages and acknowledge missing evidence. Ollama, the embedding model, and browser speech services may process local user content, so users should avoid sensitive data and review their local service configuration.
