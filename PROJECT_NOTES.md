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
5. **Response selection:** A predicted intent selects one of its authored responses. Low-confidence predictions trigger a fallback.
6. **Text-to-speech:** Browser SpeechSynthesis API, with a UI control to disable spoken replies.

## Dataset
`intents.json` contains hand-written patterns and response variants for 22 categories. It is an educational dataset created for this prototype, not a benchmark dataset. Patterns should be expanded and balanced before interpreting results as a general measure of language understanding.

## Evaluation
`train.py` makes a reproducible stratified holdout split, trains an evaluation model only on the training partition, and computes holdout accuracy. It then trains the deployable model on all patterns. The exact held-out accuracy, train accuracy, dataset sizes, and losses are written to `artifacts/metrics.json`. With the current small dataset, the holdout result is about 50%, indicating limited generalization and overfitting. Use this as a real limitation in the report rather than reporting training accuracy as the model's general performance.

## Limitations to explain
- The intent model can only recognize 22 predefined intent categories.
- Word-feature classification is not the same as open-ended language understanding or a large language model.
- Unseen phrasings can be misclassified; confidence thresholding reduces but does not eliminate this issue.
- Browser speech recognition support and privacy behavior depend on the browser. Voice transcription may require a network service.
- Speech synthesis uses browser-provided voices; voice quality and availability vary.

## Responsible future improvements
- Collect additional utterances from multiple speakers and writing styles.
- Use a separate untouched test set after tuning; include precision, recall, macro-F1, and a confusion matrix.
- Compare against an LSTM or transformer encoder using the same split.
- Consider a local ASR engine for offline use, additional languages, and stronger privacy.
