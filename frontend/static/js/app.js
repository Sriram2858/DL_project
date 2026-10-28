(() => {
  'use strict';

  const form = document.getElementById('chatForm');
  const input = document.getElementById('messageInput');
  const messageList = document.getElementById('messageList');
  const conversationScroll = document.getElementById('conversationScroll');
  const sendButton = document.getElementById('sendButton');
  const micButton = document.getElementById('micButton');
  const listeningBanner = document.getElementById('listeningBanner');
  const listeningText = document.getElementById('listeningText');
  const speechSupportNote = document.getElementById('speechSupportNote');
  const voiceReplyToggle = document.getElementById('voiceReplyToggle');
  const clearChatButton = document.getElementById('clearChatButton');
  const cancelListeningButton = document.getElementById('cancelListeningButton');
  const quickPrompts = document.getElementById('quickPrompts');
  const suggestionIntro = document.getElementById('suggestionIntro');
  const systemStatus = document.getElementById('systemStatus');
  const systemStatusText = document.getElementById('systemStatusText');
  const modelStatusRow = document.querySelector('.model-status-row');
  const modelStatusText = document.getElementById('modelStatusText');
  const modelStatusDetail = document.getElementById('modelStatusDetail');
  const modelStatusCheck = document.getElementById('modelStatusCheck');

  let recognition = null;
  let isListening = false;
  let requestInProgress = false;
  let voiceRepliesEnabled = true;
  let currentInterimTranscript = '';

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.lang = 'en-IN';
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      isListening = true;
      currentInterimTranscript = '';
      micButton.classList.add('is-listening');
      micButton.setAttribute('aria-label', 'Stop voice input');
      listeningBanner.hidden = false;
      listeningText.textContent = 'Listening… speak now';
      speechSupportNote.textContent = 'MICROPHONE ACTIVE';
    };

    recognition.onresult = (event) => {
      let interim = '';
      let finalText = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const piece = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalText += piece;
        else interim += piece;
      }
      currentInterimTranscript = finalText || interim;
      input.value = currentInterimTranscript;
      listeningText.textContent = interim ? `Hearing: “${interim}”` : 'Processing your speech…';
      if (finalText.trim()) {
        stopListeningUI();
        input.value = finalText.trim();
        sendMessage(finalText.trim());
      }
    };

    recognition.onerror = (event) => {
      const messages = {
        'not-allowed': 'Microphone blocked — allow access in browser settings.',
        'service-not-allowed': 'Speech service unavailable. You can type instead.',
        'no-speech': 'No speech detected. Try again or type a message.',
        'audio-capture': 'No microphone found. Check your input device.',
        'network': 'Speech service needs a network connection. Try typing instead.',
        'aborted': 'Voice input cancelled.'
      };
      const note = messages[event.error] || `Voice input error (${event.error}). Try typing instead.`;
      listeningText.textContent = note;
      speechSupportNote.textContent = 'TEXT INPUT AVAILABLE';
      stopListeningUI();
      if (event.error !== 'aborted') showToast(note);
    };

    recognition.onend = () => {
      stopListeningUI();
    };
    speechSupportNote.textContent = 'VOICE INPUT AVAILABLE';
  } else {
    micButton.disabled = true;
    micButton.title = 'Speech recognition is not supported in this browser';
    micButton.style.opacity = '.45';
    speechSupportNote.textContent = 'TYPE INPUT · MIC UNSUPPORTED';
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[character]);
  }

  function cleanResponseText(value) {
    return String(value || '').replace(/[`*_~]/g, '').replace(/\basterisks?\b/gi, '').replace(/[ \t]{2,}/g, ' ').trim();
  }

  function timestamp() {
    return new Intl.DateTimeFormat([], { hour: '2-digit', minute: '2-digit' }).format(new Date());
  }

  function scrollToLatest() {
    window.requestAnimationFrame(() => {
      conversationScroll.scrollTop = conversationScroll.scrollHeight;
    });
  }

  function addMessage(role, text, metadata = null) {
    const isBot = role === 'bot';
    const row = document.createElement('div');
    row.className = `message-row ${isBot ? 'bot-row' : 'user-row'}`;
    const avatar = isBot
      ? '<div class="avatar bot-avatar" aria-hidden="true"><svg viewBox="0 0 28 28"><path d="M14 5v3M6.5 12v4M10.5 9v10M14 7v14M17.5 9v10M21.5 12v4" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round"/></svg></div>'
      : '<div class="avatar user-avatar" aria-hidden="true">YOU</div>';
    const name = isBot ? 'Assistant' : 'YOU';
    const subtitle = isBot ? 'AI ASSISTANT' : 'MESSAGE SENT';
    let extra = '';
    if (isBot && metadata) {
      const detail = metadata.recognized
        ? `${metadata.intent_label} · ${Math.round(metadata.confidence * 100)}% confidence`
        : `Fallback · ${Math.round(metadata.confidence * 100)}% top score`;
      extra = `<div class="intent-chip"><i class="confidence-dot"></i>${escapeHtml(detail)}</div>`;
    }
    const displayText = cleanResponseText(text);
    row.innerHTML = `${avatar}<div class="message-content"><div class="message-meta"><strong>${name}</strong><span>${subtitle}</span></div><div class="bubble ${isBot ? 'bot-bubble' : 'user-bubble'}">${escapeHtml(displayText)}<span class="bubble-time">${timestamp()}</span></div>${extra}</div>`;
    messageList.appendChild(row);
    scrollToLatest();
    return row;
  }

  function addTypingIndicator() {
    const row = document.createElement('div');
    row.className = 'message-row bot-row';
    row.id = 'typingIndicator';
    row.innerHTML = '<div class="avatar bot-avatar" aria-hidden="true"><svg viewBox="0 0 28 28"><path d="M14 5v3M6.5 12v4M10.5 9v10M14 7v14M17.5 9v10M21.5 12v4" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round"/></svg></div><div class="message-content"><div class="message-meta"><strong>Assistant</strong><span>THINKING</span></div><div class="bubble typing-bubble" aria-label="Assistant is thinking"><i></i><i></i><i></i></div></div>';
    messageList.appendChild(row);
    scrollToLatest();
    return row;
  }

  function speak(text) {
    if (!voiceRepliesEnabled || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(cleanResponseText(text));
    utterance.lang = 'en-IN';
    utterance.rate = 1.0;
    utterance.pitch = 1.02;
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find((voice) => /^en-IN$/i.test(voice.lang))
      || voices.find((voice) => /^en-(GB|US|AU)$/i.test(voice.lang));
    if (preferred) utterance.voice = preferred;
    window.speechSynthesis.speak(utterance);
  }

  function stopListeningUI() {
    isListening = false;
    micButton.classList.remove('is-listening');
    micButton.setAttribute('aria-label', 'Start voice input');
    listeningBanner.hidden = true;
    if (recognition) speechSupportNote.textContent = 'VOICE INPUT AVAILABLE';
  }

  function stopListening() {
    if (recognition && isListening) recognition.stop();
    stopListeningUI();
  }

  function showToast(message) {
    let toast = document.getElementById('appToast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'appToast';
      toast.style.cssText = 'position:fixed;z-index:20;bottom:22px;left:50%;transform:translateX(-50%);max-width:calc(100vw - 28px);padding:11px 15px;border:1px solid rgba(237,171,255,.28);border-radius:10px;background:#1a2038;color:#e5e9ff;font:11px Manrope,sans-serif;box-shadow:0 12px 40px rgba(0,0,0,.35);';
      document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.style.display = 'block';
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => { toast.style.display = 'none'; }, 4200);
  }

  async function sendMessage(rawText) {
    const message = String(rawText || '').trim();
    if (!message || requestInProgress) return;
    if (message.length > 500) {
      showToast('Please keep messages under 500 characters.');
      return;
    }
    stopListening();
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    input.value = '';
    suggestionIntro.hidden = true;
    quickPrompts.hidden = true;
    addMessage('user', message);
    requestInProgress = true;
    sendButton.disabled = true;
    input.disabled = true;
    const typing = addTypingIndicator();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
      const data = await response.json();
      typing.remove();
      if (!response.ok) throw new Error(data.error || 'The assistant could not process that message.');
      addMessage('bot', data.response, data);
      speak(data.response);
    } catch (error) {
      typing.remove();
      addMessage('bot', error.message || 'I could not connect to the local server. Please check that Flask is running.');
      showToast('Could not reach the assistant. Check the terminal running app.py.');
    } finally {
      requestInProgress = false;
      sendButton.disabled = false;
      input.disabled = false;
      input.focus();
      scrollToLatest();
    }
  }

  async function checkHealth() {
    try {
      const response = await fetch('/api/health', { cache: 'no-store' });
      const data = await response.json();
      if (response.ok && data.model_loaded) {
        systemStatusText.textContent = 'System online';
        systemStatus.classList.remove('is-error');
        modelStatusText.textContent = 'Neural model ready';
        modelStatusDetail.textContent = `${data.intents} intent categories loaded locally`;
        modelStatusCheck.textContent = '✓';
        modelStatusRow.classList.remove('is-error');
      } else {
        throw new Error('Model not trained yet');
      }
    } catch (_error) {
      systemStatusText.textContent = 'Setup needed';
      systemStatus.classList.add('is-error');
      modelStatusText.textContent = 'Model not available';
      modelStatusDetail.textContent = 'Run python train.py, then restart app.py';
      modelStatusCheck.textContent = '!';
      modelStatusRow.classList.add('is-error');
    }
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    sendMessage(input.value);
  });

  micButton.addEventListener('click', () => {
    if (!recognition) {
      showToast('Speech input is not supported in this browser. You can type a message instead.');
      return;
    }
    if (isListening) {
      stopListening();
      return;
    }
    try {
      recognition.start();
    } catch (_error) {
      showToast('Voice input is already starting. Please try again in a moment.');
    }
  });

  cancelListeningButton.addEventListener('click', () => {
    if (recognition) recognition.abort();
    stopListeningUI();
    input.value = '';
  });

  voiceReplyToggle.addEventListener('click', () => {
    voiceRepliesEnabled = !voiceRepliesEnabled;
    voiceReplyToggle.classList.toggle('is-on', voiceRepliesEnabled);
    voiceReplyToggle.setAttribute('aria-pressed', String(voiceRepliesEnabled));
    if (!voiceRepliesEnabled && 'speechSynthesis' in window) window.speechSynthesis.cancel();
    showToast(voiceRepliesEnabled ? 'Spoken replies enabled.' : 'Spoken replies disabled.');
  });

  quickPrompts.addEventListener('click', (event) => {
    const button = event.target.closest('[data-prompt]');
    if (button) sendMessage(button.dataset.prompt);
  });

  clearChatButton.addEventListener('click', () => {
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    messageList.replaceChildren();
    quickPrompts.hidden = false;
    suggestionIntro.hidden = false;
    showToast('Conversation cleared. Nothing was saved.');
    input.focus();
  });

  checkHealth();
})();
