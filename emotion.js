/* Aryavarta: dedicated MediaPipe Face Landmarker bridge.
   MediaPipe provides face landmarks + blendshape coefficients; the six labels
   below are a coarse expression estimate derived from those signals. */
(() => {
  const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
  const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm';
  const VISION_URLS = [
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/vision_bundle.js',
    'https://unpkg.com/@mediapipe/tasks-vision@0.10.22/vision_bundle.js'
  ];

  let landmarker = null;
  let stream = null;
  let running = false;
  let loading = false;
  let lastTimestamp = 0;
  let lastRun = 0;
  let stable = { name: 'neutral', score: 0.35, candidate: 'neutral', count: 0 };

  const $ = id => document.getElementById(id);
  const labels = { happy:'खुश', sad:'उदास', angry:'गुस्सा', surprised:'आश्चर्य', fearful:'घबराहट', neutral:'सामान्य' };

  function status(text) {
    const el = $('status');
    if (el) el.textContent = text;
  }

  function setUI(name, score) {
    document.querySelectorAll('.emotion').forEach(el => {
      el.classList.toggle('active', el.dataset.emotion === name);
    });
    if ($('expr')) $('expr').textContent = 'भाव अनुमान • ' + (labels[name] || name);
    if ($('confidence')) $('confidence').textContent = 'MediaPipe expression confidence • ' + Math.round(score * 100) + '%';
    if ($('faceBadge')) $('faceBadge').textContent = 'FACE • DETECTED';
    document.documentElement.dataset.aryaExpression = name;
    window.__aryaEmotion = name;
    window.__aryaEmotionConfidence = score;
  }

  function values(result) {
    const m = Object.create(null);
    const list = result?.faceBlendshapes?.[0]?.categories || [];
    for (const c of list) m[c.categoryName] = Number(c.score) || 0;
    return m;
  }

  function avg(m, a, b) { return ((m[a] || 0) + (m[b] || 0)) / 2; }

  // Blendshape-based expression heuristic. It does not claim to measure a person's true emotion.
  function classify(result) {
    const m = values(result);
    if (!Object.keys(m).length) return ['neutral', 0.35];

    const smile = avg(m, 'mouthSmileLeft', 'mouthSmileRight');
    const frown = avg(m, 'mouthFrownLeft', 'mouthFrownRight');
    const browDown = avg(m, 'browDownLeft', 'browDownRight');
    const browUp = ((m.browInnerUp || 0) + (m.browOuterUpLeft || 0) + (m.browOuterUpRight || 0)) / 3;
    const eyeWide = avg(m, 'eyeWideLeft', 'eyeWideRight');
    const eyeSquint = avg(m, 'eyeSquintLeft', 'eyeSquintRight');
    const jaw = m.jawOpen || 0;
    const press = avg(m, 'mouthPressLeft', 'mouthPressRight');
    const stretch = avg(m, 'mouthStretchLeft', 'mouthStretchRight');
    const sneer = avg(m, 'noseSneerLeft', 'noseSneerRight');

    const scores = {
      happy: Math.min(1, smile * 1.15 + eyeSquint * 0.18),
      sad: Math.min(1, frown * 0.95 + browDown * 0.22),
      angry: Math.min(1, browDown * 0.92 + press * 0.25 + sneer * 0.18),
      surprised: Math.min(1, eyeWide * 0.58 + browUp * 0.34 + jaw * 0.72),
      fearful: Math.min(1, eyeWide * 0.45 + browUp * 0.30 + stretch * 0.30),
      neutral: Math.max(0.30, 1 - Math.max(smile, frown, browDown, eyeWide, jaw) * 0.70)
    };

    let best = 'neutral';
    for (const k of Object.keys(scores)) if (scores[k] > scores[best]) best = k;
    let score = scores[best];
    if (best !== 'neutral' && score < 0.40) return ['neutral', 0.45];
    return [best, Math.max(0.35, Math.min(0.99, score))];
  }

  function smooth(name, score) {
    if (name === stable.candidate) stable.count += 1;
    else { stable.candidate = name; stable.count = 1; }
    // Avoid flickering between expressions on individual frames.
    if (name !== stable.name && stable.count < 2) return;
    stable.name = name;
    stable.score = score;
    setUI(name, score);
  }

  async function loadModel() {
    if (landmarker || loading) return landmarker;
    loading = true;
    status('MediaPipe Face Landmarker model लोड हो रहा है…');
    try {
      let vision = null;
      let lastError = null;
      for (const url of VISION_URLS) {
        try {
          const mod = await import(url);
          vision = mod.default || mod;
          if (vision?.FaceLandmarker && vision?.FilesetResolver) break;
        } catch (e) { lastError = e; }
      }
      if (!vision?.FaceLandmarker || !vision?.FilesetResolver) {
        throw lastError || new Error('MediaPipe FaceLandmarker export unavailable');
      }

      const resolver = await vision.FilesetResolver.forVisionTasks(WASM_URL);
      // CPU is intentional here: it is slower than GPU on some devices but avoids
      // WebGL/GPU initialization failures that commonly break mobile browsers.
      landmarker = await vision.FaceLandmarker.createFromOptions(resolver, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        minFaceDetectionConfidence: 0.45,
        minFacePresenceConfidence: 0.45,
        minTrackingConfidence: 0.45
      });
      status('MediaPipe तैयार है • चेहरे के visible expression signals का on-device estimate चल रहा है।');
      return landmarker;
    } catch (e) {
      console.error('Aryavarta MediaPipe error:', e);
      status('MediaPipe model नहीं चल पाया: ' + (e?.message || String(e)).slice(0, 150));
      return null;
    } finally { loading = false; }
  }

  async function start() {
    if (running) return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'user' }, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      });
      const video = $('video');
      video.srcObject = stream;
      video.muted = true;
      await video.play();
      $('camoff')?.classList.add('hidden');
      if ($('camBtn')) $('camBtn').textContent = 'कैमरा बंद करें';
      if ($('visionState')) $('visionState').textContent = 'MediaPipe चालू';
      if ($('mode')) $('mode').textContent = 'WEB MODE • MEDIAPIPE';
      running = true;
      const model = await loadModel();
      if (!model) return;
      requestAnimationFrame(loop);
    } catch (e) {
      console.error('Camera error:', e);
      status('कैमरा permission नहीं मिली या कैमरा उपलब्ध नहीं है।');
    }
  }

  function stop() {
    running = false;
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null;
    const video = $('video');
    if (video) video.srcObject = null;
    $('camoff')?.classList.remove('hidden');
    if ($('camBtn')) $('camBtn').textContent = 'कैमरा शुरू करें';
    if ($('visionState')) $('visionState').textContent = 'कैमरा बंद';
    if ($('faceBadge')) $('faceBadge').textContent = 'FACE • प्रतीक्षा';
    if ($('expr')) $('expr').textContent = 'भाव अनुमान • —';
    if ($('confidence')) $('confidence').textContent = 'मॉडल बंद';
    document.querySelectorAll('.emotion').forEach(e => e.classList.remove('active'));
    window.__aryaEmotion = 'neutral';
    document.documentElement.dataset.aryaExpression = 'neutral';
    if ($('mode')) $('mode').textContent = 'WEB MODE • तैयार';
    status('कैमरा बंद है।');
  }

  function loop(now) {
    if (!running || !landmarker) return;
    const video = $('video');
    if (video && video.readyState >= 2 && video.currentTime > 0 && now - lastRun >= 120) {
      // MediaPipe requires monotonically increasing timestamps for VIDEO mode.
      const timestamp = Math.max(Math.round(now), lastTimestamp + 1);
      lastTimestamp = timestamp;
      lastRun = now;
      try {
        const result = landmarker.detectForVideo(video, timestamp);
        if (result?.faceLandmarks?.length) {
          const [name, score] = classify(result);
          smooth(name, score);
        } else {
          if ($('faceBadge')) $('faceBadge').textContent = 'FACE • खोज रहा है';
        }
      } catch (e) {
        console.warn('MediaPipe frame error:', e);
      }
    }
    requestAnimationFrame(loop);
  }

  function patchChatExpression() {
    const originalFetch = window.fetch.bind(window);
    window.fetch = (input, init = {}) => {
      const url = typeof input === 'string' ? input : input?.url || '';
      if (url.endsWith('/api/chat') && init.body) {
        try {
          const payload = JSON.parse(init.body);
          payload.expression = window.__aryaEmotion || 'neutral';
          init = { ...init, body: JSON.stringify(payload) };
        } catch (_) {}
      }
      return originalFetch(input, init);
    };
  }

  window.addEventListener('DOMContentLoaded', () => {
    patchChatExpression();
    // Run after the existing page module has installed its handlers, then replace
    // only the camera handler. Other chatbot/speech controls remain untouched.
    setTimeout(() => {
      const btn = $('camBtn');
      if (!btn) return;
      btn.onclick = () => running ? stop() : start();
      // Preload the JS/WASM/model only after the user can see the page; camera permission
      // is still requested only when the camera button is pressed.
      status('MediaPipe Face Landmarker ready to start.');
    }, 0);
  });
})();
