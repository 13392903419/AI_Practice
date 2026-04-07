// static/main.js

// ===== 用户交互时预解锁音频自动播放 =====
document.addEventListener('click', function _unlockAudio() {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const buf = ctx.createBuffer(1, 1, 22050);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start(0);
  ctx.resume().then(() => console.log('[Audio] 自动播放已解锁'));
  document.removeEventListener('click', _unlockAudio);
}, { once: true });

// ================= 摄像头 + ASR =================
(() => {
  const $camStatus = document.getElementById('camStatus');
  const $asrStatus = document.getElementById('asrStatus');
  const $partial   = document.getElementById('partial');
  const $finalList = document.getElementById('finalList');
  const $btnClear  = document.getElementById('btnClear');
  const $btnPcMic  = document.getElementById('btnPcMic');
  const $btnPcTts  = document.getElementById('btnPcTts');
  const $btnWebcam = document.getElementById('btnWebcam');
  const $fps       = document.getElementById('fps');
  const canvas     = document.getElementById('canvas');
  const ctx        = canvas.getContext('2d');

  // === 获取/创建聊天容器（关键补丁） ===
  let chatContainer = document.getElementById('chatContainer');

  function ensureChatContainer() {
    // 已缓存且仍在文档中
    if (chatContainer && document.body.contains(chatContainer)) return chatContainer;

    // 重新获取，防热更新或 DOM 移动
    chatContainer = document.getElementById('chatContainer');
    if (!chatContainer) {
      chatContainer = document.createElement('div');
      chatContainer.id = 'chatContainer';

      // 优先挂到 finalList 的父容器；否则挂到 partial 的父容器；再否则挂到 body 兜底
      if ($finalList && $finalList.parentElement) {
        // 隐藏原来的 finalList
        $finalList.style.display = 'none';
        // 将聊天容器挂载到 finals div 内
        $finalList.parentElement.appendChild(chatContainer);
        console.log('[chat] 创建并挂载 #chatContainer 到 finalList 区域');
      } else if ($partial && $partial.parentElement) {
        $partial.parentElement.appendChild(chatContainer);
        console.log('[chat] 创建并挂载 #chatContainer 到 partial 区域');
      } else {
        document.body.appendChild(chatContainer);
        console.warn('[chat] 未找到合适锚点，已挂到 <body>');
      }
    }
    return chatContainer;
  }

  // === 注入聊天样式（左右两侧气泡 + 时间戳，增加权重）===
  (function injectChatStyles(){
    if (document.getElementById('chat-style-injected')) return;
    const s = document.createElement('style');
    s.id = 'chat-style-injected';
    s.textContent = `
      #chatContainer{
        position: relative !important;
        overflow-y: auto !important;
        flex: 1 !important;  /* 改为使用 flex: 1 占满剩余空间 */
        min-height: 0 !important;  /* 确保 flex 子元素能正确收缩 */
        padding: 12px 12px 4px !important;
        background: #0b1020 !important;
        border: 1px solid #1d2438 !important;
        border-radius: 10px !important;
        margin-top: 12px !important;
      }
      
      /* 自定义滚动条样式 */
      #chatContainer::-webkit-scrollbar {
        width: 8px !important;
      }
      
      #chatContainer::-webkit-scrollbar-track {
        background: #0d1420 !important;
        border-radius: 4px !important;
      }
      
      #chatContainer::-webkit-scrollbar-thumb {
        background: #2a3446 !important;
        border-radius: 4px !important;
        transition: background 0.2s !important;
      }
      
      #chatContainer::-webkit-scrollbar-thumb:hover {
        background: #3a4556 !important;
      }
      
      /* Firefox 滚动条 */
      #chatContainer {
        scrollbar-width: thin !important;
        scrollbar-color: #2a3446 #0d1420 !important;
      }
      .timestamp{
        text-align:center !important;
        font-size:12px !important;
        color:#8a93a5 !important;
        margin:10px 0 !important;
        user-select:none !important;
      }
      .message{
        display:flex !important;
        gap:8px !important;
        margin:6px 0 !important;
        align-items:flex-end !important;
      }
      .message.ai{ justify-content:flex-start !important; }
      .message.user{ justify-content:flex-end !important; }

      .avatar{
        width:28px !important; height:28px !important; border-radius:50% !important;
        background:#232a3d !important; flex:0 0 28px !important;
        display:flex !important; align-items:center !important; justify-content:center !important;
        color:#9fb0c3 !important; font-size:12px !important; user-select:none !important;
        border:1px solid #29314a !important;
      }
      .message.user .avatar{ display:none !important; }

      .bubble{
        max-width: 72% !important;
        padding:10px 12px !important;
        line-height:1.45 !important;
        border-radius:14px !important;
        word-break:break-word !important;
        white-space:pre-wrap !important;
        border:1px solid transparent !important;
        box-shadow:0 2px 8px rgba(0,0,0,0.15) !important;
        font-size:14px !important;
      }
      .message.ai .bubble{
        background:#111a2e !important;
        color:#e6edf3 !important;
        border-color:#1e2740 !important;
        border-top-left-radius:6px !important;
      }
      .message.user .bubble{
        background:#2a6df4 !important;
        color:#fff !important;
        border-color:#2a6df4 !important;
        border-top-right-radius:6px !important;
      }
    `;
    document.head.appendChild(s);
  })();

  // 聊天消息管理
  let lastTimestamp = 0;
  const TIMESTAMP_INTERVAL = 5 * 60 * 1000; // 5分钟
  
  function shouldShowTimestamp() {
    const now = Date.now();
    if (now - lastTimestamp > TIMESTAMP_INTERVAL) {
      lastTimestamp = now;
      return true;
    }
    return false;
  }
  
  function formatTime(timestamp = Date.now()) {
    const date = new Date(timestamp);
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${hours}:${minutes}`;
  }
  
  function addTimestamp() {
    const container = ensureChatContainer();
    const timestampDiv = document.createElement('div');
    timestampDiv.className = 'timestamp';
    timestampDiv.textContent = formatTime();
    container.appendChild(timestampDiv);
  }
  
  function addMessage(text, isUser = false) {
    // 时间戳
    if (shouldShowTimestamp()) addTimestamp();

    const container = ensureChatContainer();

    // 行容器
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'ai'}`;

    // 左侧头像（AI）
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = isUser ? '' : 'AI';

    // 气泡
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'bubble';
    bubbleDiv.textContent = text;

    if (isUser){
      // 右侧：气泡在右
      messageDiv.appendChild(bubbleDiv);
    }else{
      // 左侧：头像 + 气泡
      messageDiv.appendChild(avatar);
      messageDiv.appendChild(bubbleDiv);
    }

    container.appendChild(messageDiv);

    // 滚动到底部
    container.scrollTop = container.scrollHeight;
  }

  function setBadge(el, ok, text){
    el.textContent = text;
    el.className = 'badge ' + (ok? 'ok' : 'err');
  }

  function navLabelAndText(raw) {
    // 去掉前缀 “[导航] ”
    const t = raw.startsWith('[导航]') ? raw.substring(4).trim() : raw;
    // 粗略判断：含“斑马线/绿灯/红灯/黄灯/过马路”归为斑马线导航，否则盲道导航
    const crossHints = ['斑马线', '绿灯', '红灯', '黄灯', '过马路'];
    const isCross = crossHints.some(k => t.includes(k));
    const label = isCross ? '【斑马线导航】' : '【盲道导航】';
    return { label, text: `${label} ${t}` };
  }

  function fitCanvas(){
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(320, Math.floor(rect.width));
    const h = Math.max(240, Math.floor(rect.width * 3/4)); // 4:3
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
    }
  }
  window.addEventListener('resize', fitCanvas); fitCanvas();

  let wsCam, wsUI, frames = 0, fpsTimer = 0;

  function drawBlob(buf){
    const blob = new Blob([buf], {type:'image/jpeg'});
    if ('createImageBitmap' in window){
      createImageBitmap(blob).then(bmp=>{
        fitCanvas();
        ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
      }).catch(()=>{});
    }else{
      const img = new Image();
      img.onload = ()=>{ fitCanvas(); ctx.drawImage(img,0,0,canvas.width,canvas.height); URL.revokeObjectURL(img.src); };
      img.src = URL.createObjectURL(blob);
    }
    frames++;
    const now = performance.now();
    if (!fpsTimer) fpsTimer = now;
    if (now - fpsTimer >= 1000){
      $fps.textContent = 'FPS: ' + frames;
      frames = 0; fpsTimer = now;
    }
  }

  function connectCamera(){
    try{ if (wsCam) wsCam.close(); }catch(e){}
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    wsCam = new WebSocket(`${proto}://${location.host}/ws/viewer`);
    setBadge($camStatus, false, 'Camera: connecting…');
    wsCam.binaryType = 'arraybuffer';
    wsCam.onopen  = ()=> setBadge($camStatus, true, 'Camera: connected');
    wsCam.onclose = ()=> setBadge($camStatus, false, 'Camera: disconnected');
    wsCam.onerror = ()=> setBadge($camStatus, false, 'Camera: error');
    wsCam.onmessage = (ev)=> drawBlob(ev.data);
  }

  // ===== 麦克风推流：浏览器 → /ws_audio（PCM16, 16kHz, 单声道, 20ms帧）=====
  let micWs = null;
  let micAudioCtx = null;
  let micStream = null;
  let micProcessor = null;
  let pcMicEnabled = false;
  let pcTtsPlaybackEnabled = false;

  function renderPcAudioButtons() {
    if ($btnPcMic) {
      $btnPcMic.textContent = `电脑麦克风: ${pcMicEnabled ? '开' : '关'}`;
      $btnPcMic.classList.toggle('primary', pcMicEnabled);
      $btnPcMic.classList.toggle('ghost', !pcMicEnabled);
    }
    if ($btnPcTts) {
      $btnPcTts.textContent = `电脑TTS播放: ${pcTtsPlaybackEnabled ? '开' : '关'}`;
      $btnPcTts.classList.toggle('primary', pcTtsPlaybackEnabled);
      $btnPcTts.classList.toggle('ghost', !pcTtsPlaybackEnabled);
    }
  }

  async function loadClientConfig() {
    try {
      const r = await fetch('/api/client-config');
      if (!r.ok) return;
      const cfg = await r.json();
      pcMicEnabled = !!cfg.pc_mic_auto_start;
      pcTtsPlaybackEnabled = !!cfg.pc_tts_playback_enabled;
    } catch (e) {
      console.warn('[Config] 加载失败，使用默认值(麦克风关/TTS关):', e);
    }
  }

  async function setServerTtsSynthMode(enableServerTtsSynth) {
    try {
      await fetch('/api/pc-audio-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enableServerTtsSynth })
      });
    } catch (e) {
      console.warn('[Config] 设置服务端TTS模式失败:', e);
    }
  }

  function syncMicState() {
    if (pcMicEnabled) {
      if (wsUI && wsUI.readyState === WebSocket.OPEN) {
        startMic();
      }
    } else {
      stopMic();
    }
  }

  async function startMic() {
    // 防重复连接：如果已有活跃连接，不再创建
    if (micWs && micWs.readyState <= WebSocket.OPEN) {
      console.log('[Mic] 已有活跃连接，跳过重复连接');
      return;
    }
    stopMic(); // 清理残留
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }, video: false });
      // 使用浏览器默认采样率，后面重采样到 16kHz
      micAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const nativeSR = micAudioCtx.sampleRate;
      const targetSR = 16000;
      const ratio = nativeSR / targetSR;
      console.log(`[Mic] 原始采样率: ${nativeSR}, 目标: ${targetSR}, 比率: ${ratio.toFixed(2)}`);

      const source = micAudioCtx.createMediaStreamSource(micStream);
      // bufferSize 要大一些确保够重采样
      const bufSize = 4096;
      micProcessor = micAudioCtx.createScriptProcessor(bufSize, 1, 1);

      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      micWs = new WebSocket(`${proto}//${location.host}/ws_audio`);
      micWs.binaryType = 'arraybuffer';

      micWs.onopen = () => {
        micWs.send('START');
        console.log('[Mic] WebSocket 已连接，发送 START');
        setBadge($asrStatus, true, 'ASR: connected');
      };
      micWs.onmessage = ev => {
        const msg = typeof ev.data === 'string' ? ev.data.trim() : '';
        if (msg === 'RESTART') { micWs.send('START'); }
        else if (msg.startsWith('OK:')) { console.log('[Mic]', msg); }
      };
      micWs.onclose = () => { setBadge($asrStatus, false, 'ASR: disconnected'); console.log('[Mic] WebSocket 已关闭'); };
      micWs.onerror = e => { setBadge($asrStatus, false, 'ASR: error'); console.error('[Mic] 错误:', e); };

      micProcessor.onaudioprocess = e => {
        if (!micWs || micWs.readyState !== WebSocket.OPEN) return;
        const float32 = e.inputBuffer.getChannelData(0);
        // 重采样：从 nativeSR 到 16kHz
        const outLen = Math.floor(float32.length / ratio);
        const pcm16 = new Int16Array(outLen);
        for (let i = 0; i < outLen; i++) {
          const srcIdx = Math.floor(i * ratio);
          pcm16[i] = Math.max(-32768, Math.min(32767, float32[srcIdx] * 32768));
        }
        micWs.send(pcm16.buffer);
      };

      source.connect(micProcessor);
      micProcessor.connect(micAudioCtx.destination);
      console.log('[Mic] 麦克风推流已启动');
    } catch(e) {
      console.error('[Mic] 启动失败:', e);
      setBadge($asrStatus, false, 'ASR: mic error');
    }
  }

  function stopMic() {
    if (micProcessor) { micProcessor.disconnect(); micProcessor = null; }
    if (micAudioCtx) { micAudioCtx.close(); micAudioCtx = null; }
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    if (micWs) { micWs.close(); micWs = null; }
    console.log('[Mic] 麦克风推流已停止');
  }

  function connectASR(){
    // connectASR 现在只负责 /ws_ui 文字接收，麦克风推流由 startMic() 单独处理
    try{ if (wsUI) wsUI.close(); }catch(e){}
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    wsUI = new WebSocket(`${proto}://${location.host}/ws_ui`);
    setBadge($asrStatus, false, 'ASR: connecting…');
    wsUI.onopen  = ()=> { setBadge($asrStatus, true, 'ASR: connected'); syncMicState(); };
    wsUI.onclose = ()=> { setBadge($asrStatus, false, 'ASR: disconnected'); stopMic(); };
    wsUI.onerror = ()=> setBadge($asrStatus, false, 'ASR: error');
    wsUI.onmessage = (ev)=>{
      const s = ev.data || '';
      if (s.startsWith('INIT:')){
        try{
          const data = JSON.parse(s.slice(5));
          $partial.textContent = data.partial || '（等待音频…）';
          
          // 初始化时加载历史消息（识别 [AI] 与 [导航]）
          if (data.finals && data.finals.length > 0) {
            data.finals.forEach(text => {
              if (text.startsWith('[AI]')) {
                addMessage(text.substring(4).trim(), false);
              } else if (text.startsWith('[导航]')) {
                const { text: show } = navLabelAndText(text);
                addMessage(show, false);
              } else {
                addMessage(text, true);
              }
            });
          }
        }catch(e){}
        return;
      }
      if (s.startsWith('PARTIAL:')){ 
        $partial.textContent = s.slice(8); 
        return; 
      }
      if (s.startsWith('FINAL:')){
        const text = s.slice(6);
        if (text.startsWith('[AI]')) {
          addMessage(text.substring(4).trim(), false);
        } else if (text.startsWith('[导航]')) {
          const { text: show } = navLabelAndText(text);
          addMessage(show, false); // 左侧 AI
        } else {
          addMessage(text, true);  // 其它仍按右侧
        }
        $partial.textContent = '（等待音频…）';
        return;
      }
      // 【TTS 音频播放】服务器推送的合成语音
      if (s.startsWith('TTS_AUDIO:')){
        if (!pcTtsPlaybackEnabled) {
          return;
        }
        const rest = s.slice(10); // "mp3:base64..." 或 "wav:base64..."
        const colonIdx = rest.indexOf(':');
        if (colonIdx > 0) {
          const fmt = rest.slice(0, colonIdx);
          const b64 = rest.slice(colonIdx + 1);
          const mime = fmt === 'mp3' ? 'audio/mpeg' : 'audio/wav';
          console.log(`[TTS] 收到音频 fmt=${fmt}, b64长度=${b64.length}`);
          try {
            const raw = atob(b64);
            const ab = new Uint8Array(raw.length);
            for (let i = 0; i < raw.length; i++) ab[i] = raw.charCodeAt(i);
            const blob = new Blob([ab], { type: mime });
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            audio.volume = 1.0;
            const playPromise = audio.play();
            if (playPromise) {
              playPromise.then(() => {
                console.log('[TTS] 播放成功');
              }).catch(e => {
                console.warn('[TTS] 自动播放被阻止，尝试用 AudioContext 解码播放:', e);
                // 回退方案：使用 AudioContext 解码播放
                try {
                  const actx = new (window.AudioContext || window.webkitAudioContext)();
                  fetch(url).then(r => r.arrayBuffer()).then(buf => {
                    actx.decodeAudioData(buf, decoded => {
                      const src = actx.createBufferSource();
                      src.buffer = decoded;
                      src.connect(actx.destination);
                      src.start(0);
                      console.log('[TTS] AudioContext 回退播放成功');
                      src.onended = () => { URL.revokeObjectURL(url); actx.close(); };
                    }, err => {
                      console.error('[TTS] AudioContext 解码失败:', err);
                      URL.revokeObjectURL(url);
                    });
                  });
                } catch(e2) {
                  console.error('[TTS] 回退播放也失败:', e2);
                  URL.revokeObjectURL(url);
                }
              });
            }
            audio.onended = () => URL.revokeObjectURL(url);
          } catch(e) { console.error('[TTS] 解码失败:', e); }
        }
        return;
      }
    }
  }

  $btnClear.onclick = () => {
    const container = ensureChatContainer();
    // 清空聊天记录
    const messages = container.querySelectorAll('.message, .timestamp');
    messages.forEach(msg => msg.remove());
    lastTimestamp = 0; // 重置时间戳计数
  };

  if ($btnPcMic) {
    $btnPcMic.onclick = () => {
      pcMicEnabled = !pcMicEnabled;
      renderPcAudioButtons();
      syncMicState();
    };
  }

  if ($btnPcTts) {
    $btnPcTts.onclick = async () => {
      pcTtsPlaybackEnabled = !pcTtsPlaybackEnabled;
      renderPcAudioButtons();
      await setServerTtsSynthMode(pcTtsPlaybackEnabled);
    };
  }

  // ===== 电脑摄像头：浏览器推帧到服务器 /ws/camera =====
  let webcamActive = false;
  let cameraStream = null;
  let cameraWs = null;
  let cameraFrameTimer = null;
  const cameraCanvas = document.createElement('canvas');
  const cameraCtx = cameraCanvas.getContext('2d');
  const cameraVideo = document.createElement('video');
  cameraVideo.autoplay = true;
  cameraVideo.playsInline = true;
  cameraVideo.muted = true;

  async function startBrowserCamera() {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      cameraVideo.srcObject = cameraStream;
      await cameraVideo.play();

      // 通知服务端初始化导航器
      await fetch('/api/webcam/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });

      // 建立 WebSocket 推帧连接
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      cameraWs = new WebSocket(`${proto}//${location.host}/ws/camera`);
      cameraWs.binaryType = 'arraybuffer';

      cameraWs.onopen = () => {
        console.log('[Camera] WebSocket 推帧已连接');
        // 每 66ms 推一帧（约 15fps，显示与YOLO已分离不会阻塞）
        cameraFrameTimer = setInterval(() => {
          if (cameraWs.readyState !== WebSocket.OPEN) return;
          const vw = cameraVideo.videoWidth, vh = cameraVideo.videoHeight;
          if (!vw || !vh) return;
          cameraCanvas.width = vw;
          cameraCanvas.height = vh;
          cameraCtx.drawImage(cameraVideo, 0, 0, vw, vh);
          cameraCanvas.toBlob(blob => {
            if (blob && cameraWs.readyState === WebSocket.OPEN) {
              blob.arrayBuffer().then(buf => cameraWs.send(buf));
            }
          }, 'image/jpeg', 0.75);
        }, 66);
      };

      cameraWs.onerror = e => console.error('[Camera] WebSocket 错误:', e);
      cameraWs.onclose = () => { console.log('[Camera] WebSocket 已关闭'); clearInterval(cameraFrameTimer); };

      webcamActive = true;
      $btnWebcam.textContent = '停止摄像头';
      $btnWebcam.classList.remove('ghost');
      $btnWebcam.classList.add('primary');
      $camStatus.textContent = 'Camera: 电脑摄像头';
      $camStatus.classList.add('ok');
      document.getElementById('canvasHint').style.display = 'none';
      console.log('[Camera] 浏览器摄像头已启动');
    } catch (e) {
      console.error('[Camera] 启动失败:', e);
      alert('启动摄像头失败: ' + e.message);
    }
  }

  function stopBrowserCamera() {
    clearInterval(cameraFrameTimer);
    if (cameraWs) { cameraWs.close(); cameraWs = null; }
    if (cameraStream) { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }
    fetch('/api/webcam/stop', { method: 'POST' });
    webcamActive = false;
    $btnWebcam.textContent = '电脑摄像头';
    $btnWebcam.classList.remove('primary');
    $btnWebcam.classList.add('ghost');
    $camStatus.textContent = 'Camera: 已停止';
    $camStatus.classList.remove('ok');
    console.log('[Camera] 浏览器摄像头已停止');
  }

  $btnWebcam.onclick = () => webcamActive ? stopBrowserCamera() : startBrowserCamera();

  async function initPage() {
    await loadClientConfig();
    renderPcAudioButtons();
    await setServerTtsSynthMode(pcTtsPlaybackEnabled);
    // 页面加载时：仅连接画面订阅和 ASR，不自动开启本机摄像头
    connectCamera();
    connectASR();
  }

  initPage();
})();
