// static/mobile.js
// 手机端摄像头帧采集 + 麦克风 PCM 采集
// 通过 WebSocket 发送到后端，复用 ESP32 的处理管线

(() => {
  'use strict';

  // ====== 配置 ======
  const CAMERA_FPS     = 15;        // 帧率（手机端不需要太高，省电省带宽）
  const JPEG_QUALITY   = 0.6;       // JPEG 压缩质量 0-1
  const AUDIO_SAMPLE_RATE = 16000;  // ASR 需要 16kHz
  const AUDIO_CHUNK_MS = 20;        // 每 20ms 发送一包 PCM
  const FACING_MODE_KEY = 'mobile_facing_mode';

  // ====== DOM 元素 ======
  const $mobileToggle  = document.getElementById('mobileToggle');
  const $mobilePanel   = document.getElementById('mobilePanel');
  const $btnStartCam   = document.getElementById('btnStartCam');
  const $btnStopCam    = document.getElementById('btnStopCam');
  const $btnFlipCam    = document.getElementById('btnFlipCam');
  const $btnStartMic   = document.getElementById('btnStartMic');
  const $btnStopMic    = document.getElementById('btnStopMic');
  const $mobileCamStatus = document.getElementById('mobileCamStatus');
  const $mobileMicStatus = document.getElementById('mobileMicStatus');

  if (!$mobileToggle) return; // 非手机端页面不执行

  // ====== 状态 ======
  let videoStream = null;
  let videoEl     = null;
  let captureCanvas = null;
  let captureCtx    = null;
  let captureTimer  = null;
  let wsCam = null;

  let audioStream   = null;
  let audioCtx      = null;
  let audioWorklet  = null;
  let wsAudio       = null;
  let audioStarted  = false;

  let facingMode = localStorage.getItem(FACING_MODE_KEY) || 'environment'; // 默认后置

  // ====== 工具函数 ======
  const proto = () => location.protocol === 'https:' ? 'wss' : 'ws';

  function setStatus(el, ok, text) {
    if (!el) return;
    el.textContent = text;
    el.className = 'badge ' + (ok ? 'ok' : 'err');
  }

  // ====== 摄像头 ======
  async function startCamera() {
    try {
      // 先停止旧的
      stopCamera();

      setStatus($mobileCamStatus, false, '摄像头: 请求权限…');

      const constraints = {
        video: {
          facingMode: facingMode,
          width:  { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false
      };

      videoStream = await navigator.mediaDevices.getUserMedia(constraints);

      // 创建隐藏的 video 元素
      videoEl = document.createElement('video');
      videoEl.srcObject = videoStream;
      videoEl.setAttribute('playsinline', ''); // iOS 必须
      videoEl.setAttribute('autoplay', '');
      videoEl.muted = true;
      await videoEl.play();

      // 等待视频实际渲染
      await new Promise(resolve => {
        if (videoEl.videoWidth > 0) return resolve();
        videoEl.onloadedmetadata = resolve;
      });

      // 创建截帧用的 canvas
      captureCanvas = document.createElement('canvas');
      captureCanvas.width  = videoEl.videoWidth  || 640;
      captureCanvas.height = videoEl.videoHeight || 480;
      captureCtx = captureCanvas.getContext('2d');

      // 连接 WebSocket
      connectCameraWS();

      setStatus($mobileCamStatus, true, `摄像头: ${captureCanvas.width}x${captureCanvas.height}`);
      $btnStartCam.disabled = true;
      $btnStopCam.disabled  = false;
      $btnFlipCam.disabled  = false;

      console.log(`[mobile] 摄像头已启动: ${captureCanvas.width}x${captureCanvas.height}, facing=${facingMode}`);
    } catch (e) {
      console.error('[mobile] 摄像头启动失败:', e);
      setStatus($mobileCamStatus, false, '摄像头: 权限被拒绝');
    }
  }

  function stopCamera() {
    if (captureTimer) { clearInterval(captureTimer); captureTimer = null; }
    if (wsCam) { try { wsCam.close(); } catch(e){} wsCam = null; }
    if (videoStream) {
      videoStream.getTracks().forEach(t => t.stop());
      videoStream = null;
    }
    if (videoEl) { videoEl.srcObject = null; videoEl = null; }
    setStatus($mobileCamStatus, false, '摄像头: 已停止');
    $btnStartCam.disabled = false;
    $btnStopCam.disabled  = true;
    $btnFlipCam.disabled  = true;
  }

  async function flipCamera() {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    localStorage.setItem(FACING_MODE_KEY, facingMode);
    if (videoStream) {
      await startCamera(); // 重新启动
    }
  }

  function connectCameraWS() {
    if (wsCam) { try { wsCam.close(); } catch(e){} }
    wsCam = new WebSocket(`${proto()}://${location.host}/ws/camera`);
    wsCam.binaryType = 'arraybuffer';

    wsCam.onopen = () => {
      console.log('[mobile] camera WS connected');
      // 开始定时截帧
      if (captureTimer) clearInterval(captureTimer);
      captureTimer = setInterval(captureAndSend, 1000 / CAMERA_FPS);
    };

    wsCam.onclose = () => {
      console.log('[mobile] camera WS closed');
      if (captureTimer) { clearInterval(captureTimer); captureTimer = null; }
      // 自动重连（如果摄像头仍在运行）
      if (videoStream) {
        setTimeout(connectCameraWS, 2000);
      }
    };

    wsCam.onerror = (e) => {
      console.error('[mobile] camera WS error:', e);
    };
  }

  function captureAndSend() {
    if (!wsCam || wsCam.readyState !== WebSocket.OPEN) return;
    if (!videoEl || videoEl.videoWidth === 0) return;

    // 绘制当前帧
    captureCtx.drawImage(videoEl, 0, 0, captureCanvas.width, captureCanvas.height);

    // 转 JPEG blob 并发送
    captureCanvas.toBlob(blob => {
      if (blob && wsCam && wsCam.readyState === WebSocket.OPEN) {
        blob.arrayBuffer().then(buf => {
          wsCam.send(buf);
        });
      }
    }, 'image/jpeg', JPEG_QUALITY);
  }

  // ====== 麦克风 ======
  async function startMicrophone() {
    try {
      stopMicrophone();
      setStatus($mobileMicStatus, false, '麦克风: 请求权限…');

      audioStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: AUDIO_SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });

      // 连接 audio WebSocket
      wsAudio = new WebSocket(`${proto()}://${location.host}/ws_audio`);
      wsAudio.binaryType = 'arraybuffer';

      wsAudio.onopen = () => {
        console.log('[mobile] audio WS connected, sending START');
        wsAudio.send('START');
      };

      wsAudio.onmessage = (ev) => {
        const msg = typeof ev.data === 'string' ? ev.data : '';
        if (msg === 'OK:STARTED') {
          console.log('[mobile] ASR started');
          setupAudioPipeline();
        } else if (msg === 'RESTART') {
          // ASR 要求重启
          console.log('[mobile] ASR restart requested');
          restartMicrophone();
        }
      };

      wsAudio.onclose = () => {
        console.log('[mobile] audio WS closed');
        if (audioStarted) {
          setTimeout(restartMicrophone, 2000);
        }
      };

      wsAudio.onerror = (e) => {
        console.error('[mobile] audio WS error:', e);
      };

      audioStarted = true;
      setStatus($mobileMicStatus, true, '麦克风: 已连接');
      $btnStartMic.disabled = true;
      $btnStopMic.disabled  = false;

    } catch (e) {
      console.error('[mobile] 麦克风启动失败:', e);
      setStatus($mobileMicStatus, false, '麦克风: 权限被拒绝');
    }
  }

  function setupAudioPipeline() {
    if (!audioStream || !wsAudio) return;

    audioCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: AUDIO_SAMPLE_RATE
    });

    const source = audioCtx.createMediaStreamSource(audioStream);

    // 使用 ScriptProcessorNode（兼容性好）
    // bufferSize = sampleRate * chunkMs / 1000 = 16000 * 0.02 = 320
    const bufSize = Math.max(256, Math.round(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_MS / 1000));
    // ScriptProcessor 要求 bufferSize 是 2 的幂
    const powerOf2 = Math.pow(2, Math.ceil(Math.log2(bufSize)));
    const processor = audioCtx.createScriptProcessor(powerOf2, 1, 1);

    processor.onaudioprocess = (event) => {
      if (!wsAudio || wsAudio.readyState !== WebSocket.OPEN) return;

      const floatData = event.inputBuffer.getChannelData(0);

      // 重采样到精确的 16kHz（如果 AudioContext 的采样率不同）
      let pcm16;
      if (audioCtx.sampleRate !== AUDIO_SAMPLE_RATE) {
        // 简单线性重采样
        const ratio = audioCtx.sampleRate / AUDIO_SAMPLE_RATE;
        const newLen = Math.round(floatData.length / ratio);
        pcm16 = new Int16Array(newLen);
        for (let i = 0; i < newLen; i++) {
          const srcIdx = i * ratio;
          const lo = Math.floor(srcIdx);
          const hi = Math.min(lo + 1, floatData.length - 1);
          const frac = srcIdx - lo;
          const sample = floatData[lo] * (1 - frac) + floatData[hi] * frac;
          pcm16[i] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
        }
      } else {
        pcm16 = new Int16Array(floatData.length);
        for (let i = 0; i < floatData.length; i++) {
          pcm16[i] = Math.max(-32768, Math.min(32767, Math.round(floatData[i] * 32767)));
        }
      }

      // 发送 PCM16 二进制
      wsAudio.send(pcm16.buffer);
    };

    source.connect(processor);
    processor.connect(audioCtx.destination); // 必须连接到 destination 才能触发

    audioWorklet = { source, processor };
    console.log(`[mobile] 音频管线已建立: sampleRate=${audioCtx.sampleRate}, bufSize=${powerOf2}`);
  }

  function stopMicrophone() {
    audioStarted = false;
    if (audioWorklet) {
      try { audioWorklet.processor.disconnect(); } catch(e){}
      try { audioWorklet.source.disconnect(); } catch(e){}
      audioWorklet = null;
    }
    if (audioCtx) {
      try { audioCtx.close(); } catch(e){}
      audioCtx = null;
    }
    if (wsAudio) {
      try { wsAudio.send('STOP'); } catch(e){}
      try { wsAudio.close(); } catch(e){}
      wsAudio = null;
    }
    if (audioStream) {
      audioStream.getTracks().forEach(t => t.stop());
      audioStream = null;
    }
    setStatus($mobileMicStatus, false, '麦克风: 已停止');
    $btnStartMic.disabled = false;
    $btnStopMic.disabled  = true;
  }

  async function restartMicrophone() {
    stopMicrophone();
    await new Promise(r => setTimeout(r, 500));
    await startMicrophone();
  }

  // ====== 音频播放（拉取 /stream.wav）======
  let streamAudio = null;

  function startAudioPlayback() {
    if (streamAudio) return;
    // 用 <audio> 元素拉取后端的流式 WAV，实现语音播报
    streamAudio = new Audio(`${location.protocol}//${location.host}/stream.wav`);
    streamAudio.autoplay = true;
    // iOS 需要用户交互才能播放，会在按钮点击时触发
    streamAudio.play().catch(() => {
      console.log('[mobile] 音频自动播放被阻止，将在下次用户交互时重试');
    });
  }

  function stopAudioPlayback() {
    if (streamAudio) {
      streamAudio.pause();
      streamAudio.src = '';
      streamAudio = null;
    }
  }

  // ====== UI 交互 ======
  $mobileToggle.addEventListener('click', () => {
    const visible = $mobilePanel.style.display !== 'none';
    $mobilePanel.style.display = visible ? 'none' : 'flex';
    $mobileToggle.textContent = visible ? '📱 手机模式' : '📱 关闭手机模式';
  });

  $btnStartCam.addEventListener('click', () => {
    startCamera();
    startAudioPlayback(); // 摄像头启动时同时启动音频播放（需要用户交互触发）
  });
  $btnStopCam.addEventListener('click', stopCamera);
  $btnFlipCam.addEventListener('click', flipCamera);
  $btnStartMic.addEventListener('click', () => {
    startMicrophone();
    startAudioPlayback();
  });
  $btnStopMic.addEventListener('click', stopMicrophone);

  // 初始状态
  $btnStopCam.disabled = true;
  $btnFlipCam.disabled = true;
  $btnStopMic.disabled = true;

  // 检测是否移动设备，自动显示提示
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  if (isMobile) {
    $mobileToggle.style.display = 'inline-block';
    console.log('[mobile] 检测到移动设备');
  } else {
    // PC 端也显示按钮，但不自动展开
    $mobileToggle.style.display = 'inline-block';
  }
})();
