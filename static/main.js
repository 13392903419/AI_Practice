// static/main.js

// ================= 摄像头 + ASR =================
(() => {
  const $camStatus = document.getElementById('camStatus');
  const $asrStatus = document.getElementById('asrStatus');
  const $partial   = document.getElementById('partial');
  const $finalList = document.getElementById('finalList');
  const $btnClear  = document.getElementById('btnClear');
  const $btnRe     = document.getElementById('btnReconnect');
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

  function connectASR(){
    try{ if (wsUI) wsUI.close(); }catch(e){}
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    wsUI = new WebSocket(`${proto}://${location.host}/ws_ui`);
    setBadge($asrStatus, false, 'ASR: connecting…');
    wsUI.onopen  = ()=> setBadge($asrStatus, true, 'ASR: connected');
    wsUI.onclose = ()=> setBadge($asrStatus, false, 'ASR: disconnected');
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
    }
  }

  $btnClear.onclick = ()=> { 
    const container = ensureChatContainer();
    // 清空聊天记录
    const messages = container.querySelectorAll('.message, .timestamp');
    messages.forEach(msg => msg.remove());
    lastTimestamp = 0; // 重置时间戳计数
  };
  $btnRe.onclick    = ()=> { connectCamera(); connectASR(); };

  connectCamera();
  connectASR();
})();
