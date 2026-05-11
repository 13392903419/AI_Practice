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
  const $btnLocate = document.getElementById('btnLocate');
  const $destinationInput = document.getElementById('destinationInput');
  const $btnSearchRoute = document.getElementById('btnSearchRoute');
  const $currentLocationText = document.getElementById('currentLocationText');
  const $destinationText = document.getElementById('destinationText');
  const $routeStatusText = document.getElementById('routeStatusText');
  const $routePreview = document.getElementById('routePreview');
  const $routeDistanceText = document.getElementById('routeDistanceText');
  const $routeEtaText = document.getElementById('routeEtaText');
  const canvas     = document.getElementById('canvas');
  const ctx        = canvas.getContext('2d');
  const clientConfig = {
    amapJsApiKey: '',
    amapJsApiKeyConfigured: false,
    amapSecurityJsCode: ''
  };
  let amapSdkPromise = null;
  let amapMap = null;
  let amapRouteLine = null;
  let amapTrackLine = null;
  let amapCurrentMarker = null;
  let amapStartMarker = null;
  let amapEndMarker = null;
  let amapRenderSeq = 0;

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

  function formatDistance(meters) {
    const value = Number(meters || 0);
    if (!value) return '--';
    return value >= 1000 ? `${(value / 1000).toFixed(1)} 公里` : `${Math.round(value)} 米`;
  }

  function formatDuration(seconds) {
    const value = Number(seconds || 0);
    if (!value) return '--';
    const minutes = Math.max(1, Math.round(value / 60));
    return `${minutes} 分钟`;
  }

  function validPoint(point) {
    return point && Number.isFinite(Number(point.lon)) && Number.isFinite(Number(point.lat));
  }

  function toAmapLngLat(point) {
    return [Number(point.lon), Number(point.lat)];
  }

  function getValidPath(points) {
    return Array.isArray(points) ? points.filter(validPoint).map(toAmapLngLat) : [];
  }

  function resetRoutePreviewClass() {
    if ($routePreview) $routePreview.classList.remove('amap-ready');
  }

  function showMapNotice(message) {
    if ($routeStatusText) $routeStatusText.textContent = message;
    if ($routePreview && !amapMap) {
      resetRoutePreviewClass();
      $routePreview.textContent = message;
    }
  }

  function loadAmapSdk() {
    if (window.AMap) return Promise.resolve(window.AMap);
    if (amapSdkPromise) return amapSdkPromise;
    if (!clientConfig.amapJsApiKey) {
      return Promise.reject(new Error('缺少 AMAP_JS_API_KEY，请在 .env 配置高德 Web端(JS API) Key'));
    }
    if (clientConfig.amapSecurityJsCode) {
      window._AMapSecurityConfig = { securityJsCode: clientConfig.amapSecurityJsCode };
    }
    amapSdkPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = `https://webapi.amap.com/maps?v=2.0&key=${encodeURIComponent(clientConfig.amapJsApiKey)}&plugin=AMap.Scale,AMap.ToolBar`;
      script.async = true;
      script.onload = () => window.AMap ? resolve(window.AMap) : reject(new Error('高德地图 SDK 未正确加载'));
      script.onerror = () => reject(new Error('高德地图 SDK 加载失败'));
      document.head.appendChild(script);
    });
    return amapSdkPromise;
  }

  async function ensureAmapMap() {
    const AMap = await loadAmapSdk();
    if (amapMap) return { AMap, map: amapMap };
    if (!$routePreview) throw new Error('地图容器不存在');
    $routePreview.textContent = '';
    $routePreview.classList.add('amap-ready');
    amapMap = new AMap.Map($routePreview, {
      viewMode: '2D',
      resizeEnable: true,
      zoom: 16,
      mapStyle: 'amap://styles/normal',
      showLabel: true,
      layers: [new AMap.TileLayer({ visible: true, zIndex: 1, opacity: 1 })],
      features: ['bg', 'road', 'building', 'point']
    });
    amapMap.setFeatures(['bg', 'road', 'building', 'point']);
    amapMap.addControl(new AMap.Scale());
    amapMap.addControl(new AMap.ToolBar({ position: { right: '10px', top: '10px' } }));
    return { AMap, map: amapMap };
  }

  function createAmapMarker(AMap, point, color, label) {
    return new AMap.Marker({
      position: toAmapLngLat(point),
      anchor: 'center',
      content: `<div class="amap-pin" style="width:14px;height:14px;border-radius:50%;background:${color};border:3px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.28);"></div>`,
      title: label
    });
  }

  function updateAmapMarkers(AMap, map, points, position) {
    const startPoint = points.find(validPoint);
    const endPoint = [...points].reverse().find(validPoint);
    const currentPoint = validPoint(position) ? position : null;

    if (amapStartMarker) map.remove(amapStartMarker);
    if (amapEndMarker) map.remove(amapEndMarker);
    if (amapCurrentMarker) map.remove(amapCurrentMarker);

    amapStartMarker = startPoint ? createAmapMarker(AMap, startPoint, '#4A7C59', '起点') : null;
    amapEndMarker = endPoint ? createAmapMarker(AMap, endPoint, '#C5453A', '目的地') : null;
    amapCurrentMarker = currentPoint ? createAmapMarker(AMap, currentPoint, '#2563eb', '当前位置') : null;

    [amapStartMarker, amapEndMarker, amapCurrentMarker].filter(Boolean).forEach(marker => map.add(marker));
  }

  async function renderAmapPreview(points, position, history, fallback) {
    const seq = ++amapRenderSeq;
    if ($routePreview && !amapMap) {
      resetRoutePreviewClass();
      $routePreview.textContent = '正在加载高德地图...';
    }
    try {
      const { AMap, map } = await ensureAmapMap();
      if (seq !== amapRenderSeq) return;
      const routePath = getValidPath(points);
      const hasRoute = routePath.length >= 2;
      const trackPath = hasRoute ? getValidPath(history) : [];
      if (amapRouteLine) map.remove(amapRouteLine);
      if (amapTrackLine) map.remove(amapTrackLine);

      amapRouteLine = hasRoute ? new AMap.Polyline({
        path: routePath,
        strokeColor: '#e85d04',
        strokeWeight: 7,
        strokeOpacity: 0.95,
        lineJoin: 'round',
        lineCap: 'round',
        showDir: true
      }) : null;
      amapTrackLine = trackPath.length >= 2 ? new AMap.Polyline({
        path: trackPath,
        strokeColor: '#2563eb',
        strokeWeight: 4,
        strokeOpacity: 0.7,
        strokeStyle: 'dashed',
        lineJoin: 'round',
        lineCap: 'round'
      }) : null;

      [amapRouteLine, amapTrackLine].filter(Boolean).forEach(line => map.add(line));
      updateAmapMarkers(AMap, map, points, position);

      const fitItems = [amapRouteLine, amapTrackLine, amapStartMarker, amapEndMarker, amapCurrentMarker].filter(Boolean);
      if (hasRoute && fitItems.length) map.setFitView(fitItems, false, [36, 36, 36, 36]);
      else if (validPoint(position)) map.setZoomAndCenter(17, toAmapLngLat(position));
    } catch (e) {
      console.warn('[Navigation] 高德地图不可用:', e);
      amapMap = null;
      showMapNotice(e.message || '高德地图加载失败');
    }
  }

  function buildProjector(points, width, height, padding) {
    const validPoints = points.filter(validPoint);
    const lons = validPoints.map(p => Number(p.lon));
    const lats = validPoints.map(p => Number(p.lat));
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const lonSpan = Math.max(maxLon - minLon, 0.00001);
    const latSpan = Math.max(maxLat - minLat, 0.00001);
    const scale = Math.min((width - padding * 2) / lonSpan, (height - padding * 2) / latSpan);
    const routeWidth = lonSpan * scale;
    const routeHeight = latSpan * scale;
    const offsetX = (width - routeWidth) / 2;
    const offsetY = (height - routeHeight) / 2;
    return point => {
      const x = offsetX + (Number(point.lon) - minLon) * scale;
      const y = height - offsetY - (Number(point.lat) - minLat) * scale;
      return { x, y, text: `${x.toFixed(1)},${y.toFixed(1)}` };
    };
  }

  function pathFromPoints(points, project) {
    return points.filter(validPoint).map((point, index) => {
      return `${index === 0 ? 'M' : 'L'} ${project(point).text}`;
    }).join(' ');
  }

  function drawRoutePreview(points, position, history) {
    if (!$routePreview) return;
    if (!points || points.length < 2) {
      resetRoutePreviewClass();
      $routePreview.textContent = '路线信息将在定位和规划完成后显示';
      return;
    }

    if (clientConfig.amapJsApiKey) {
      renderAmapPreview(points, position, history, () => drawSvgRoutePreview(points, position, history));
      return;
    }

    if (!clientConfig.amapJsApiKeyConfigured) {
      showMapNotice('缺少 AMAP_JS_API_KEY，请配置高德 Web端(JS API) Key 后重启后端');
      return;
    }

    drawSvgRoutePreview(points, position, history);
  }

  function drawSvgRoutePreview(points, position, history) {
    resetRoutePreviewClass();

    const width = Math.max(320, $routePreview.clientWidth || 520);
    const height = Math.max(220, $routePreview.clientHeight || 260);
    const padding = 24;
    const historyPoints = Array.isArray(history) ? history.filter(validPoint) : [];
    const projectorPoints = [...points, ...historyPoints, position].filter(validPoint);
    const project = buildProjector(projectorPoints, width, height, padding);
    const path = pathFromPoints(points, project);
    const trackPath = pathFromPoints(historyPoints, project);
    const start = project(points[0]);
    const end = project(points[points.length - 1]);
    const current = validPoint(position) ? project(position) : null;
    const currentMarker = current ? `
        <circle cx="${current.x.toFixed(1)}" cy="${current.y.toFixed(1)}" r="16" fill="#2563eb" opacity="0.16"></circle>
        <circle cx="${current.x.toFixed(1)}" cy="${current.y.toFixed(1)}" r="7" fill="#2563eb" stroke="#f5f1e8" stroke-width="3"></circle>
        <text x="${current.x.toFixed(1)}" y="${(current.y - 18).toFixed(1)}" text-anchor="middle" fill="#1f3f8b" font-size="12">当前位置</text>` : '';
    const trackLine = trackPath ? `<path d="${trackPath}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="2 8" opacity="0.72"></path>` : '';

    $routePreview.innerHTML = `
      <svg class="route-map-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="导航路线预览">
        <rect width="${width}" height="${height}" fill="#f5f1e8"></rect>
        <path d="${path}" fill="none" stroke="#d4cfc5" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"></path>
        <path d="${path}" fill="none" stroke="#e85d04" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"></path>
        ${trackLine}
        <circle cx="${start.x.toFixed(1)}" cy="${start.y.toFixed(1)}" r="8" fill="#4A7C59"></circle>
        <circle cx="${end.x.toFixed(1)}" cy="${end.y.toFixed(1)}" r="8" fill="#C5453A"></circle>
        ${currentMarker}
      </svg>
    `;
  }

  function drawLocationPreview(position, history) {
    if (!$routePreview) return;
    if (!position) {
      resetRoutePreviewClass();
      $routePreview.textContent = '路线信息将在定位和规划完成后显示';
      return;
    }

    if (clientConfig.amapJsApiKey) {
      renderAmapPreview([], position, [], () => drawSvgLocationPreview(position, []));
      return;
    }

    if (!clientConfig.amapJsApiKeyConfigured) {
      showMapNotice('缺少 AMAP_JS_API_KEY，请配置高德 Web端(JS API) Key 后重启后端');
      return;
    }

    drawSvgLocationPreview(position, []);
  }

  function drawSvgLocationPreview(position, history) {
    resetRoutePreviewClass();

    const width = Math.max(320, $routePreview.clientWidth || 520);
    const height = Math.max(220, $routePreview.clientHeight || 260);
    const padding = 32;
    const historyPoints = Array.isArray(history) ? history.filter(validPoint) : [];
    const projectorPoints = [...historyPoints, position].filter(validPoint);
    const project = buildProjector(projectorPoints, width, height, padding);
    const current = project(position);
    const trackPath = pathFromPoints(historyPoints, project);
    const trackLine = trackPath ? `<path d="${trackPath}" fill="none" stroke="#2563eb" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity="0.72"></path>` : '';
    $routePreview.innerHTML = `
      <svg class="route-map-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="当前位置预览">
        <rect width="${width}" height="${height}" fill="#f5f1e8"></rect>
        <path d="M 0 ${height / 2} H ${width} M ${width / 2} 0 V ${height}" stroke="#d4cfc5" stroke-width="1" stroke-dasharray="6 8"></path>
        ${trackLine}
        <circle cx="${current.x.toFixed(1)}" cy="${current.y.toFixed(1)}" r="18" fill="#2563eb" opacity="0.16"></circle>
        <circle cx="${current.x.toFixed(1)}" cy="${current.y.toFixed(1)}" r="8" fill="#2563eb" stroke="#f5f1e8" stroke-width="3"></circle>
        <text x="${current.x.toFixed(1)}" y="${(current.y + 34).toFixed(1)}" text-anchor="middle" fill="#6B6560" font-size="13">当前位置</text>
      </svg>
    `;
  }

  function renderNavigationStatus(status) {
    if (!status) return;
    const position = status.position;
    const history = status.position_history || [];
    const plan = status.plan;
    const localState = status.local_navigation_state;
    if ($currentLocationText) {
      $currentLocationText.textContent = position
        ? `${position.lat.toFixed(6)}, ${position.lon.toFixed(6)}${position.provider ? ` · ${position.provider}` : ''}${Number.isFinite(position.accuracy) ? ` · ±${Math.round(position.accuracy)}m` : ''}${Number.isFinite(position.age_s) ? ` · ${position.age_s}s前` : ''}`
        : '等待定位...';
    }
    if ($destinationText) {
      $destinationText.textContent = status.destination_text || '尚未设置';
    }
    if ($routeStatusText) {
      const stateParts = [];
      if (status.active) stateParts.push('目的地导航中');
      else if (plan) stateParts.push('路线已规划');
      if (localState === 'BLINDPATH_NAV') stateParts.push('盲道识别中');
      else if (localState && !['CHAT', 'IDLE'].includes(localState)) stateParts.push(localState);
      $routeStatusText.textContent = status.error || (stateParts.length ? stateParts.join(' · ') : '等待导航指令');
    }
    if ($routeDistanceText) {
      $routeDistanceText.textContent = `距离: ${plan ? formatDistance(plan.total_distance_m) : '--'}`;
    }
    if ($routeEtaText) {
      $routeEtaText.textContent = `预计: ${plan ? formatDuration(plan.total_duration_s) : '--'}`;
    }
    if (plan) {
      drawRoutePreview(plan.route_points, position, history);
    } else {
      drawLocationPreview(position, history);
    }
  }

  let navStatusLoading = false;

  async function fetchNavigationStatus() {
    if (navStatusLoading) return;
    navStatusLoading = true;
    try {
      const response = await fetch('/api/navigation/status', { cache: 'no-store' });
      if (!response.ok) return;
      renderNavigationStatus(await response.json());
    } catch (e) {
      console.warn('[Navigation] 状态获取失败:', e);
    } finally {
      navStatusLoading = false;
    }
  }

  async function submitLocation(position) {
    const coords = position.coords;
    const payload = {
      lon: coords.longitude,
      lat: coords.latitude,
      accuracy: coords.accuracy,
      provider: 'browser'
    };
    const response = await fetch('/api/location', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error(`定位上报失败: ${response.status}`);
    const data = await response.json();
    renderNavigationStatus(data.status || data);
  }

  function locateOnce() {
    if (!navigator.geolocation) {
      if ($routeStatusText) $routeStatusText.textContent = '当前浏览器不支持定位';
      return;
    }
    if ($routeStatusText) $routeStatusText.textContent = '正在获取当前位置...';
    navigator.geolocation.getCurrentPosition(
      position => submitLocation(position).catch(e => {
        console.error('[Navigation] 定位上报失败:', e);
        if ($routeStatusText) $routeStatusText.textContent = e.message;
      }),
      error => {
        console.warn('[Navigation] 浏览器定位失败:', error);
        if ($routeStatusText) $routeStatusText.textContent = `定位失败: ${error.message}`;
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 5000 }
    );
  }

  async function startRouteSearch() {
    const destination = ($destinationInput && $destinationInput.value || '').trim();
    if (!destination) {
      if ($routeStatusText) $routeStatusText.textContent = '请输入目的地';
      return;
    }
    if ($destinationText) $destinationText.textContent = destination;
    if ($routeStatusText) $routeStatusText.textContent = '正在规划路线...';
    try {
      const response = await fetch('/api/navigation/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ destination })
      });
      const data = await response.json();
      renderNavigationStatus(data.status || data);
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || '路线规划失败');
      }
    } catch (e) {
      console.error('[Navigation] 路线规划失败:', e);
      if ($routeStatusText) $routeStatusText.textContent = e.message;
    }
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
      clientConfig.amapJsApiKey = cfg.amap_js_api_key || '';
      clientConfig.amapJsApiKeyConfigured = !!cfg.amap_js_api_key_configured;
      clientConfig.amapSecurityJsCode = cfg.amap_security_js_code || '';
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
      if (s.startsWith('NAV_STATUS:')){
        try {
          renderNavigationStatus(JSON.parse(s.slice(11)));
        } catch(e) {
          console.warn('[Navigation] 状态推送解析失败:', e);
        }
        return;
      }
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

  if ($btnLocate) {
    $btnLocate.onclick = locateOnce;
  }

  if ($btnSearchRoute) {
    $btnSearchRoute.onclick = startRouteSearch;
  }

  if ($destinationInput) {
    $destinationInput.addEventListener('keydown', event => {
      if (event.key === 'Enter') startRouteSearch();
    });
  }

  // ===== 电脑摄像头：浏览器推帧到服务器 /ws/camera =====
  let webcamActive = false;
  let cameraStream = null;
  let cameraWs = null;
  let cameraFrameTimer = null;
  const CAMERA_PUSH_INTERVAL_MS = 40; // 25fps
  const CAMERA_JPEG_QUALITY = 0.70;   // 略降码率，减轻传输与解码负担
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
        // 每 50ms 推一帧（约 20fps）
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
          }, 'image/jpeg', CAMERA_JPEG_QUALITY);
        }, CAMERA_PUSH_INTERVAL_MS);
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
    fetchNavigationStatus();
    setInterval(fetchNavigationStatus, 1000);
  }

  initPage();
})();
