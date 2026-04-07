package com.example.huixing0406;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.widget.Button;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import java.util.Locale;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    private static final long FINAL_FALLBACK_DELAY_MS = 1200L;

    private TextView tvStatus;
    private TextView tvPartial;
    private TextView tvFinal;
    private PreviewView previewView;
    private Button btnToggle;

    private UiWebSocketClient uiClient;
    private AudioWebSocketClient audioClient;
    private CameraWebSocketClient cameraClient;
    private CameraPipeline cameraPipeline;
    private TextToSpeech tts;
    private boolean streaming;

    private android.media.MediaPlayer ttsAudioPlayer;
    private String lastFinalText = "";

    private final android.os.Handler ttsFallbackHandler = new android.os.Handler(android.os.Looper.getMainLooper());
    private Runnable pendingFinalTtsFallback = null;
    private long ttsFallbackSeq = 0L;
    private long lastServerTtsArrivedAtMs = 0L;
    private String lastTtsMsgSignature = "";
    private long lastTtsMsgAtMs = 0L;

    private final ActivityResultLauncher<String[]> permissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), this::onPermissionResult);

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvStatus = findViewById(R.id.tvStatus);
        tvPartial = findViewById(R.id.tvPartial);
        tvFinal = findViewById(R.id.tvFinal);
        previewView = findViewById(R.id.previewView);
        btnToggle = findViewById(R.id.btnToggle);

        tts = new android.speech.tts.TextToSpeech(this, status -> {
            if (status != android.speech.tts.TextToSpeech.SUCCESS) {
                android.util.Log.e("MainActivity", "TTS init failed, status=" + status);
                runOnUiThread(() -> tvStatus.setText("TTS init failed"));
                return;
            }

            int langResult = tts.setLanguage(java.util.Locale.CHINESE);
            boolean langOk = langResult != android.speech.tts.TextToSpeech.LANG_MISSING_DATA
                    && langResult != android.speech.tts.TextToSpeech.LANG_NOT_SUPPORTED;

            if (!langOk) {
                android.util.Log.e("MainActivity", "Chinese TTS not supported or missing voice data");
                runOnUiThread(() -> tvStatus.setText("TTS zh-CN unsupported"));
            } else {
                tts.setSpeechRate(1.0f);
                tts.setPitch(1.0f);
                android.util.Log.i("MainActivity", "TTS ready");
            }
        });

        btnToggle.setOnClickListener(v -> {
            if (streaming) {
                stopStreaming();
            } else {
                ensurePermissionsAndStart();
            }
        });
    }

    private void ensurePermissionsAndStart() {
        boolean camGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
        boolean micGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
        if (camGranted && micGranted) {
            startStreaming();
        } else {
            permissionLauncher.launch(new String[]{Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO});
        }
    }

    private void onPermissionResult(Map<String, Boolean> result) {
        boolean camGranted = Boolean.TRUE.equals(result.get(Manifest.permission.CAMERA));
        boolean micGranted = Boolean.TRUE.equals(result.get(Manifest.permission.RECORD_AUDIO));
        if (camGranted && micGranted) {
            startStreaming();
        } else {
            tvStatus.setText("Permission denied");
        }
    }

    private void startStreaming() {
        uiClient = new UiWebSocketClient(new UiWebSocketClient.Listener() {
            @Override
            public void onOpen() {
                runOnUiThread(() -> tvStatus.setText("ws_ui connected"));
            }

            @Override
            public void onClosed() {
                runOnUiThread(() -> tvStatus.setText("ws_ui closed"));
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> tvStatus.setText("ws_ui error: " + error));
            }

            @Override
            public void onPartial(String text) {
                runOnUiThread(() -> tvPartial.setText(text));
            }

            @Override
            public void onFinal(String text) {
                runOnUiThread(() -> tvFinal.setText(text));
                lastFinalText = normalizeSpeechText(text);
                if (lastFinalText.isEmpty()) return;

                // 每次 FINAL 都生成新的序号，旧兜底自动失效
                final long seq = ++ttsFallbackSeq;

                // 取消之前未触发的兜底任务
                if (pendingFinalTtsFallback != null) {
                    ttsFallbackHandler.removeCallbacks(pendingFinalTtsFallback);
                    pendingFinalTtsFallback = null;
                }

                // 兜底策略：等待服务端 TTS_AUDIO 800ms，若没来再本地 TTS
                pendingFinalTtsFallback = () -> {
                    if (seq != ttsFallbackSeq) return; // 已被更新，说明新消息或音频已到

                    long sinceServerAudio = System.currentTimeMillis() - lastServerTtsArrivedAtMs;
                    if (sinceServerAudio >= 0 && sinceServerAudio < 2500) {
                        return; // 刚收到过服务端音频，避免重复播报
                    }

                    if (tts != null && !lastFinalText.isEmpty()) {
                        try {
                            tts.speak(lastFinalText, android.speech.tts.TextToSpeech.QUEUE_FLUSH, null, "final-fallback");
                            android.util.Log.i("MainActivity", "Fallback local TTS: " + lastFinalText);
                        } catch (Exception e) {
                            android.util.Log.e("MainActivity", "Fallback local TTS failed: " + e.getMessage(), e);
                        }
                    }
                };
                ttsFallbackHandler.postDelayed(pendingFinalTtsFallback, FINAL_FALLBACK_DELAY_MS);
            }


            @Override
            public void onInit(String payload) {
                runOnUiThread(() -> tvStatus.setText("INIT received"));
            }

            @Override
            public void onTtsAudio(String format, String b64) {
                handleIncomingTtsAudio(format, b64, "ws_ui");
            }
        });
        uiClient.connect();

        audioClient = new AudioWebSocketClient(new AudioWebSocketClient.Listener() {
            @Override
            public void onLog(String msg) {
                runOnUiThread(() -> tvStatus.setText(msg));
            }

            @Override
            public void onError(String msg) {
                runOnUiThread(() -> tvStatus.setText("Audio error: " + msg));
            }

            @Override
            public void onTtsAudio(String format, String b64) {
                handleIncomingTtsAudio(format, b64, "ws_audio");
            }
        });
        audioClient.connectAndStart();

        cameraClient = new CameraWebSocketClient(new CameraWebSocketClient.Listener() {
            @Override
            public void onOpen() {
                runOnUiThread(() -> tvStatus.setText("ws_camera connected"));
            }

            @Override
            public void onClosed() {
                runOnUiThread(() -> tvStatus.setText("ws_camera closed"));
            }

            @Override
            public void onError(String msg) {
                runOnUiThread(() -> tvStatus.setText("Camera WS error: " + msg));
            }
        });
        cameraClient.connect();

        cameraPipeline = new CameraPipeline(this, this, previewView, jpeg -> {
            if (cameraClient != null) {
                cameraClient.sendJpeg(jpeg);
            }
        });
        cameraPipeline.start();

        streaming = true;
        btnToggle.setText("Stop Phone Streaming");
    }

    private void stopStreaming() {
        if (uiClient != null) uiClient.close();
        if (audioClient != null) audioClient.stop();
        if (cameraClient != null) cameraClient.close();
        if (cameraPipeline != null) cameraPipeline.stop();

        if (pendingFinalTtsFallback != null) {
            ttsFallbackHandler.removeCallbacks(pendingFinalTtsFallback);
            pendingFinalTtsFallback = null;
        }

        if (ttsAudioPlayer != null) {
            try { ttsAudioPlayer.stop(); } catch (Exception ignored) {}
            try { ttsAudioPlayer.release(); } catch (Exception ignored) {}
            ttsAudioPlayer = null;
        }

        if (tts != null) {
            try { tts.stop(); } catch (Exception ignored) {}
        }

        streaming = false;
        btnToggle.setText("Start Phone Streaming");
        tvStatus.setText("Stopped");
    }

    @Override
    protected void onDestroy() {
        stopStreaming();
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
        super.onDestroy();
    }

    private String normalizeSpeechText(String raw) {
        if (raw == null) return "";
        String s = raw.trim();
        if (s.isEmpty()) return "";

        // 兼容后端 UI 前缀，避免把标记本身读出来
        if (s.startsWith("[AI]")) {
            s = s.substring(4).trim();
        } else if (s.startsWith("[导航]")) {
            s = s.substring(4).trim();
        } else if (s.startsWith("[系统]")) {
            s = s.substring(4).trim();
        }

        // 仅展示识别结果的提示，不需要语音播报
        if (s.startsWith("已识别:")) {
            return "";
        }

        return s;
    }

    private void handleIncomingTtsAudio(String format, String b64, String channel) {
        if (b64 == null || b64.isEmpty()) return;

        long now = System.currentTimeMillis();
        String head = b64.length() > 24 ? b64.substring(0, 24) : b64;
        String signature = (format == null ? "" : format) + "|" + head + "|" + b64.length();

        synchronized (this) {
            if (signature.equals(lastTtsMsgSignature) && (now - lastTtsMsgAtMs) < 1200) {
                return;
            }
            lastTtsMsgSignature = signature;
            lastTtsMsgAtMs = now;
        }

        runOnUiThread(() -> {
            lastServerTtsArrivedAtMs = now;
            ttsFallbackSeq++;
            if (pendingFinalTtsFallback != null) {
                ttsFallbackHandler.removeCallbacks(pendingFinalTtsFallback);
                pendingFinalTtsFallback = null;
            }
        });

        android.util.Log.i("MainActivity", "TTS_AUDIO from " + channel + ", format=" + format + ", len=" + b64.length());
        playBase64AudioToSpeaker(format, b64);
    }

    // ====== 贴到 MainActivity 类里（任意方法外，类内部）======
    private void playBase64AudioToSpeaker(String format, String b64) {
        if (b64 == null || b64.isEmpty()) return;

        try {
            byte[] audioBytes = android.util.Base64.decode(b64, android.util.Base64.DEFAULT);

            // 用临时文件交给 MediaPlayer，兼容 mp3/wav
            String ext = "mp3";
            if (format != null) {
                String f = format.trim().toLowerCase(java.util.Locale.ROOT);
                if (f.contains("wav")) ext = "wav";
            }

            java.io.File outFile = new java.io.File(getCacheDir(), "tts_from_server." + ext);
            try (java.io.FileOutputStream fos = new java.io.FileOutputStream(outFile, false)) {
                fos.write(audioBytes);
                fos.flush();
            }

            runOnUiThread(() -> {
                try {
                    // 停掉本地 TTS，避免和服务端音频打架
                    if (tts != null) {
                        tts.stop();
                    }

                    if (ttsAudioPlayer != null) {
                        try { ttsAudioPlayer.stop(); } catch (Exception ignored) {}
                        try { ttsAudioPlayer.release(); } catch (Exception ignored) {}
                        ttsAudioPlayer = null;
                    }

                    ttsAudioPlayer = new android.media.MediaPlayer();
                    ttsAudioPlayer.setDataSource(outFile.getAbsolutePath());
                    ttsAudioPlayer.setAudioStreamType(android.media.AudioManager.STREAM_MUSIC);
                    ttsAudioPlayer.setOnPreparedListener(android.media.MediaPlayer::start);
                    ttsAudioPlayer.setOnCompletionListener(mp -> {
                        try { mp.release(); } catch (Exception ignored) {}
                        ttsAudioPlayer = null;
                    });
                    ttsAudioPlayer.setOnErrorListener((mp, what, extra) -> {
                        try { mp.release(); } catch (Exception ignored) {}
                        ttsAudioPlayer = null;
                        return true;
                    });
                    ttsAudioPlayer.prepareAsync();
                } catch (Exception e) {
                    android.util.Log.e("MainActivity", "playBase64AudioToSpeaker failed: " + e.getMessage(), e);
                }
            });
        } catch (Exception e) {
            android.util.Log.e("MainActivity", "decode TTS audio failed: " + e.getMessage(), e);
        }
    }
}
