package com.sofia.blindnav;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;

public class AudioWebSocketClient {
    public interface Listener {
        void onLog(String msg);
        void onError(String msg);
        void onTtsAudio(String format, String b64);
    }

    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNEL_MASK = AudioFormat.CHANNEL_IN_MONO;
    private static final int ENCODING = AudioFormat.ENCODING_PCM_16BIT;

    private final OkHttpClient client = new OkHttpClient();
    private final ExecutorService audioExecutor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean recording = new AtomicBoolean(false);
    private final Listener listener;

    private WebSocket ws;
    private AudioRecord audioRecord;

    public AudioWebSocketClient(Listener listener) {
        this.listener = listener;
    }

    public void connectAndStart() {
        Request request = new Request.Builder().url(NetworkConfig.wsAudioUrl()).build();
        ws = client.newWebSocket(request, new WebSocketListener() {
            @Override
            public void onOpen(WebSocket webSocket, Response response) {
                log("ws_audio connected");
                webSocket.send("START");
                startMicLoop();
            }

            @Override
            public void onMessage(WebSocket webSocket, String text) {
                log("ws_audio msg: " + text);
                if (text != null && text.startsWith("TTS_AUDIO:")) {
                    String[] parts = text.split(":", 3);
                    if (parts.length == 3 && listener != null) {
                        listener.onTtsAudio(parts[1], parts[2]);
                    }
                    return;
                }
                if (text != null && text.startsWith("REJECT:")) {
                    stop();
                }
            }

            @Override
            public void onFailure(WebSocket webSocket, Throwable t, Response response) {
                error(t == null ? "ws_audio failure" : t.getMessage());
                stopRecordingOnly();
            }

            @Override
            public void onClosed(WebSocket webSocket, int code, String reason) {
                log("ws_audio closed: " + code + " " + reason);
                stopRecordingOnly();
            }
        });
    }

    private void startMicLoop() {
        if (!recording.compareAndSet(false, true)) return;

        int minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_MASK, ENCODING);
        int bufferSize = Math.max(minBuffer, 3200);
        audioRecord = new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_MASK,
                ENCODING,
                bufferSize
        );

        audioExecutor.execute(() -> {
            byte[] pcmBuffer = new byte[3200];
            try {
                audioRecord.startRecording();
                while (recording.get() && ws != null) {
                    int read = audioRecord.read(pcmBuffer, 0, pcmBuffer.length);
                    if (read > 0) {
                        ws.send(okio.ByteString.of(pcmBuffer, 0, read));
                    }
                }
            } catch (Throwable t) {
                error("Mic loop error: " + t.getMessage());
            } finally {
                stopRecordingOnly();
            }
        });
    }

    private void stopRecordingOnly() {
        recording.set(false);
        if (audioRecord != null) {
            try {
                audioRecord.stop();
            } catch (Exception ignored) {
            }
            try {
                audioRecord.release();
            } catch (Exception ignored) {
            }
            audioRecord = null;
        }
    }

    public void stop() {
        if (ws != null) {
            try {
                ws.send("STOP");
            } catch (Exception ignored) {
            }
            ws.close(1000, "stop");
            ws = null;
        }
        stopRecordingOnly();
    }

    private void log(String msg) {
        if (listener != null) listener.onLog(msg);
    }

    private void error(String msg) {
        if (listener != null) listener.onError(msg);
    }
}
