package com.sofia.blindnav;

public final class NetworkConfig {
    private NetworkConfig() {}

    // Replace with your server LAN IP, e.g. 192.168.1.100
    public static final String SERVER_HOST = "192.168.1.100";
    public static final int SERVER_PORT = 8081;

    public static String wsUiUrl() {
        return "ws://" + SERVER_HOST + ":" + SERVER_PORT + "/ws_ui";
    }

    public static String wsAudioUrl() {
        return "ws://" + SERVER_HOST + ":" + SERVER_PORT + "/ws_audio?source=phone";
    }

    public static String wsCameraUrl() {
        return "ws://" + SERVER_HOST + ":" + SERVER_PORT + "/ws/camera?source=phone";
    }

    public static String runtimeConfigUrl() {
        return "http://" + SERVER_HOST + ":" + SERVER_PORT + "/api/runtime/config";
    }
}
