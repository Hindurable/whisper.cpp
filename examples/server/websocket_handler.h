#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <iostream>

#include "whisper.h"

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;

// Forward declarations
class WebSocketSession;
class Listener;

// Audio format enum
enum class AudioFormat {
    PCM_FLOAT32,
    PCM_INT16
};

// Ring buffer for audio samples
class AudioRingBuffer {
public:
    explicit AudioRingBuffer(size_t capacity = 16000 * 30) // 30 seconds at 16kHz
        : capacity_(capacity) {}

    void push(const std::vector<float>& samples) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const float& sample : samples) {
            buffer_.push_back(sample);
            if (buffer_.size() > capacity_) {
                buffer_.pop_front();
            }
        }
    }

    std::vector<float> get_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::vector<float>(buffer_.begin(), buffer_.end());
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.clear();
    }

    size_t size() const {
        return buffer_.size();
    }

private:
    std::deque<float> buffer_;
    const size_t capacity_;
    std::mutex mutex_;
};

// Audio processor class
class AudioProcessor {
public:
    struct Config {
        float silence_threshold = -40.0f;     // dB, default silence detection threshold
        float min_audio_level = -60.0f;       // dB, minimum audio level for processing
        float silence_duration = 0.5f;        // seconds, duration of silence to trigger processing
        float min_audio_duration = 2.0f;      // seconds, minimum duration of audio to process
    };

    AudioProcessor(whisper_context* ctx, whisper_full_params params) 
        : ctx_(ctx), params_(params), format_(AudioFormat::PCM_FLOAT32) {}

    void process_binary(const std::string& message, websocket::stream<tcp::socket>& ws);
    void update_config(const Config& new_config) { config_ = new_config; }
    const Config& get_config() const { return config_; }
    void set_audio_format(AudioFormat format) { format_ = format; }
    AudioFormat get_audio_format() const { return format_; }
    
private:
    std::vector<float> convert_pcm16_to_float(const int16_t* samples, size_t n_samples);
    float calculate_rms(const std::vector<float>& audio);
    bool detect_silence(const std::vector<float>& audio);
    bool is_too_quiet(const std::vector<float>& audio);
    std::string get_transcription();
    bool parse_config_message(const std::string& message);

    whisper_context* ctx_;
    whisper_full_params params_;
    std::vector<float> buffer_;
    AudioFormat format_;
    Config config_;
};

// WebSocket session class
class WebSocketSession : public std::enable_shared_from_this<WebSocketSession> {
public:
    WebSocketSession(tcp::socket socket, whisper_context* ctx, whisper_full_params params);
    ~WebSocketSession();

    void run();
    AudioProcessor& processor() { return processor_; }

private:
    void on_accept(beast::error_code ec);
    void do_read();
    void on_read(beast::error_code ec, std::size_t bytes_transferred);

    websocket::stream<tcp::socket> ws_;
    beast::flat_buffer buffer_;
    AudioProcessor processor_;
    tcp::endpoint client_endpoint_;
};

// Listener class for accepting WebSocket connections
class Listener : public std::enable_shared_from_this<Listener> {
public:
    Listener(
        net::io_context& ioc,
        tcp::endpoint endpoint,
        whisper_context* ctx,
        whisper_full_params params,
        const AudioProcessor::Config& config = AudioProcessor::Config()
    );

    void run() {
        do_accept();
    }

private:
    void do_accept();
    void on_accept(beast::error_code ec, tcp::socket socket);
    void fail(beast::error_code ec, char const* what) {
        std::cerr << what << ": " << ec.message() << "\n";
    }

    net::io_context& ioc_;
    tcp::acceptor acceptor_;
    whisper_context* ctx_;
    whisper_full_params params_;
    AudioProcessor::Config config_;
};

// Start WebSocket server
void start_websocket_server(
    net::io_context& ioc,
    unsigned short port,
    whisper_context* ctx,
    whisper_full_params params,
    const AudioProcessor::Config& config = AudioProcessor::Config()
);
