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
#include <chrono>
#include "whisper.h"

using namespace std;

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
        std::string language;  // Optional language override
        bool auto_settings = false;          // Enable automatic parameter adjustment
    };

    struct NoiseStats {
        float background_noise_level;  // RMS level of background noise
        float background_noise_max;    // Maximum observed noise level
        float background_noise_min;    // Minimum observed noise level
        float speech_level;           // Average RMS level of detected speech
        float speech_level_max;       // Maximum observed speech level
        float speech_level_min;       // Minimum observed speech level
        size_t noise_samples;         // Number of noise samples analyzed
        size_t speech_samples;        // Number of speech samples analyzed
        
        // Speech pattern statistics
        float avg_speech_duration;   // Average duration of speech segments
        float avg_pause_duration;    // Average duration of pauses between speech
        size_t false_triggers;       // Count of probable false triggers
        size_t missed_speech;        // Count of probable missed speech
        std::chrono::steady_clock::time_point last_adjustment;
        
        NoiseStats() : background_noise_level(-100.0f), background_noise_max(-100.0f),
                      background_noise_min(0.0f), speech_level(-100.0f),
                      speech_level_max(-100.0f), speech_level_min(0.0f),
                      noise_samples(0), speech_samples(0),
                      avg_speech_duration(0.0f), avg_pause_duration(0.0f),
                      false_triggers(0), missed_speech(0),
                      last_adjustment(std::chrono::steady_clock::now()) {}
    };

    AudioProcessor(whisper_context* ctx, whisper_full_params params) 
        : ctx_(ctx), params_(params), format_(AudioFormat::PCM_INT16),  // Default to PCM_INT16
          config_{-40.0f, -60.0f, 0.5f, 2.0f, "", false} {}

    void process_binary(const std::string& message, websocket::stream<tcp::socket>& ws);
    void update_config(const Config& new_config) { 
        config_ = new_config; 
        if (config_.auto_settings) {
            // Reset statistics when enabling auto settings
            noise_stats_ = NoiseStats();
            adaptive_state_ = AdaptiveState();
        }
    }
    const Config& get_config() const { return config_; }
    void set_audio_format(AudioFormat format) { format_ = format; }
    AudioFormat get_audio_format() const { return format_; }
    std::string get_transcription();
    void on_close();  // Added this method
    
private:
    std::vector<float> convert_pcm16_to_float(const int16_t* samples, size_t n_samples);
    float calculate_rms(const std::vector<float>& audio);
    bool detect_silence(const std::vector<float>& audio);
    bool is_too_quiet(const std::vector<float>& audio);
    bool parse_config_message(const std::string& message);
    float calculate_energy_variance(const std::vector<float>& samples);
    bool has_speech_characteristics(const std::vector<float>& samples);
    float get_segment_confidence();

    void update_noise_stats(const std::vector<float>& samples, bool is_silence);
    std::string get_recommended_config() const;
    void reset_noise_stats();

    whisper_context* ctx_;
    whisper_full_params params_;
    std::vector<float> buffer_;
    AudioFormat format_;
    Config config_;
    NoiseStats noise_stats_;
    
    // Adaptive settings tracking
    struct AdaptiveState {
        std::chrono::steady_clock::time_point speech_start;
        std::chrono::steady_clock::time_point last_speech_end;
        std::chrono::steady_clock::time_point last_good_transcription;
        bool in_speech;
        float confidence_sum;
        size_t confidence_samples;
        
        // Good settings memory
        struct SettingsMemory {
            float silence_threshold;
            float min_audio_level;
            float silence_duration;
            float min_audio_duration;
            float confidence;
            bool has_good_settings;
            
            SettingsMemory() : has_good_settings(false), confidence(0.0f) {}
        } memory;
        
        // Adjustment parameters
        static constexpr float ADJUSTMENT_INTERVAL_SEC = 10.0f;     // Base interval for adjustments
        static constexpr float MAX_ADJUSTMENT_STEP = 2.0f;         // Maximum dB change per adjustment
        static constexpr float CONFIDENCE_THRESHOLD = 0.3f;        // Threshold for good transcription
        static constexpr float GOOD_SETTINGS_THRESHOLD = 0.8f;     // Threshold to consider settings "good"
        static constexpr float MAX_SILENCE_WITHOUT_DECAY = 30.0f;  // After this much silence, start decaying adjustments
        static constexpr float DECAY_FACTOR = 0.5f;               // How much to reduce adjustment step size during silence
        
        AdaptiveState() : in_speech(false), confidence_sum(0.0f), confidence_samples(0),
                         last_good_transcription(std::chrono::steady_clock::now()) {}
                         
        float get_adjustment_factor(std::chrono::steady_clock::time_point now) const {
            using namespace std::chrono;
            float silence_duration = duration_cast<duration<float>>(
                now - last_good_transcription).count();
                
            if (silence_duration <= MAX_SILENCE_WITHOUT_DECAY) {
                return 1.0f;  // Full adjustment
            }
            
            // Exponential decay after MAX_SILENCE_WITHOUT_DECAY
            float excess_silence = silence_duration - MAX_SILENCE_WITHOUT_DECAY;
            return std::exp(-DECAY_FACTOR * excess_silence);
        }
    } adaptive_state_;

    // New methods for adaptive settings
    void update_adaptive_settings();
    void track_speech_pattern(bool is_speech, float confidence);
    void adjust_parameters();
    bool should_adjust_parameters() const;
};

// WebSocket session class
class WebSocketSession : public std::enable_shared_from_this<WebSocketSession> {
public:
    WebSocketSession(tcp::socket socket, whisper_context* ctx, whisper_full_params params);
    ~WebSocketSession();

    void run();
    AudioProcessor& processor() { return processor_; }
    void on_close();  // Added this method
    
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
