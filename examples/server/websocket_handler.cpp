#include "websocket_handler.h"
#include "common.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio.hpp>

// Listener implementation
Listener::Listener(
    net::io_context& ioc,
    tcp::endpoint endpoint,
    whisper_context* ctx,
    whisper_full_params params,
    const AudioProcessor::Config& config
)
    : ioc_(ioc)
    , acceptor_(ioc)
    , ctx_(ctx)
    , params_(params)
    , config_(config)
{
    beast::error_code ec;

    acceptor_.open(endpoint.protocol(), ec);
    if (ec) {
        fail(ec, "open");
        return;
    }

    acceptor_.set_option(net::socket_base::reuse_address(true), ec);
    if (ec) {
        fail(ec, "set_option");
        return;
    }

    acceptor_.bind(endpoint, ec);
    if (ec) {
        fail(ec, "bind");
        return;
    }

    acceptor_.listen(net::socket_base::max_listen_connections, ec);
    if (ec) {
        fail(ec, "listen");
        return;
    }
}

void Listener::do_accept() {
    acceptor_.async_accept(
        net::make_strand(ioc_),
        beast::bind_front_handler(&Listener::on_accept, shared_from_this()));
}

void Listener::on_accept(beast::error_code ec, tcp::socket socket) {
    if (ec) {
        std::cerr << "accept: " << ec.message() << std::endl;
    } else {
        auto session = std::make_shared<WebSocketSession>(std::move(socket), ctx_, params_);
        session->processor().update_config(config_);  // Set initial config from command line
        session->run();
        std::cerr << "Client connected to WebSocket server" << std::endl;
    }

    do_accept();
}

// WebSocketSession implementation
WebSocketSession::WebSocketSession(tcp::socket socket, whisper_context* ctx, whisper_full_params params)
    : ws_(std::move(socket)), processor_(ctx, params) {
    try {
        client_endpoint_ = ws_.next_layer().remote_endpoint();
    } catch(...) {
        client_endpoint_ = tcp::endpoint();
    }
}

WebSocketSession::~WebSocketSession() {
    std::cerr << "Client disconnected from WebSocket server [" 
              << client_endpoint_.address().to_string() << "]" << std::endl;
}

void WebSocketSession::run() {
    ws_.set_option(websocket::stream_base::decorator(
        [](websocket::response_type& res) {
            res.set(beast::http::field::server, "Whisper WebSocket Server");
        }));

    std::cerr << "New WebSocket connection from " 
              << client_endpoint_.address().to_string() << std::endl;

    ws_.async_accept(beast::bind_front_handler(&WebSocketSession::on_accept, shared_from_this()));
}

void WebSocketSession::on_accept(beast::error_code ec) {
    if (ec) {
        return;
    }
    do_read();
}

void WebSocketSession::do_read() {
    ws_.async_read(buffer_, beast::bind_front_handler(&WebSocketSession::on_read, shared_from_this()));
}

void WebSocketSession::on_read(beast::error_code ec, std::size_t bytes_transferred) {
    if (ec) {
        return;
    }

    std::string message = beast::buffers_to_string(buffer_.data());
    buffer_.consume(buffer_.size());

    processor_.process_binary(message, ws_);
    do_read();
}

void WebSocketSession::on_close() {
    processor_.on_close();
    std::cout << "WebSocket connection closed" << std::endl;
}

// AudioProcessor implementation
struct NoiseStats {
    float background_noise_level = 0.0f;
    float background_noise_max = 0.0f;
    float background_noise_min = 0.0f;
    float speech_level = 0.0f;
    float speech_level_max = 0.0f;
    float speech_level_min = 0.0f;
    int noise_samples = 0;
    int speech_samples = 0;
    float avg_speech_duration = 0.0f;
    float avg_pause_duration = 0.0f;
    int false_triggers = 0;
    std::chrono::steady_clock::time_point last_adjustment;
};

struct AdaptiveState {
    static constexpr float CONFIDENCE_THRESHOLD = 0.5f;
    static constexpr float ADJUSTMENT_INTERVAL_SEC = 10.0f;
    static constexpr float MAX_ADJUSTMENT_STEP = 1.0f;
    static constexpr float GOOD_SETTINGS_THRESHOLD = 0.8f;

    bool in_speech = false;
    std::chrono::steady_clock::time_point speech_start;
    std::chrono::steady_clock::time_point last_speech_end;
    float confidence_sum = 0.0f;
    int confidence_samples = 0;
    std::chrono::steady_clock::time_point last_good_transcription;
    struct SettingsMemory {
        bool has_good_settings = false;
        float silence_threshold = 0.0f;
        float min_audio_level = 0.0f;
        float silence_duration = 0.0f;
        float min_audio_duration = 0.0f;
        float confidence = 0.0f;
    } memory;

    float get_adjustment_factor(const std::chrono::steady_clock::time_point& now) {
        using namespace std::chrono;
        float time_since_last = duration_cast<duration<float>>(now - last_good_transcription).count();
        float decay_factor = std::max(0.0f, 1.0f - time_since_last / 60.0f);  // Decay over 1 minute
        return decay_factor;
    }
};

float AudioProcessor::calculate_rms(const std::vector<float>& audio) {
    if (audio.empty()) return -100.0f;
    
    float sum = 0.0f;
    for (float sample : audio) {
        sum += sample * sample;
    }
    float rms = std::sqrt(sum / audio.size());
    return 20.0f * std::log10(rms);
}

bool AudioProcessor::detect_silence(const std::vector<float>& audio) {
    if (audio.size() < config_.silence_duration * 16000) { // Convert seconds to samples
        return false;
    }
    float rms_db = calculate_rms(audio);
    return rms_db < config_.silence_threshold;
}

bool AudioProcessor::is_too_quiet(const std::vector<float>& audio) {
    float rms_db = calculate_rms(audio);
    return rms_db < config_.min_audio_level;
}

std::vector<float> AudioProcessor::convert_pcm16_to_float(const int16_t* samples, size_t n_samples) {
    std::vector<float> float_samples;
    float_samples.reserve(n_samples);
    
    // Convert PCM16 to float32 (normalize to [-1, 1])
    const float scale = 1.0f / 32768.0f;
    for (size_t i = 0; i < n_samples; i++) {
        float_samples.push_back(samples[i] * scale);
    }
    
    return float_samples;
}

bool AudioProcessor::parse_config_message(const std::string& message) {
    try {
        // Quick validation - must start with { and contain "format" or some config field
        if (message.empty() || message[0] != '{' || 
            (message.find("\"format\"") == std::string::npos && 
             message.find("\"silence_threshold\"") == std::string::npos &&
             message.find("\"auto_settings\"") == std::string::npos &&
             message.find("\"lang\"") == std::string::npos)) {
            return false;
        }

        auto extract_value = [](const std::string& json, const std::string& key) 
            -> std::pair<bool, std::string> {
            size_t pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) {
                return {false, ""};
            }
            
            pos = json.find(':', pos);
            if (pos == std::string::npos) {
                return {false, ""};
            }
            
            // Skip whitespace
            pos++;
            while (pos < json.length() && std::isspace(json[pos])) {
                pos++;
            }
            
            std::string value;
            if (pos < json.length()) {
                if (json[pos] == '"') {
                    // String value
                    size_t end = json.find('"', pos + 1);
                    if (end != std::string::npos) {
                        value = json.substr(pos + 1, end - pos - 1);
                    }
                } else {
                    // Number or boolean value
                    size_t end = pos;
                    while (end < json.length() && 
                           (std::isdigit(json[end]) || json[end] == '.' || json[end] == '-' ||
                            json[end] == 'e' || json[end] == 'E' || json[end] == '+' ||
                            json[end] == 't' || json[end] == 'r' || json[end] == 'u' ||
                            json[end] == 'f' || json[end] == 'a' || json[end] == 'l' ||
                            json[end] == 's' || json[end] == 'e')) {
                        end++;
                    }
                    value = json.substr(pos, end - pos);
                }
            }
            
            return {true, value};
        };

        // Parse format first
        auto [format_found, format_str] = extract_value(message, "format");
        if (format_found) {
            if (format_str == "pcm16") {
                format_ = AudioFormat::PCM_INT16;
            } else if (format_str == "float32") {
                format_ = AudioFormat::PCM_FLOAT32;
            }
        }
        
        auto parse_float = [](const std::pair<bool, std::string>& result, float& target) {
            if (result.first) {
                try {
                    target = std::stof(result.second);
                } catch (...) {
                    // Keep existing value if parsing fails
                }
            }
        };
        
        auto parse_bool = [](const std::pair<bool, std::string>& result, bool& target) {
            if (result.first) {
                std::string value = result.second;
                // Convert to lowercase for comparison
                std::transform(value.begin(), value.end(), value.begin(), ::tolower);
                if (value == "true") {
                    target = true;
                } else if (value == "false") {
                    target = false;
                }
                // Keep existing value if parsing fails
            }
        };

        parse_float(extract_value(message, "silence_threshold"), config_.silence_threshold);
        parse_float(extract_value(message, "min_audio_level"), config_.min_audio_level);
        parse_float(extract_value(message, "silence_duration"), config_.silence_duration);
        parse_float(extract_value(message, "min_audio_duration"), config_.min_audio_duration);
        parse_bool(extract_value(message, "auto_settings"), config_.auto_settings);

        // Parse optional language field
        auto lang_result = extract_value(message, "lang");
        if (lang_result.first) {
            config_.language = lang_result.second;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing config message: " << e.what() << std::endl;
        return false;
    }
}

void AudioProcessor::update_noise_stats(const std::vector<float>& samples, bool is_silence) {
    float rms = calculate_rms(samples);
    
    if (is_silence) {
        // Update background noise level with exponential moving average
        if (noise_stats_.noise_samples == 0) {
            noise_stats_.background_noise_level = rms;
            noise_stats_.background_noise_max = rms;
            noise_stats_.background_noise_min = rms;
        } else {
            const float alpha = 0.1f;  // Smoothing factor
            noise_stats_.background_noise_level = (1-alpha) * noise_stats_.background_noise_level + alpha * rms;
            noise_stats_.background_noise_max = std::max(noise_stats_.background_noise_max, rms);
            noise_stats_.background_noise_min = std::min(noise_stats_.background_noise_min, rms);
        }
        noise_stats_.noise_samples++;
    } else {
        // Update speech level with exponential moving average
        if (noise_stats_.speech_samples == 0) {
            noise_stats_.speech_level = rms;
            noise_stats_.speech_level_max = rms;
            noise_stats_.speech_level_min = rms;
        } else {
            const float alpha = 0.1f;  // Smoothing factor
            noise_stats_.speech_level = (1-alpha) * noise_stats_.speech_level + alpha * rms;
            noise_stats_.speech_level_max = std::max(noise_stats_.speech_level_max, rms);
            noise_stats_.speech_level_min = std::min(noise_stats_.speech_level_min, rms);
        }
        noise_stats_.speech_samples++;
    }
}

std::string AudioProcessor::get_recommended_config() const {
    if (noise_stats_.speech_samples < 100 || noise_stats_.noise_samples < 5) {
        return "";  // Not enough data for recommendations
    }

    // Calculate optimal silence_threshold:
    // - Should be between speech_level_avg and noise_level_max
    // - Closer to speech_level if we have more speech samples
    float speech_weight = std::min(1.0f, noise_stats_.speech_samples / 500.0f);
    float silence_threshold = noise_stats_.noise_level_max * (1 - speech_weight) + 
                            noise_stats_.speech_level_avg * speech_weight;
    
    // Calculate optimal min_audio_level:
    // - Should be below speech_level_avg but above noise_level_avg
    float min_audio_level = (noise_stats_.noise_level_avg + noise_stats_.speech_level_avg) / 2;
    
    // Calculate optimal silence_duration:
    // - Should be proportional to avg_pause_duration but not too long
    float silence_duration = std::min(0.8f, std::max(0.3f, noise_stats_.avg_pause_duration * 0.8f));
    
    // Calculate optimal min_audio_duration:
    // - Should be proportional to avg_speech_duration
    float min_audio_duration = std::min(2.0f, std::max(1.0f, noise_stats_.avg_speech_duration * 1.2f));

    // Build recommendations JSON
    std::string recommendations = "{\"recommendations\":{" +
        std::string("\"silence_threshold\":") + std::to_string(silence_threshold) + "," +
        "\"min_audio_level\":" + std::to_string(min_audio_level) + "," +
        "\"silence_duration\":" + std::to_string(silence_duration) + "," +
        "\"min_audio_duration\":" + std::to_string(min_audio_duration) + "," +
        "\"noise_level_avg\":" + std::to_string(noise_stats_.noise_level_avg) + "," +
        "\"noise_level_max\":" + std::to_string(noise_stats_.noise_level_max) + "," +
        "\"noise_level_min\":" + std::to_string(noise_stats_.noise_level_min) + "," +
        "\"speech_level_avg\":" + std::to_string(noise_stats_.speech_level_avg) + "," +
        "\"speech_level_max\":" + std::to_string(noise_stats_.speech_level_max) + "," +
        "\"speech_level_min\":" + std::to_string(noise_stats_.speech_level_min) + "," +
        "\"noise_samples\":" + std::to_string(noise_stats_.noise_samples) + "," +
        "\"speech_samples\":" + std::to_string(noise_stats_.speech_samples) + "," +
        "\"avg_speech_duration\":" + std::to_string(noise_stats_.avg_speech_duration) + "," +
        "\"avg_pause_duration\":" + std::to_string(noise_stats_.avg_pause_duration) + "}}";

    return recommendations;
}

void AudioProcessor::reset_noise_stats() {
    noise_stats_ = NoiseStats();
}

void AudioProcessor::track_speech_pattern(bool is_speech, float confidence) {
    using namespace std::chrono;
    auto now = steady_clock::now();
    
    // Update last good transcription time if confidence is high
    if (confidence > AdaptiveState::GOOD_SETTINGS_THRESHOLD) {
        adaptive_state_.last_good_transcription = now;
        
        // Store current settings if they're better than previous
        if (!adaptive_state_.memory.has_good_settings || 
            confidence > adaptive_state_.memory.confidence) {
            adaptive_state_.memory.silence_threshold = config_.silence_threshold;
            adaptive_state_.memory.min_audio_level = config_.min_audio_level;
            adaptive_state_.memory.silence_duration = config_.silence_duration;
            adaptive_state_.memory.min_audio_duration = config_.min_audio_duration;
            adaptive_state_.memory.confidence = confidence;
            adaptive_state_.memory.has_good_settings = true;
        }
    }
    
    if (is_speech && !adaptive_state_.in_speech) {
        // Speech started
        adaptive_state_.speech_start = now;
        adaptive_state_.in_speech = true;
        
        if (adaptive_state_.last_speech_end.time_since_epoch().count() > 0) {
            // Calculate pause duration
            float pause_duration = duration_cast<duration<float>>(
                now - adaptive_state_.last_speech_end).count();
            
            // Update average pause duration with exponential moving average
            if (noise_stats_.avg_pause_duration == 0.0f) {
                noise_stats_.avg_pause_duration = pause_duration;
            } else {
                const float alpha = 0.1f;
                noise_stats_.avg_pause_duration = (1-alpha) * noise_stats_.avg_pause_duration + alpha * pause_duration;
            }
        }
    } else if (!is_speech && adaptive_state_.in_speech) {
        // Speech ended
        adaptive_state_.last_speech_end = now;
        adaptive_state_.in_speech = false;
        
        // Calculate speech duration
        float speech_duration = duration_cast<duration<float>>(
            now - adaptive_state_.speech_start).count();
        
        // Update average speech duration with exponential moving average
        if (noise_stats_.avg_speech_duration == 0.0f) {
            noise_stats_.avg_speech_duration = speech_duration;
        } else {
            const float alpha = 0.1f;
            noise_stats_.avg_speech_duration = (1-alpha) * noise_stats_.avg_speech_duration + alpha * speech_duration;
        }
        
        // Track confidence
        adaptive_state_.confidence_sum += confidence;
        adaptive_state_.confidence_samples++;
        
        // Detect potential false triggers
        if (confidence < AdaptiveState::CONFIDENCE_THRESHOLD) {
            noise_stats_.false_triggers++;
        }
    }
}

bool AudioProcessor::should_adjust_parameters() const {
    using namespace std::chrono;
    auto now = steady_clock::now();
    float time_since_last = duration_cast<duration<float>>(
        now - noise_stats_.last_adjustment).count();
    
    return time_since_last >= AdaptiveState::ADJUSTMENT_INTERVAL_SEC;
}

void AudioProcessor::adjust_parameters() {
    if (!should_adjust_parameters()) {
        return;
    }
    
    using namespace std::chrono;
    auto now = steady_clock::now();
    noise_stats_.last_adjustment = now;
    
    // Calculate average confidence
    float avg_confidence = adaptive_state_.confidence_samples > 0
        ? adaptive_state_.confidence_sum / adaptive_state_.confidence_samples
        : 0.0f;
    
    // Reset confidence tracking
    adaptive_state_.confidence_sum = 0.0f;
    adaptive_state_.confidence_samples = 0;
    
    // Get adjustment decay factor based on silence duration
    float adjustment_factor = adaptive_state_.get_adjustment_factor(now);
    
    // If we have good settings and confidence is dropping, revert towards good settings
    if (adaptive_state_.memory.has_good_settings && 
        avg_confidence < adaptive_state_.memory.confidence * 0.8f) {
        
        // Helper for reverting to remembered good settings
        auto revert_param = [adjustment_factor](float& current, float good, float max_step) {
            float diff = good - current;
            float step = std::clamp(diff, -max_step, max_step) * adjustment_factor;
            current += step;
        };
        
        // Revert towards good settings
        revert_param(config_.silence_threshold, 
                    adaptive_state_.memory.silence_threshold, 
                    AdaptiveState::MAX_ADJUSTMENT_STEP);
        revert_param(config_.min_audio_level,
                    adaptive_state_.memory.min_audio_level,
                    AdaptiveState::MAX_ADJUSTMENT_STEP);
        revert_param(config_.silence_duration,
                    adaptive_state_.memory.silence_duration,
                    0.1f);
        revert_param(config_.min_audio_duration,
                    adaptive_state_.memory.min_audio_duration,
                    0.2f);
                    
        // Early return since we're reverting to known good settings
        return;
    }
    
    // Helper for clamped parameter adjustment
    auto adjust_param = [adjustment_factor](float& param, float target, float max_step) {
        float diff = target - param;
        float step = std::clamp(diff, -max_step, max_step) * adjustment_factor;
        param += step;
    };
    
    // Normal parameter adjustment logic with decay factor applied
    if (noise_stats_.false_triggers > 5) {
        float target = config_.silence_threshold - 2.0f;
        adjust_param(config_.silence_threshold, target, AdaptiveState::MAX_ADJUSTMENT_STEP);
        noise_stats_.false_triggers = 0;
    } else if (avg_confidence < AdaptiveState::CONFIDENCE_THRESHOLD) {
        float target = config_.silence_threshold + 2.0f;
        adjust_param(config_.silence_threshold, target, AdaptiveState::MAX_ADJUSTMENT_STEP);
    }
    
    if (noise_stats_.avg_pause_duration > 0.0f) {
        float target_duration = noise_stats_.avg_pause_duration * 0.5f;
        target_duration = std::clamp(target_duration, 0.3f, 1.0f);
        adjust_param(config_.silence_duration, target_duration, 0.1f);
    }
    
    if (noise_stats_.avg_speech_duration > 0.0f) {
        float target_duration = noise_stats_.avg_speech_duration * 0.8f;
        target_duration = std::clamp(target_duration, 1.0f, 5.0f);
        adjust_param(config_.min_audio_duration, target_duration, 0.2f);
    }
    
    // Keep parameters within safe bounds
    config_.silence_threshold = std::clamp(config_.silence_threshold, -45.0f, -25.0f);
    config_.min_audio_level = std::clamp(config_.min_audio_level, -65.0f, -45.0f);
    config_.silence_duration = std::clamp(config_.silence_duration, 0.3f, 1.0f);
    config_.min_audio_duration = std::clamp(config_.min_audio_duration, 1.0f, 5.0f);
}

void AudioProcessor::update_adaptive_settings() {
    if (!config_.auto_settings) {
        return;
    }
    
    // Get current transcription confidence
    float confidence = get_segment_confidence();
    
    // Update speech pattern tracking
    bool is_speech = !detect_silence(buffer_);
    track_speech_pattern(is_speech, confidence);
    
    // Periodically adjust parameters
    adjust_parameters();
}

void AudioProcessor::process_binary(const std::string& message, websocket::stream<tcp::socket>& ws) {
    static bool config_sent = false;
    
    // Check if this is a configuration message (JSON)
    if (!config_sent && parse_config_message(message)) {
        // Send acknowledgment with current config
        ws.text(true);
        beast::error_code ec;
        std::string response = "{\"status\":\"ok\",\"format\":\"" + 
            std::string(format_ == AudioFormat::PCM_FLOAT32 ? "float32" : "pcm16") +
            "\",\"config\":{" +
            "\"silence_threshold\":" + std::to_string(config_.silence_threshold) +
            ",\"min_audio_level\":" + std::to_string(config_.min_audio_level) +
            ",\"silence_duration\":" + std::to_string(config_.silence_duration) +
            ",\"min_audio_duration\":" + std::to_string(config_.min_audio_duration) +
            ",\"auto_settings\":" + (config_.auto_settings ? "true" : "false");
        
        // Add language if set
        if (!config_.language.empty()) {
            response += ",\"lang\":\"" + config_.language + "\"";
        }
        
        response += "}}";
        
        ws.write(boost::asio::buffer(response), ec);
        std::cout << "Received client settings: " << message << std::endl;
        std::cout << "Sent acknowledgment: " << response << std::endl;
        config_sent = true;
        return;
    }

    // Process audio data only if it's not a config message
    if (!parse_config_message(message)) {
        std::vector<float> new_samples;
        size_t n_samples;

        if (format_ == AudioFormat::PCM_FLOAT32) {
            const float* samples = reinterpret_cast<const float*>(message.data());
            n_samples = message.size() / sizeof(float);
            new_samples.insert(new_samples.end(), samples, samples + n_samples);
        } else { // PCM_INT16
            const int16_t* samples = reinterpret_cast<const int16_t*>(message.data());
            n_samples = message.size() / sizeof(int16_t);
            new_samples = convert_pcm16_to_float(samples, n_samples);
        }

        // Add samples to buffer
        buffer_.insert(buffer_.end(), new_samples.begin(), new_samples.end());

        // Check if we have enough audio data
        float min_samples = config_.min_audio_duration * 16000;
        if (buffer_.size() >= min_samples) {
            // Check if the last portion is silence
            size_t silence_samples = static_cast<size_t>(config_.silence_duration * 16000);
            std::vector<float> last_samples(buffer_.end() - silence_samples, buffer_.end());
            
            float last_rms = calculate_rms(last_samples);
            float buffer_rms = calculate_rms(buffer_);
            
            bool is_silence = detect_silence(last_samples);
            // Update noise statistics
            update_noise_stats(last_samples, is_silence);
            
            if (is_silence) {
                if (is_too_quiet(buffer_)) {
                    buffer_.clear();
                    return;
                }
                
                std::string response = get_transcription();
                std::string recommendations = get_recommended_config();
                if (!recommendations.empty()) {
                    response.pop_back();
                    response += "," + recommendations.substr(1);
                }
                
                ws.text(true);
                beast::error_code ec;
                std::cout << "Generated JSON response: " << response << std::endl;
                ws.write(boost::asio::buffer(response), ec);
                if (ec) {
                    std::cerr << "Failed to send transcription: " << ec.message() << std::endl;
                } else {
                    std::cout << "Sent transcription: " << response << std::endl;
                }

                buffer_.clear();
            }
        }
        
        update_adaptive_settings();
    }
}

void AudioProcessor::on_close() {
    static bool config_sent = false;
    config_sent = false;
}

float AudioProcessor::calculate_energy_variance(const std::vector<float>& samples) {
    if (samples.empty()) return 0.0f;

    // Calculate mean energy
    float mean_energy = 0.0f;
    std::vector<float> energies(samples.size());
    
    for (size_t i = 0; i < samples.size(); i++) {
        energies[i] = samples[i] * samples[i];  // Square of amplitude
        mean_energy += energies[i];
    }
    mean_energy /= samples.size();

    // Calculate variance of energy
    float variance = 0.0f;
    for (const float& energy : energies) {
        float diff = energy - mean_energy;
        variance += diff * diff;
    }
    variance /= samples.size();

    return variance;
}

bool AudioProcessor::has_speech_characteristics(const std::vector<float>& samples) {
    // Speech typically has:
    // 1. Higher energy variance (due to syllables and pauses)
    // 2. Regular energy fluctuations
    
    const float MIN_VARIANCE = 1e-7f;  // Minimum variance for speech
    float variance = calculate_energy_variance(samples);
    
    return variance > MIN_VARIANCE;
}

float AudioProcessor::get_segment_confidence() {
    if (whisper_full_n_segments(ctx_) == 0) {
        return 0.0f;
    }

    // Get probability of the first token in the first segment
    float token_prob = whisper_full_get_token_p(ctx_, 0, 0);
    return token_prob;
}

std::string AudioProcessor::get_transcription() {
    const int n_samples = buffer_.size();
    if (n_samples == 0) {
        return "{\"error\":\"no audio data\"}";
    }

    // Check if the audio has speech-like characteristics
    if (!has_speech_characteristics(buffer_)) {
        return "{\"error\":\"no speech detected\"}";
    }

    // Process audio with whisper
    int result = whisper_full(ctx_, params_, buffer_.data(), buffer_.size());
    if (result != 0) {
        return "{\"error\":\"failed to process audio\"}";
    }

    int n_segments = whisper_full_n_segments(ctx_);
    if (n_segments <= 0) {
        return "{\"error\":\"no transcription\"}";
    }

    float confidence = get_segment_confidence();
    std::cout << "Transcription confidence: " << confidence << std::endl;
    
    if (confidence < 0.1f) {  
        return "{\"error\":\"low confidence\"}";
    }

    // Get the text from the first segment
    const char* text = whisper_full_get_segment_text(ctx_, 0);
    std::string escaped_text;
    
    // Escape special characters
    for (const char* p = text; *p; ++p) {
        if (*p == '"') {
            escaped_text += "\\\"";
        } else if (*p == '\\') {
            escaped_text += "\\\\";
        } else if (*p == '\n') {
            escaped_text += "\\n";
        } else if (*p == '\r') {
            escaped_text += "\\r";
        } else if (*p == '\t') {
            escaped_text += "\\t";
        } else if (static_cast<unsigned char>(*p) < 0x20) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(*p));
            escaped_text += buf;
        } else {
            escaped_text += *p;
        }
    }

    // Start building the response JSON
    std::string response = "{\"text\":\"" + escaped_text + "\"";
    
    // Add current settings if auto_settings is enabled
    if (config_.auto_settings) {
        response += ",\"current_settings\":{" +
            std::string("\"silence_threshold\":") + std::to_string(config_.silence_threshold) + "," +
            "\"min_audio_level\":" + std::to_string(config_.min_audio_level) + "," +
            "\"silence_duration\":" + std::to_string(config_.silence_duration) + "," +
            "\"min_audio_duration\":" + std::to_string(config_.min_audio_duration) + "," +
            "\"avg_speech_duration\":" + std::to_string(noise_stats_.avg_speech_duration) + "," +
            "\"avg_pause_duration\":" + std::to_string(noise_stats_.avg_pause_duration) + "}";
    }
    
    // Close the root object
    response += "}";
    
    return response;
}

void start_websocket_server(
    net::io_context& ioc,
    unsigned short port,
    whisper_context* ctx,
    whisper_full_params params,
    const AudioProcessor::Config& config
) {
    auto const address = net::ip::make_address("0.0.0.0");
    auto endpoint = tcp::endpoint{address, port};

    try {
        std::make_shared<Listener>(ioc, endpoint, ctx, params, config)->run();
        std::cerr << "\nwhisper WebSocket server listening at ws://0.0.0.0:" << port << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to start WebSocket server: " << e.what() << std::endl;
    }
}
