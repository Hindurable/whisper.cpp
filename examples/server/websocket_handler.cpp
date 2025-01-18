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

// AudioProcessor implementation
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
        // Basic JSON validation
        if (message.empty() || message[0] != '{' || message.back() != '}') {
            return false;
        }

        auto extract_value = [](const std::string& msg, const std::string& key) -> std::pair<bool, std::string> {
            std::string search = "\"" + key + "\":";
            size_t start = msg.find(search);
            if (start == std::string::npos) {
                return {false, ""};
            }
            start += search.length();
            
            // Skip whitespace
            while (start < msg.length() && std::isspace(msg[start])) {
                start++;
            }

            // Handle string values
            if (start < msg.length() && msg[start] == '"') {
                start++;
                size_t end = msg.find('"', start);
                if (end == std::string::npos) return {false, ""};
                return {true, msg.substr(start, end - start)};
            }
            
            // Handle numeric values
            size_t end = start;
            while (end < msg.length() && (std::isdigit(msg[end]) || msg[end] == '.' || msg[end] == '-')) {
                end++;
            }
            if (end > start) {
                return {true, msg.substr(start, end - start)};
            }
            
            return {false, ""};
        };

        // Parse format
        auto [format_found, format] = extract_value(message, "format");
        if (!format_found) {
            return false;
        }

        if (format == "pcm16") {
            format_ = AudioFormat::PCM_INT16;
        } else if (format == "float32") {
            format_ = AudioFormat::PCM_FLOAT32;
        }

        // Parse config values
        auto parse_float = [](const std::pair<bool, std::string>& result, float& target) {
            if (result.first) {
                try {
                    target = std::stof(result.second);
                } catch (...) {
                    // Keep existing value if parsing fails
                }
            }
        };

        parse_float(extract_value(message, "silence_threshold"), config_.silence_threshold);
        parse_float(extract_value(message, "min_audio_level"), config_.min_audio_level);
        parse_float(extract_value(message, "silence_duration"), config_.silence_duration);
        parse_float(extract_value(message, "min_audio_duration"), config_.min_audio_duration);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing config: " << e.what() << std::endl;
        return false;
    }
}

void AudioProcessor::process_binary(const std::string& message, websocket::stream<tcp::socket>& ws) {
    // Check if this is a configuration message (JSON)
    if (parse_config_message(message)) {
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
            "}}";
        ws.write(boost::asio::buffer(response), ec);
        std::cerr << "Received client settings: " << message << std::endl;
        std::cerr << "Sent acknowledgment: " << response << std::endl;
        return;
    }

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
        
        // Ensure we have enough samples for silence detection
        if (buffer_.size() < silence_samples) {
            return;
        }
        
        std::vector<float> last_samples(buffer_.end() - silence_samples, buffer_.end());
        
        float last_rms = calculate_rms(last_samples);
        float buffer_rms = calculate_rms(buffer_);
        
        std::cerr << "DEBUG - Buffer: " << buffer_.size() / 16000.0f << "s (" << buffer_rms << " dB), Last " 
                  << silence_samples / 16000.0f << "s: " << last_rms 
                  << " dB (threshold: " << config_.silence_threshold << ")" << std::endl;
        
        if (detect_silence(last_samples)) {
            std::cerr << "DEBUG - Silence detected!" << std::endl;
            
            // Check if the entire buffer is too quiet
            if (is_too_quiet(buffer_)) {
                std::cerr << "DEBUG - Buffer too quiet (" << buffer_rms << " dB < " << config_.min_audio_level << " dB), discarding" << std::endl;
                buffer_.clear();
                return;
            }
            
            // Get transcription
            std::string response = get_transcription();
            
            // Send response
            ws.text(true);
            beast::error_code ec;
            ws.write(boost::asio::buffer(response), ec);
            if (ec) {
                std::cerr << "Failed to send transcription: " << ec.message() << std::endl;
            } else {
                std::cerr << "Sent transcription: " << response << std::endl;
            }

            // Clear buffer
            buffer_.clear();
        }
    }
}

std::string AudioProcessor::get_transcription() {
    const int n_samples = buffer_.size();
    if (n_samples == 0) {
        std::cerr << "DEBUG - No audio data for transcription" << std::endl;
        return "{\"error\":\"no audio data\"}";
    }

    int result = whisper_full(ctx_, params_, buffer_.data(), n_samples);
    if (result != 0) {  // whisper_full returns 0 on success
        std::cerr << "DEBUG - Whisper processing failed with code: " << result << std::endl;
        return "{\"error\":\"failed to process audio\"}";
    }

    const int n_segments = whisper_full_n_segments(ctx_);
    if (n_segments <= 0) {
        std::cerr << "DEBUG - No segments in transcription" << std::endl;
        return "{\"error\":\"no transcription\"}";
    }

    std::string text;
    for (int i = 0; i < n_segments; ++i) {
        const char* segment_text = whisper_full_get_segment_text(ctx_, i);
        text += segment_text;
    }

    // Escape JSON special characters
    std::string escaped_text;
    for (char c : text) {
        switch (c) {
            case '"': escaped_text += "\\\""; break;
            case '\\': escaped_text += "\\\\"; break;
            case '\b': escaped_text += "\\b"; break;
            case '\f': escaped_text += "\\f"; break;
            case '\n': escaped_text += "\\n"; break;
            case '\r': escaped_text += "\\r"; break;
            case '\t': escaped_text += "\\t"; break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    escaped_text += buf;
                } else {
                    escaped_text += c;
                }
        }
    }

    return "{\"text\":\"" + escaped_text + "\"}";
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
