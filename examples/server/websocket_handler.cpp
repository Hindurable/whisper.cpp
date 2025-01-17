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
        fprintf(stderr, "Client connected to WebSocket server\n");
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
    fprintf(stderr, "Client disconnected from WebSocket server [%s]\n", 
            client_endpoint_.address().to_string().c_str());
}

void WebSocketSession::run() {
    ws_.set_option(websocket::stream_base::decorator(
        [](websocket::response_type& res) {
            res.set(beast::http::field::server, "Whisper WebSocket Server");
        }));

    fprintf(stderr, "New WebSocket connection from %s\n", 
            client_endpoint_.address().to_string().c_str());

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
    if (message.size() > 2 && message[0] == '{') {
        // Parse audio format
        if (message.find("\"format\":\"pcm16\"") != std::string::npos) {
            format_ = AudioFormat::PCM_INT16;
        } else if (message.find("\"format\":\"float32\"") != std::string::npos) {
            format_ = AudioFormat::PCM_FLOAT32;
        }

        // Parse silence threshold
        size_t pos = message.find("\"silence_threshold\":");
        if (pos != std::string::npos) {
            pos += 19; // Length of "silence_threshold":
            config_.silence_threshold = std::stof(message.substr(pos));
        }

        // Parse minimum audio level
        pos = message.find("\"min_audio_level\":");
        if (pos != std::string::npos) {
            pos += 17; // Length of "min_audio_level":
            config_.min_audio_level = std::stof(message.substr(pos));
        }

        // Parse silence duration
        pos = message.find("\"silence_duration\":");
        if (pos != std::string::npos) {
            pos += 18; // Length of "silence_duration":
            config_.silence_duration = std::stof(message.substr(pos));
        }

        // Parse minimum audio duration
        pos = message.find("\"min_audio_duration\":");
        if (pos != std::string::npos) {
            pos += 20; // Length of "min_audio_duration":
            config_.min_audio_duration = std::stof(message.substr(pos));
        }

        return true;
    }
    return false;
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
    if (buffer_.size() >= config_.min_audio_duration * 16000) { // Convert seconds to samples
        // Check if the last portion is silence
        size_t silence_samples = static_cast<size_t>(config_.silence_duration * 16000);
        std::vector<float> last_samples(buffer_.end() - silence_samples, buffer_.end());
        
        if (detect_silence(last_samples)) {
            // Check if the entire buffer is too quiet
            if (is_too_quiet(buffer_)) {
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
                std::cerr << "Failed to send response: " << ec.message() << std::endl;
            }

            // Clear buffer
            buffer_.clear();
        }
    }
}

std::string AudioProcessor::get_transcription() {
    const int n_samples = buffer_.size();
    if (n_samples == 0) {
        return "{\"error\":\"no audio data\"}";
    }

    int result = whisper_full(ctx_, params_, buffer_.data(), n_samples);
    if (result != 0) {  // whisper_full returns 0 on success
        return "{\"error\":\"failed to process audio\"}";
    }

    const int n_segments = whisper_full_n_segments(ctx_);
    if (n_segments <= 0) {
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
        fprintf(stderr, "\nwhisper WebSocket server listening at ws://0.0.0.0:%d\n", port);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to start WebSocket server: %s\n", e.what());
    }
}
