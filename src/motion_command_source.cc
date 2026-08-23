#include "motion_command_source.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

namespace
{

    float EnvFloat(const char *name, float default_value)
    {
        const char *text = std::getenv(name);
        if (!text || text[0] == '\0')
        {
            return default_value;
        }
        return std::strtof(text, nullptr);
    }

    int EnvInt(const char *name, int default_value)
    {
        const char *text = std::getenv(name);
        if (!text || text[0] == '\0')
        {
            return default_value;
        }
        return std::atoi(text);
    }

    std::string EnvString(const char *name, const char *default_value)
    {
        const char *text = std::getenv(name);
        return (text && text[0] != '\0') ? std::string(text) : std::string(default_value);
    }

    /// 学習時の指令は 0 / ±1 の離散値だけだった。制御側でも念のため落としておく。
    float Quantize(float value, float dead_zone) noexcept
    {
        if (value > dead_zone) return 1.0f;
        if (value < -dead_zone) return -1.0f;
        return 0.0f;
    }

    // ---------------------------------------------------------------------
    /// 環境変数で固定。既定。再現性のある評価に使う。
    class EnvCommandSource : public MotionCommandSource
    {
    public:
        EnvCommandSource()
        {
            command_ = {
                EnvFloat("BIG_RABBIT_MOTION_COMMAND_X", 0.0f),
                EnvFloat("BIG_RABBIT_MOTION_COMMAND_Y", 0.0f),
                EnvFloat("BIG_RABBIT_MOTION_COMMAND_YAW", 0.0f),
            };
        }
        void Poll(double) override {}
        std::array<float, 3> command() const override { return command_; }
        const char *name() const override { return "env"; }
        std::string status() const override
        {
            std::ostringstream text;
            text << "command = (" << command_[0] << ", " << command_[1] << ", " << command_[2] << ")";
            return text.str();
        }

    private:
        std::array<float, 3> command_{};
    };

    // ---------------------------------------------------------------------
    /// UDP 受信。`tools/gamepad_command.py` が送る 1 行 ASCII を読む。
    ///
    ///   CMD,<seq>,<x>,<y>,<yaw>*<XX>\n
    ///
    /// **watchdog は制御側の責務**。送信が止まったら指令をゼロへ落とす。
    /// 指令ゼロは参照振幅 0 = crouch に退化するので「通信が切れたら立ち止まる」が
    /// 構造的に保証される。
    class UdpCommandSource : public MotionCommandSource
    {
    public:
        UdpCommandSource()
        {
            port_ = EnvInt("BIG_RABBIT_COMMAND_PORT", 51234);
            timeout_s_ = EnvFloat("BIG_RABBIT_COMMAND_TIMEOUT_S", 0.2f);
            quantize_ = EnvInt("BIG_RABBIT_COMMAND_QUANTIZE", 1) != 0;
            dead_zone_ = EnvFloat("BIG_RABBIT_COMMAND_DEAD_ZONE", 0.4f);

            socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
            if (socket_ < 0)
            {
                status_ = "socket を開けなかった";
                return;
            }
            // SO_REUSEADDR は**付けない**。付けると同じポートを 2 プロセスが bind でき、
            // 指令フレームをどちらが受けるか不定になる（実際にこれで「効かない」現象が出た）。
            // 制御入力なので、二重起動は bind 失敗として明示的に落とす方が安全。
            // ノンブロッキングにして、制御ループを止めないようにする。
            const int flags = ::fcntl(socket_, F_GETFL, 0);
            ::fcntl(socket_, F_SETFL, flags | O_NONBLOCK);

            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_addr.s_addr = htonl(INADDR_ANY);
            address.sin_port = htons(static_cast<uint16_t>(port_));
            if (::bind(socket_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0)
            {
                std::cerr << "[FATAL] udp *:" << port_ << " を bind できない。\n"
                          << "        同じポートを使う sim が既に動いていないか確認する:\n"
                          << "          pgrep -af big_rabbit_mujoco_sim\n"
                          << "        別ポートを使うなら BIG_RABBIT_COMMAND_PORT を変える。" << std::endl;
                status_ = "bind に失敗した（ポートが使用中）";
                ::close(socket_);
                socket_ = -1;
                bind_failed_ = true;
                return;
            }

            std::ostringstream text;
            text << "udp *:" << port_ << " 受信待ち  timeout " << timeout_s_ << " s  "
                 << (quantize_ ? "量子化 on" : "アナログ");
            status_ = text.str();
        }

        ~UdpCommandSource() override
        {
            if (socket_ >= 0)
            {
                ::close(socket_);
            }
        }

        void Poll(double time_s) override
        {
            if (socket_ < 0)
            {
                // bind に失敗しているので指令は受け取れない。指令ゼロ = crouch のまま。
                command_ = {0.0f, 0.0f, 0.0f};
                return;
            }

            // 溜まっている分は読み切って**最後のフレームだけ採用**する。
            // 古い指令で動くと操作感が破綻するため。
            char buffer[256];
            bool got_valid = false;
            std::array<float, 3> latest{};
            while (true)
            {
                const ssize_t length = ::recv(socket_, buffer, sizeof(buffer) - 1, 0);
                if (length <= 0)
                {
                    break;
                }
                buffer[length] = '\0';
                std::array<float, 3> parsed{};
                int seq = 0;
                if (Parse(buffer, seq, parsed))
                {
                    latest = parsed;
                    got_valid = true;
                    if (seq != (last_seq_ + 1) % 65536 && received_ > 0)
                    {
                        dropped_++;
                    }
                    last_seq_ = seq;
                    received_++;
                }
                else
                {
                    rejected_++;
                }
            }

            if (got_valid)
            {
                if (received_first_ == false)
                {
                    received_first_ = true;
                    std::cout << "[command] 受信開始（udp *:" << port_ << "）" << std::endl;
                }
                last_valid_time_s_ = time_s;
                command_ = latest;
                if (quantize_)
                {
                    for (auto &value : command_)
                    {
                        value = Quantize(value, dead_zone_);
                    }
                }
            }
            else if (time_s - last_valid_time_s_ > timeout_s_)
            {
                // watchdog。指令ゼロ = crouch へ退化する。
                if (command_ != std::array<float, 3>{0.0f, 0.0f, 0.0f})
                {
                    std::cout << "[command] timeout (" << timeout_s_ << " s) -> 指令ゼロ" << std::endl;
                }
                command_ = {0.0f, 0.0f, 0.0f};
            }

            // 送信側が起動していないことに気づけるよう、一度だけ警告する。
            // --monitor は表示専用で送信しないので、ここで取り違えやすい。
            if (!received_first_ && !warned_ && time_s > 2.0)
            {
                warned_ = true;
                std::cout << "[command] 警告: " << time_s << " s 経っても指令フレームが来ていない。\n"
                          << "          送信側を起動する: python3 tools/gamepad_command.py\n"
                          << "          （--monitor は表示専用で送信しない）" << std::endl;
            }

            // 指令が変わったら出す。効いているかを目で確認できるようにする。
            if (command_ != last_reported_)
            {
                std::cout << "[command] (" << command_[0] << ", " << command_[1] << ", "
                          << command_[2] << ")" << std::endl;
                last_reported_ = command_;
            }
        }

        std::array<float, 3> command() const override { return command_; }
        const char *name() const override { return "udp"; }
        std::string status() const override { return status_; }

    private:
        /// "CMD,<seq>,<x>,<y>,<yaw>*<XX>" を解釈する。チェックサム不一致は捨てる。
        static bool Parse(const char *line, int &seq, std::array<float, 3> &command)
        {
            const char *star = std::strchr(line, '*');
            if (!star || std::strncmp(line, "CMD,", 4) != 0)
            {
                return false;
            }
            unsigned int expected = 0;
            if (std::sscanf(star + 1, "%2x", &expected) != 1)
            {
                return false;
            }
            unsigned int actual = 0;
            for (const char *cursor = line; cursor < star; ++cursor)
            {
                actual ^= static_cast<unsigned char>(*cursor);
            }
            if ((actual & 0xFF) != (expected & 0xFF))
            {
                return false;
            }
            float x = 0.0f;
            float y = 0.0f;
            float yaw = 0.0f;
            if (std::sscanf(line, "CMD,%d,%f,%f,%f", &seq, &x, &y, &yaw) != 4)
            {
                return false;
            }
            command = {
                std::clamp(x, -1.0f, 1.0f),
                std::clamp(y, -1.0f, 1.0f),
                std::clamp(yaw, -1.0f, 1.0f),
            };
            return true;
        }

        int socket_ = -1;
        int port_ = 51234;
        float timeout_s_ = 0.2f;
        bool quantize_ = true;
        float dead_zone_ = 0.4f;
        std::array<float, 3> command_{};
        double last_valid_time_s_ = -1.0e9;
        bool received_first_ = false;
        bool bind_failed_ = false;
        bool warned_ = false;
        std::array<float, 3> last_reported_{};
        int last_seq_ = -1;
        long long received_ = 0;
        long long dropped_ = 0;
        long long rejected_ = 0;
        std::string status_;
    };

    // ---------------------------------------------------------------------
    /// 時刻で切り替わる指令列。指令切替の再現テストに使う。
    /// 書式は play_big_rabbit_command_sequence.py と同じ "forward:4,stop:2,..."。
    class SequenceCommandSource : public MotionCommandSource
    {
    public:
        SequenceCommandSource()
        {
            const std::string text = EnvString(
                "BIG_RABBIT_COMMAND_SEQUENCE",
                "forward:4,stop:2,backward:4,stop:2,turnleft:4,stop:2,turnright:4,stop:2");
            std::istringstream stream(text);
            std::string token;
            double total = 0.0;
            while (std::getline(stream, token, ','))
            {
                const std::size_t colon = token.rfind(':');
                if (colon == std::string::npos)
                {
                    continue;
                }
                const std::string label = token.substr(0, colon);
                const double seconds = std::atof(token.substr(colon + 1).c_str());
                std::array<float, 3> command{0.0f, 0.0f, 0.0f};
                if (label == "forward") command = {1.0f, 0.0f, 0.0f};
                else if (label == "backward") command = {-1.0f, 0.0f, 0.0f};
                else if (label == "turnleft") command = {0.0f, 0.0f, 1.0f};
                else if (label == "turnright") command = {0.0f, 0.0f, -1.0f};
                else if (label != "stop") continue;
                total += seconds;
                segments_.push_back({label, command, total});
            }
            std::ostringstream status;
            status << segments_.size() << " 区間 / 合計 " << total << " s";
            status_ = status.str();
        }

        void Poll(double time_s) override
        {
            for (const auto &segment : segments_)
            {
                if (time_s < segment.end_s)
                {
                    if (segment.label != current_label_)
                    {
                        std::cout << "[command] t=" << time_s << " -> " << segment.label << std::endl;
                        current_label_ = segment.label;
                    }
                    command_ = segment.command;
                    return;
                }
            }
            command_ = {0.0f, 0.0f, 0.0f};
        }

        std::array<float, 3> command() const override { return command_; }
        const char *name() const override { return "sequence"; }
        std::string status() const override { return status_; }

    private:
        struct Segment
        {
            std::string label;
            std::array<float, 3> command;
            double end_s;
        };
        std::vector<Segment> segments_;
        std::array<float, 3> command_{};
        std::string current_label_;
        std::string status_;
    };

}  // namespace

std::unique_ptr<MotionCommandSource> MakeMotionCommandSourceFromEnv()
{
    const std::string source = EnvString("BIG_RABBIT_COMMAND_SOURCE", "env");
    if (source == "udp")
    {
        return std::make_unique<UdpCommandSource>();
    }
    if (source == "sequence")
    {
        return std::make_unique<SequenceCommandSource>();
    }
    if (source != "env")
    {
        std::cerr << "[WARN] 未知の BIG_RABBIT_COMMAND_SOURCE=" << source << "。env を使う" << std::endl;
    }
    return std::make_unique<EnvCommandSource>();
}
