#include "h264_source.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

/* read(2) 的 EINTR 重试包装：返回真实读到的字节数，0 = EOF */
static ssize_t read_retry(int fd, void *dst, size_t n)
{
    for (;;) {
        ssize_t r = ::read(fd, dst, n);
        if (r >= 0 || errno != EINTR)
            return r;
    }
}

/* 有界回收子进程：最多等 ms 毫秒，返回 true = 已回收（或被别人收走）。
 * 用 WNOHANG 轮询而不是阻塞 waitpid —— ffmpeg 卡在内核 D 状态
 * （网络栈里出不来）时 SIGKILL 也要等它返回，阻塞等待会把 stop()/join()
 * 永久挂住，Python 进程再也退不出去。 */
static bool wait_exit_bounded(pid_t pid, int ms)
{
    for (int waited = 0; waited < ms; waited += 10) {
        pid_t r = ::waitpid(pid, nullptr, WNOHANG);
        if (r == pid)
            return true;
        if (r < 0) {
            if (errno == ECHILD)   /* 不是我们的子进程/已被回收 */
                return true;
            if (errno != EINTR)    /* 其它错误：等下去也没用 */
                return false;
        }
        ::usleep(10 * 1000);
    }
    return false;
}

/* ---------------- H264FileSource ---------------- */

H264FileSource::H264FileSource(const std::string &path)
{
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        throw std::runtime_error("打开 " + path + " 失败: " + std::string(strerror(errno)));
}

H264FileSource::~H264FileSource()
{
    if (fd_ >= 0)
        ::close(fd_);
}

size_t H264FileSource::read(uint8_t *dst, size_t n)
{
    return (size_t)read_retry(fd_, dst, n);
}

/* ---------------- RtspH264Source ---------------- */

RtspH264Source::RtspH264Source(const std::string &url)
{
    int pipefd[2];
    if (::pipe(pipefd) != 0)
        throw std::runtime_error("pipe 失败");

    pid_ = ::fork();
    if (pid_ < 0) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        throw std::runtime_error("fork 失败");
    }
    if (pid_ == 0) {
        /* 子进程：stdout 接管道，stderr 继承（ffmpeg 报错直接打到我们终端） */
        ::dup2(pipefd[1], 1);
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        char arg_timeout[] = "10000000";   /* 10s socket IO 超时（微秒）。
                                              ffmpeg 5.0 起 rtsp 的这个参数由
                                              -stimeout 改名为 -timeout；名字写错
                                              ffmpeg 会直接退出（本机是 ffmpeg 6.0） */
        char *argv[] = {
            (char *)"ffmpeg",
            (char *)"-nostdin",
            (char *)"-loglevel",  (char *)"warning",
            (char *)"-rtsp_transport", (char *)"tcp",
            (char *)"-timeout",  arg_timeout,
            (char *)"-i",         (char *)url.c_str(),
            (char *)"-map",       (char *)"0:v:0",   /* 只要第一路视频 */
            (char *)"-c",         (char *)"copy",    /* 不重编码，只拆包 */
            /* RTSP 的 SPS/PPS 是带外的（在 SDP 里），-f h264 这个裸流 muxer
             * 不会把它写进输出，MPP 拿不到参数集就一帧都解不出来。dump_extra
             * 只是把 extradata 里的 SPS/PPS 作为前置 NAL 插回每个关键帧前，
             * 不重编码、不碰像素 */
            (char *)"-bsf:v",     (char *)"dump_extra",
            (char *)"-f",         (char *)"h264",    /* Annex-B 裸流 */
            (char *)"pipe:1",
            nullptr};
        ::execvp("ffmpeg", argv);
        _exit(127);   /* exec 失败才会到这 */
    }
    ::close(pipefd[1]);
    fd_ = pipefd[0];
}

void RtspH264Source::kick()
{
    std::lock_guard<std::mutex> lk(proc_mu_);
    if (pid_ > 0)
        ::kill(pid_, SIGTERM);
}

RtspH264Source::~RtspH264Source()
{
    /* 顺序很重要：先捅子进程，再关读端。反过来若 fd 号被别的线程/新 source
     * 复用，阻塞中的 read() 会读到不相干的数据；而 close 本身并不能保证
     * 解阻塞阻塞在读上的线程，真正解阻塞靠的是子进程退出 → EOF。 */
    std::lock_guard<std::mutex> lk(proc_mu_);
    const pid_t pid = pid_;
    pid_ = -1;   /* 先置 -1：即便后面回收失败，也不会再对可能被复用的 pid 发信号 */

    if (pid > 0)
        ::kill(pid, SIGTERM);
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    if (pid <= 0)
        return;

    /* SIGTERM 最多等 2s，还活着就 SIGKILL，再最多等 1s。全程有界：
     * 最坏 3s 后无论如何返回，绝不把 join() 挂死。 */
    if (!wait_exit_bounded(pid, 2000)) {
        ::kill(pid, SIGKILL);
        if (!wait_exit_bounded(pid, 1000))
            std::fprintf(stderr,
                         "[cpp] ffmpeg(pid %d) SIGKILL 后仍未退出（可能卡在 D 状态），"
                         "放弃回收\n", (int)pid);
    }
}

size_t RtspH264Source::read(uint8_t *dst, size_t n)
{
    ssize_t r = read_retry(fd_, dst, n);
    if (r == 0)
        throw std::runtime_error(
            "RTSP 流结束（ffmpeg 退出）——检查 URL / 网络 / 摄像头是否在线，"
            "ffmpeg 的报错就打在上面的 stderr 里");
    return (size_t)r;
}
