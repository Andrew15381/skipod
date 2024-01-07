#include "mpi.h"
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <array>
namespace fs = std::filesystem;

constexpr int TQ = 20;
constexpr int WQ = 20;
constexpr int RQ = 1;
constexpr int WC = 4;
constexpr int RC = 10;
constexpr int N = 30;
constexpr int Ts = 100;
constexpr int Tb = 1;

constexpr int TAG_REQ = 0;
constexpr int TAG_WRITE = 1;
constexpr int TAG_READ = 2;

const char fspref[] = "fs";
const char servpref[] = "serv";
const char fnpref[] = "file";

const fs::path workdir = fs::current_path();
const fs::path fsdir = workdir / fspref;

int myid;

enum ReqType : int {
    READ,
    WRITE,
    STOP
} read = READ, write = WRITE, stop = STOP;

int main(int argc, char* argv[]) {
    /*
     * Единственный читатель-писатель - процесс 0, поэтому блокировки и чтение перед записью не нужны
     * Все запросы успешные, поэтому можно отправлять не всем
     */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Status status;

    if (myid == 0) {
        int v = 0;
        int time = 0;

        for (int i = 0; i < WC; ++i) {
            for (int p = 1; p <= WQ; ++p) {
                MPI_Send(&write, 1, MPI_INT, p, TAG_REQ, MPI_COMM_WORLD);
                time += Ts + Tb * sizeof(ReqType);
            }

            int cons;
            for (int p = 1; p <= WQ; ++p) {
                MPI_Recv(&v, 1, MPI_INT, p, TAG_WRITE, MPI_COMM_WORLD, &status);
                if (p == 1) {
                    cons = v;
                } else if (cons != v) {
                    printf("Write failed on p=%d, consensus version expected=%d, got=%d\n", p, cons, v);
                    std::fflush(stdout);
                    return -1;
                }
                time += Ts + Tb * sizeof(int);
            }

            auto text = std::string(N, '0' + i);
            printf("Writing text=\"%s\"\n", text.c_str());
            std::fflush(stdout);

            for (int p = 1; p <= WQ; ++p) {
                MPI_Send(text.c_str(), N, MPI_CHAR, p, TAG_WRITE, MPI_COMM_WORLD);
                time += Ts + Tb * N;
            }
        }

        for (int i = 0; i < RC; ++i) {
            std::array<int, RQ> vs;
            for (int p = 1; p <= RQ; ++p) {
                MPI_Send(&read, 1, MPI_INT, p, TAG_REQ, MPI_COMM_WORLD);
                time += Ts + Tb * sizeof(ReqType);
            }

            for (int p = 1; p <= RQ; ++p) {
                MPI_Recv(&v, 1, MPI_INT, p, TAG_READ, MPI_COMM_WORLD, &status);
                time += Ts + Tb * sizeof(int);
                vs[p - 1] = v;
            }

            int maxv = 0, maxp = 1;
            for (int k = 0; k < RQ; ++k) {
                if (vs[k] > maxv) {
                    maxp = k + 1;
                    maxv = vs[k];
                }
            }

            std::printf("Reading version=%d from p=%d\n", maxv, maxp);
            std::fflush(stdout);
            fs::path file_path = fsdir / (servpref + std::to_string(maxp)) / (fnpref + std::to_string(maxv));
            std::fstream file(file_path);

            int length = fs::file_size(file_path);
            char buf[length + 1];
            memset(buf, 0, length + 1);

            file.read(buf, length);
            file.close();

            std::printf("Read text=\"%.*s\"\n", length, buf);
            std::fflush(stdout);
            time += Ts + Tb * N;
        }

        std::printf("Total time=%d\n", time);
        std::fflush(stdout);

        for (int p = 1; p <= TQ; ++p) {
            MPI_Send(&stop, 1, MPI_INT, p, TAG_REQ, MPI_COMM_WORLD);
        }
    } else {
        const std::string server_dir = servpref + std::to_string(myid);
        int v = 0;
        bool poll = true;

        ReqType req;

        std::printf("Server dir created for p=%d\n", myid);
        std::fflush(stdout);
        fs::create_directories(fsdir / server_dir);
        fs::copy_file(workdir / fnpref, fsdir / server_dir / (fnpref + std::string("0")));
        
        while (poll) {
            MPI_Recv(&req, 1, MPI_INT, 0, TAG_REQ, MPI_COMM_WORLD, &status);
            switch (req) {
                case READ: {
                    MPI_Send(&v, 1, MPI_INT, 0, TAG_READ, MPI_COMM_WORLD);
                    break;
                }
                case WRITE: {
                    MPI_Send(&v, 1, MPI_INT, 0, TAG_WRITE, MPI_COMM_WORLD);

                    char buf[N];
                    MPI_Recv(buf, N, MPI_CHAR, 0, TAG_WRITE, MPI_COMM_WORLD, &status);
                    std::string text(buf, N);

                    fs::path fp = fsdir / (servpref + std::to_string(myid)) / (fnpref + std::to_string(v));
                    fs::path newfp = fsdir / (servpref + std::to_string(myid)) / (fnpref + std::to_string(v + 1));
                    fs::copy(fp, newfp);

                    std::fstream file(newfp, std::ios::app);
                    file << text;
                    file.close();

                    v += 1;
                    break;
                }
                case STOP: {
                    poll = false;
                    break;
                }
            }
        }
    }

    MPI_Finalize();
}
