#include <mpi.h>
#include <cstring>
#include <iostream>
#include <optional>

#include "2d.h"

enum class Mode {
    TwoDim,
    ThreeDim,
    Balanced,
};

struct Config {
    bool print_result = false;
    std::optional<double> g_value;
    const char *path_a = NULL, *path_b = NULL;
    std::optional<Mode> mode;
    int layers = 0;
};

Config parse_args(int argc, char *argv[]) {
    Config conf;
    for (int i = 1; i < argc; i++) {
        auto following_arg = [&](const char *flag) {
            if (++i == argc) {
                std::cerr << "error: missing argument after flag '" << flag << "'\n";
                exit(1);
            }
            return argv[i];
        };

        if (strcmp(argv[i], "-a") == 0) {
            conf.path_a = following_arg("-a");
        } else if (strcmp(argv[i], "-b") == 0) {
            conf.path_b = following_arg("-b");
        } else if (strcmp(argv[i], "-v") == 0) {
            conf.print_result = true;
        } else if (strcmp(argv[i], "-g") == 0) {
            conf.g_value = std::atoi(following_arg("-g"));
        } else if (strcmp(argv[i], "-t") == 0) {
            auto mode = following_arg("-t");
            if (strcmp(mode, "2D") == 0) {
                conf.mode = Mode::TwoDim;
            } else if (strcmp(mode, "3D") == 0) {
                conf.mode = Mode::ThreeDim;
            } else if (strcmp(mode, "balanced") == 0) {
                conf.mode = Mode::Balanced;
            } else {
                std::cerr << "error: unknown mode '" << mode << "'\n";
                exit(1);
            }
        } else if (strcmp(argv[i], "-l") == 0) {
            conf.layers = std::atoi(following_arg("-l"));
        } else {
            std::cerr << "error: unknown argument '" << argv[i] << "'\n";
            exit(1);
        }
    }

    return conf;
}

int main(int argc, char *argv[]) {
    Config conf = parse_args(argc, argv);

    if (!conf.path_a && !conf.path_b) {
        std::cerr << "error: matrices are not provided\n";
        return 1;
    }

    if (!conf.mode) {
        /* Use 2D by default */
        std::cerr << "warning: no mode was provided, using 2D\n";
        conf.mode = Mode::TwoDim;
    }

    MPI_Init(&argc, &argv);

    switch (*conf.mode) {
        case Mode::TwoDim:
            summa2d(conf.path_a, conf.path_b, conf.print_result, conf.g_value);
            break;

        default:
            std::cerr << "error: mode unimplemented\n";
    }

    MPI_Finalize();

    return 0;
}