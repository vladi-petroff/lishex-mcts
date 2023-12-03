#ifndef BENCH_H_
#define BENCH_H_

#include "types.h"
#include "board.h"
#include "threads.h"

void bench(std::thread &search_thread, board_t *board, searchinfo_t *info);

#endif // BENCH_H_
