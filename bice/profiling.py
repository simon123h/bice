import time


start_time = None
time_stats = {}


def start_profiling():
    global start_time
    start_time = time.time()

def profile(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        # store cumulative time in dictionary
        name = method.__qualname__
        if name not in time_stats:
            time_stats[name] = 0
        time_stats[name] += te - ts
        return result
    return timed


def print_profiling_summary():
    global start_time
    total_time = time.time() - start_time
    print("Time used in methods:")
    sorted_stats = {k: v for k, v in sorted(
        time_stats.items(), key=lambda item: item[1], reverse=True)}
    for k in sorted_stats:
        print(" {} :  {}".format(k.ljust(30), time_stats[k]/total_time))
