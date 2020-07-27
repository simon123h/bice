import time
import numpy as np

"""
This submodule is used for profiling our methods.
Simply decorate a method with @profile and it's total execution time will be measured.
Afterwards, have a look at the execution times with Profiler.print_summary().
"""


def profile(method):
    """
    This is a decorator (@profile), that triggers profiling of the execution time of a method
    """

    def do_profile(*args, **kw):
        # if profiling is turned off: do nothing but execute the method
        if not Profiler.is_active():
            return method(*args, **kw)
        # measure time taken for execution of the method
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        # store execution time in dictionary
        name = method.__qualname__
        if name not in Profiler.execution_times:
            Profiler.execution_times[name] = []
        Profiler.execution_times[name].append(te - ts)
        # return the result of the method
        return result
    return do_profile


class Profiler:
    """
    The Profiler is a static class that contains the methods
    needed for accessing/controlling the profiling of the code.
    """

    __start_time = None
    execution_times = {}

    # (Re)start the Profiler
    @staticmethod
    def start():
        # reset the execution time dictionary
        Profiler.execution_times = {}
        # reset the start time
        Profiler.__start_time = time.time()

    @staticmethod
    # Is the Profiler active/running?
    def is_active():
        return Profiler.__start_time is not None

    @staticmethod
    # Print a summary on the execution times of the methods that were decorated with @profile
    def print_summary():
        # check if Profiler is active
        if not Profiler.is_active():
            print("Profiler is inactive.")
            return
        # check if any profiled methods were executed
        if len(Profiler.execution_times) == 0:
            print("No profiled methods were executed.")
            return
        # calculate total time
        total_time = time.time() - Profiler.__start_time
        # sort methods by execution time (descending)
        sorted_method_stats = {k: v for k, v in sorted(
            Profiler.execution_times.items(), key=lambda item: sum(item[1]), reverse=True)}
        # determine the length of the name column
        column_len = max([len(k) for k in sorted_method_stats])
        # print the stats
        print("Profiler results:")
        print("{} {:>11} {:>11}".format(
            "method name".ljust(column_len), "total", "relative"))
        print("-"*60)
        for k in sorted_method_stats:
            # calculate stats
            data = Profiler.execution_times[k]
            T_tot = sum(data)
            T_rel = T_tot / total_time
            T_rel_stddev = np.std(data) / total_time
            # give times in seconds (absolute and relative)
            print("{:s} {:10.3f}s {:11.3%}  ±{:.2%}".format(
                k.ljust(column_len), T_tot, T_rel, T_rel_stddev))
