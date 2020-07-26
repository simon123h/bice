import time

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
        # store cumulative execution time in dictionary
        name = method.__qualname__
        if name not in Profiler.execution_times:
            Profiler.execution_times[name] = 0
        Profiler.execution_times[name] += te - ts
        # return the result of the method
        return result
    return do_profile


class Profiler:

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
    # give a summary on the execution times of the methods that were decorated with @profile
    def print_summary(absolute_time=False):
        total_time = time.time() - Profiler.__start_time
        print("Time used in methods:")
        sorted_method_stats = {k: v for k, v in sorted(
            Profiler.execution_times.items(), key=lambda item: item[1], reverse=True)}
        for k in sorted_method_stats:
            if absolute_time:
                print(" {:s} :  {:.3}s".format(
                    k.ljust(30), Profiler.execution_times[k]))
            else:
                print(" {:s} :  {:.4%}".format(
                    k.ljust(30), Profiler.execution_times[k]/total_time))
