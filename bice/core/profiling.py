import time
from functools import wraps

"""
This submodule is used for profiling our methods.
Simply decorate a method with @profile and it's total execution time will be measured.
Afterwards, have a look at the execution times with Profiler.print_summary().
"""


def profile(method):
    """
    This is a decorator (@profile), that triggers profiling of the execution time of a method
    """
    @wraps(method)
    def do_profile(*args, **kw):
        # if profiling is turned off: do nothing but execute the method
        if not Profiler.is_active():
            return method(*args, **kw)
        # get the full name of the method
        name = method.__qualname__
        # save the MethodProfile that we're coming from
        parent_profile = Profiler._current_profile
        # open the MethodProfile of the current method
        if name not in parent_profile.nested_profiles:
            # if it doesn't exist in the parent, create it
            current_profile = MethodProfile(name)
            parent_profile.nested_profiles[name] = current_profile
        else:
            # else, use the one already stored in the parent
            current_profile = parent_profile.nested_profiles[name]
        # we'll now be in the current method, so the Profiler should know that
        Profiler._current_profile = current_profile
        # execute the method while measuring the execution time
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        # save the execution time to the current profile
        current_profile.execution_time += te - ts
        current_profile.ncalls += 1
        # we're out of the method, reset current profile to parent
        Profiler._current_profile = parent_profile
        # return the result of the method
        return result
    return do_profile


class MethodProfile:
    """
    This class saves the execution times of a method and serves
    as a node in the tree data structure of nested method calls.
    """

    def __init__(self, name: str) -> None:
        # name of the method
        self.name = name
        # list of all the measured execution times
        self.execution_time = 0.
        # the total number of calls
        self.ncalls = 0
        # any nested method's and their profiles
        self.nested_profiles = {}

    def flattened_data(self) -> dict:
        """Drop all the nesting data and give a simple dictionary of each method's execution times"""
        # empty result dict
        data = {}
        # add own execution times to the result dict
        if self.execution_time > 0:
            data[self.name] = self
        # for each nested MethodProfile...
        for _, p in self.nested_profiles.items():
            # ...get their flattened dicts
            for name, ps in p.flattened_data().items():
                # append the data to the result dict
                if name not in data:
                    data[name] = MethodProfile(name)
                data[name].execution_time += ps.execution_time
                data[name].ncalls += ps.ncalls
        # return the dict
        return data

    def print_stats(self, total_time, indentation=0, nested=True, last=False) -> None:
        """Print the stats on this method's profile and all the nested methods recursively"""
        if self.execution_time > 0:
            # calculate stats
            Ncalls = self.ncalls
            T_tot = self.execution_time
            T_rel = T_tot / total_time
            # if nested: generate tree view for name
            if nested:
                corn = "└" if last else "├"
                name = ("│ "*indentation) + \
                    (""+corn+"─" if indentation > 0 else "") + self.name
            else:
                name = self.name
            # pretty print the stats
            print(f"{name:<70} {T_tot:10.3f}s {T_rel:11.2%} {Ncalls:8d}")
            # if nested view: increase indentation for nested methods and use relative total time
            if nested:
                indentation += 1
                total_time = T_tot
        # if we're showing nested calls or if this is the root call
        if nested or self.name == "":
            # if not nested, flatten the tree
            if nested:
                profiles = self.nested_profiles
            else:
                profiles = self.flattened_data()
            # sort the nested profiles by total execution time
            profiles = sorted(profiles.values(), key=lambda item:
                              item.execution_time, reverse=True)
            # print their summary recursively
            for i, p in enumerate(profiles):
                is_last = i == len(profiles) - 1
                p.print_stats(total_time, indentation, nested, last=is_last)


class Profiler:
    """
    The Profiler is a static class that contains the methods
    needed for accessing/controlling the profiling of the code.
    """

    __start_time = 0
    __root_profile = MethodProfile("")
    _current_profile = __root_profile
    execution_times = {}

    @staticmethod
    def start() -> None:
        """(Re)start the Profiler"""
        # reset the execution time dictionary
        Profiler.execution_times = {}
        # reset the start time
        Profiler.__start_time = time.time()

    @staticmethod
    def is_active() -> bool:
        """Is the Profiler active/running?"""
        return Profiler.__start_time is not None

    @staticmethod
    def print_summary(nested=True) -> None:
        """Print a summary on the execution times of the methods that were decorated with @profile"""
        # check if Profiler is active
        if not Profiler.is_active():
            print("Profiler is inactive.")
            return

        # calculate total time
        total_time = time.time() - Profiler.__start_time
        # print the header
        print("Profiler results:")
        print("{:<70} {:>11} {:>11} {:>8}".format(
            "method name", "total", "relative", "#calls"))
        print("-"*103)
        # print the stats of each call recursively, starting with the root call
        Profiler.__root_profile.print_stats(total_time, nested=nested)
