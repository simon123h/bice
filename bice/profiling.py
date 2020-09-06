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
        # get the full name of the method
        name = method.__qualname__
        # save the MethodProfile that we're coming from
        parent_profile = Profiler.current_profile
        # open the MethodProfile of the current method
        if name not in parent_profile.nested_profiles:
            # if it doesn't exist in the parent, create it
            current_profile = MethodProfile(name)
            parent_profile.nested_profiles[name] = current_profile
        else:
            # else, use the one already stored in the parent
            current_profile = parent_profile.nested_profiles[name]
        # we'll now be in the current method, so the Profiler should know that
        Profiler.current_profile = current_profile
        # execute the method while measuring the execution time
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        # save the execution time to the current profile
        current_profile.execution_times.append(te - ts)
        # we're out of the method, reset current profile to parent
        Profiler.current_profile = parent_profile
        # return the result of the method
        return result
    return do_profile


class MethodProfile:
    """
    This class saves the execution times of a method and serves
    as a node in the tree data structure of nested method calls.
    """

    def __init__(self, name):
        # name of the method
        self.name = name
        # list of all the measured execution times
        self.execution_times = []
        # any nested method's and their profiles
        self.nested_profiles = {}

    # drop all the nesting data and give a simple dictionary of each method's execution times
    def flattened_data(self):
        # empty result dict
        data = {}
        # add own execution times to the result dict
        if len(self.execution_times) > 0:
            data[self.name] = self
        # for each nested MethodProfile...
        for _, p in self.nested_profiles.items():
            # ...get their flattened dicts
            for name, ps in p.flattened_data().items():
                # append the data to the result dict
                if name not in data:
                    data[name] = MethodProfile(name)
                data[name].execution_times += ps.execution_times
        # return the dict
        return data

    # print the stats on this method's profile and all the nested methods recursively
    def print_stats(self, total_time, indentation=0, nested=True):
        if len(self.execution_times) > 0:
            # calculate stats
            Ncalls = len(self.execution_times)
            T_tot = sum(self.execution_times)
            T_rel = T_tot / total_time
            T_rel_stddev = np.std(self.execution_times) / total_time
            # if nested: generate tree view for name
            if nested:
                name = ("│ "*indentation) + \
                    ("└─" if indentation > 0 else "") + self.name
            else:
                name = self.name
            # pretty print the stats
            print("{:<70} {:10.3f}s {:11.2%}  ±{:6.2%} {:8d}".format(
                name, T_tot, T_rel, T_rel_stddev, Ncalls))
            # if nested view: increase indentation for nested methods and use relative total time
            if nested:
                indentation += 1
                total_time = T_tot
        # if we're showing nested calls or if this is the root call
        if nested or self.name == "":
            # if not nested, flatten the tree
            if nested:
                profs = self.nested_profiles
            else:
                profs = self.flattened_data()
            # sort the nested profiles by total execution time
            profs = {k: v for k, v in sorted(
                profs.items(), key=lambda item: sum(item[1].execution_times), reverse=True)}
            # print their summary recursively
            for _, p in profs.items():
                p.print_stats(total_time, indentation, nested)


class Profiler:
    """
    The Profiler is a static class that contains the methods
    needed for accessing/controlling the profiling of the code.
    """

    __start_time = None
    __root_profile = MethodProfile("")
    current_profile = __root_profile

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
    def print_summary(nested=True):
        # check if Profiler is active
        if not Profiler.is_active():
            print("Profiler is inactive.")
            return

        # calculate total time
        total_time = time.time() - Profiler.__start_time
        # print the header
        print("Profiler results:")
        print("{:<70} {:>11} {:>11} {:>17}".format(
            "method name", "total", "relative", "#calls"))
        print("-"*112)
        # print the stats of each call recursively, starting with the root call
        Profiler.__root_profile.print_stats(total_time, nested=nested)
