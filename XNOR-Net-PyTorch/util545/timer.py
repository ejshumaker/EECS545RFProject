# timer.py

# A class to perform timing

import time


class Timer():
    '''
    A class to perform timing.
    Takes 'desc', a string to be printed whenever end_time() is called and 'printflag' is true
    'printflag' states whether or not this timer should print the time.

    methods:
    - start_time(): record starting time
    - end_time(): print and save elapsed time
    - get_saved_times(): return a list of all saved times
    '''
    def __init__(self, desc="", printflag=True):
        self.prev_start_time = 0
        self.desc = desc
        self.printflag = printflag
        self.saved_times = []

    # Call this function to reset the timer
    def start_time(self):
        self.prev_start_time = time.time()

    # Print time since start_time() was called
    def end_time(self):
        elapsed_time = time.time() - self.prev_start_time
        self.saved_times.append(elapsed_time)

        # Print elapsed time if printing is enabled
        if self.printflag:
            print("%s: %.06f s" % (self.desc, elapsed_time))

    # Return saved times
    def get_saved_times(self):
        return self.saved_times
