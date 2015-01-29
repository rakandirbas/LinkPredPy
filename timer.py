import time

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.strt = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.strt
        self.msecs = self.secs * 1000  # millisecs
        self.min = self.secs / 60
        if self.verbose:
            print 'elapsed time: %f ms, %f s, %f m.' % (self.msecs, self.secs, self.min)

class Timerx():
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def start(self):
        self.strt = time.time()
        return self
    
    def stop(self):
        self.end = time.time()
        self.secs = self.end - self.strt
        self.msecs = self.secs * 1000  # millisecs
        self.min = self.secs / 60
        if self.verbose:
            print 'elapsed time: %f ms, %f s, %f m.' % (self.msecs, self.secs, self.min)
            
        return "Secs: ", self.secs, ", mSecs: ", self.msecs, ", Min: ", self.min

# 
# with Timer() as t:
#     print ("foo", "bar")
# print "=> elasped lpush: %s s" % t.secs



