import threading
from threading import Thread
from multiprocessing import Process
from multiprocessing import Manager

class ThreadingUtils:

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    @staticmethod
    def run(function, input_list, n_cores, log_each=None, log=False):
        print(('run function %s with n_cores = %i' % (function, n_cores)))
        print(function)
        # print 'with input list of len'
        # print len(input_list)
        # print 'in groups of %d threads' % n_threads

        assert n_cores <= 20

        # the type of input_list has to be a list. If not
        # then it can a single element list and we cast it to list.
        if not isinstance(type(input_list[0]), type(list)):
            input_list = [[i] for i in input_list]

        n_groups = int(len(input_list) / n_cores + 1)
        # print 'n groups', n_groups

        n_done = 0
        for group_i in range(n_groups):
            start, end = group_i * n_cores, (group_i + 1) * n_cores
            # print 'start', start, 'end', end

            threads = [None] * (end - start)
            for i, pi in enumerate(range(start, min(end, len(input_list)))):
                next_args = input_list[pi]
                if log:
                    print(next_args)
                # print next_kmer
                threads[i] = Process(target=function, args=next_args)
                # print 'starting process #', i
                threads[i].start()

            # print  threads
            # print 'joining threads...'
            # do some other stuff
            for i in range(len(threads)):
                if threads[i] is None:
                    continue
                threads[i].join()

                n_done += 1
                if log_each is not None and log_each % n_done == 0:
                    print('Done %i so far' % n_done)
        print('done...')

