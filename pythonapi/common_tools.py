# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import json
import operator
import os
import threading

from six.moves import queue


def synchronized(lock):
    def wrap(f):
        def newFunction(*args, **kw):
            lock.acquire()
            try:
                return f(*args, **kw)
            finally:
                lock.release()
        return newFunction
    return wrap


def to_jsonl(obj):
    return json.dumps(obj, ensure_ascii=True, allow_nan=False, indent=None, sort_keys=True)


def each_file_tuple(root):
    tasks = [(root, '')]
    while tasks:
        p0, r0 = tasks.pop()
        for d1 in os.listdir(p0):
            p1 = os.path.join(p0, d1)
            r1 = os.path.join(r0, d1)
            if os.path.isdir(p1):
                tasks.append((p1, r1))
            elif os.path.isfile(p1):
                yield p1, r1, d1


def exists_and_newer(subj, obj, strict=False):
    assert os.path.exists(obj)
    if not os.path.exists(subj):
        return False

    def newer(a, b):
        return a > b or (not strict and a == b)
    return newer(os.stat(subj).st_mtime, os.stat(obj).st_mtime)


def reduce_sum(*args, **kwargs):
    return functools.reduce(operator.add, *args, **kwargs)


def multithreaded_tid(func, args_list, num_thread, logfunc=None):
    assert 0 < num_thread
    n = len(args_list)
    args_list = [args if isinstance(args, list) or isinstance(args, tuple)
                 else (args, ) for args in args_list]
    q = queue.Queue()
    for i in range(n):
        q.put(i)
    p = queue.Queue()

    def parallel_work(tid):
        while True:
            try:
                i = q.get(block=False)
                p.put(i)
            except queue.Empty as e:
                return
            func(*args_list[i], tid=tid)
    threads = [threading.Thread(target=parallel_work, args=(i, ))
               for i in range(num_thread)]
    for t in threads:
        t.start()
    if logfunc is not None:
        while 0 < n:
            try:
                logfunc(*args_list[p.get(block=True, timeout=1)])
                n -= 1
            except queue.Empty as e:
                pass
    for t in threads:
        t.join()


def multithreaded(func, args_list, num_thread, logfunc=None):
    def foo(*args, **kwargs):
        func(*args)
    return multithreaded_tid(foo, args_list, num_thread, logfunc=logfunc)
