# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import threading

from six.moves import queue


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


def multithreaded_tid(func, args_list, num_thread):
    assert 0 < num_thread
    q = queue.Queue()
    for args in args_list:
        q.put(args if isinstance(args, list) or isinstance(args, tuple)
              else (args, ))

    def parallel_work(tid):
        while True:
            try:
                t = q.get(block=False)
            except queue.Empty as e:
                return
            func(*t, tid=tid)
    threads = [threading.Thread(target=parallel_work, args=(i, ))
               for i in range(num_thread)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def multithreaded(func, args_list, num_thread):
    def foo(*args, **kwargs):
        func(*args)
    return multithreaded_tid(foo, args_list, num_thread)
