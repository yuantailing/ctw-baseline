# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import six
import subprocess


image = 'ctw-worker:latest'
codalab_tmp = '/tmp/codalab'

def call(args):
    print(*args)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, shell=False)
    stdout, _ = p.communicate()
    p.wait()
    assert 0 == p.returncode
    return stdout.decode()


def main():
    assert os.path.dirname(os.path.abspath(__file__)) == os.path.abspath('.')

    parser = argparse.ArgumentParser()
    parser.add_argument('broker_url', type=six.text_type)
    parser.add_argument('--ctw_root', type=six.text_type, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    args = parser.parse_args()

    # kill running containers
    running = call(['docker', 'ps', '--filter', 'ancestor={}'.format(image), '--format', '{{.ID}}'])
    for container_id in running.strip().splitlines():
        call(['docker', 'stop', container_id])

    # build image
    stdout = call(['docker', 'build', '-t', image, '.'])

    # run container
    if not os.path.isdir(codalab_tmp):
        os.makedirs(codalab_tmp)
    broker_url = args.broker_url.replace('@competitions.codalab.org/', '@competitions.codalab.org:5671/')
    container_id = call([
        'docker', 'run',
        '-v', '/var/run/docker.sock:/var/run/docker.sock',
        '-v', '{0}:{0}'.format(codalab_tmp),
        '--env', 'BROKER_URL={}'.format(broker_url),
        '--env', 'BROKER_USE_SSL=True',
        '--env', 'CTW_ROOT={}'.format(args.ctw_root),
        '-d',
        '--restart', 'unless-stopped',
        image,
    ]).strip()
    print(container_id)


if __name__ == '__main__':
    exit(main())
