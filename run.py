#!/usr/bin/env python3
# meant to be used as a script, so if we want to remove the .py extension at some point, this is needed

import argparse

TRED = '\033[31m'  # Green Text
TGREEN = '\033[32m'  # Green Text
ENDC = '\033[m'  # reset to the defaults


def col_print(col, *args, get_lambda=False, sep=" ", end="\n"):
    orig_args = args
    endy = end

    def printy(*args):
        print(col, end="")
        for arg in orig_args:
            print(str(arg) + sep, end="")
        for arg in args:
            print(str(arg) + sep, end="")
        print(ENDC, end=endy)

    if get_lambda:
        return printy
    else:
        printy()

com_print = col_print(TGREEN, get_lambda=True, sep="", end=" ")

def help():
    def chain(*args):
        com_names = args[0:len(args) - 1]
        descr = args[len(args) - 1]
        print("\t", end="")
        if len(com_names) == 1:
            com_print(com_names[0])
        else:
            for i in range(len(com_names) - 1):
                com_print(com_names[i])
                print("or ", end="")
            com_print(com_names[len(com_names) - 1])
        print(descr)

    print("List of possible commands. Will greedily get called sequentially in the order you list them")
    chain("init", "downloads and sets up dependencies and docker compose, etc.")
    chain("train", "trains model given a policy. By default, uses Pure Pursuit. To change, see -p or --policy.")
    chain("run", "runs the trained policy locally in the simulator.")
    print("List of possible -args.")
    chain("-h", "--help", "shows this current help message")
    chain("-p", "--policy", "sets the current policy. Must match a filename found in ./1_develop/custom. If unspecified, defaults to lf_slim_pp")
    chain("-v", "--verbose", "does everything as spammy as possible")

class ArgParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        help()  # TODO refactor help() to use stadard argparse help stuff, so that help is genned dynamically
        exit(-1)

parser = ArgParser(description='RL utility. -h or --help for help')
parser.add_argument("scripts", nargs='*')
parser.add_argument("-v", "--verbose", nargs="?", default=False, const=True)
parser.add_argument("-p", "--policy", default="lf_slim_pp")
args = parser.parse_args()

import os
import subprocess
import contextlib

@contextlib.contextmanager
def chdir(path):
    retval = os.getcwd()
    os.chdir(path)
    yield None
    os.chdir(retval)

def call(cmd):
    x = subprocess.check_output(cmd, universal_newlines=True, shell=True)
    if args.verbose:
        print(x)
    return x

def _init():
    # From Duckietown: https://docs.duckietown.org/DT19/AIDO/out/embodied_rpl.html
    with chdir("1_develop"):
        call("git submodule init")
        call("git submodule update")
        call('git submodule foreach "(git checkout daffy; git pull)"')

        # We Interrupt Your Regularly Scheduled Programming to Bring You Bugfixes

        # Docker's latest version broke the `version` flag in docker-compose.yml 2.x, and said flag isn't in 3.x at all.
        # So now we have to install the nvidia-container-runtime from the source? Not 100% sure what the reason is, but hey.
        # It works.
        #
        # https://nvidia.github.io/nvidia-container-runtime/
        call("""curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
      sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)""")
        call("""curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list""")
        call("apt-get update")

        # Cont.: https://github.com/NVIDIA/nvidia-container-runtime#docker-engine-setup
        call("apt-get install nvidia-container-runtime -y")
        call("""echo '{ "runtimes": { "nvidia": { "path": "/usr/bin/nvidia-container-runtime", "runtimeArgs": [] } } }' >> /etc/docker/daemon.json""")
        call("pkill -SIGHUP dockerd")
        call("service docker restart")

        # We Return to Your Regularly Scheduled Programming

        col_print(TRED, "This is going to be VERY LONG and use lots of CPU if it's your first time doing init. Prepare for the long haul.")
        call("docker-compose up")

for script in args.scripts:
    print("Running "); com_print(script); print()
    locals()["_"+script]()

print(args.scripts)

