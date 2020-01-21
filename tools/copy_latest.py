#!/usr/bin/env python3




import argparse
import os
import shutil




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--destination', '-d', default='.')
    args = parser.parse_args()

    os.makedirs(args.destination, exist_ok=True)

    for directory in args.directories:
        if os.path.isdir(directory):

            chkpts = sorted([item for item in os.listdir(directory) if item.startswith('chkpt')])

            if chkpts:
                old_name = chkpts[-1]
                new_name = old_name.replace('chkpt', os.path.split(directory)[-1])

                src = os.path.join(directory, old_name)
                dst = os.path.join(args.destination, new_name)

                print('cp {} {}'.format(src, dst))
                shutil.copyfile(src, dst)
