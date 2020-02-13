#!/usr/bin/env python3




import argparse
import os
import shutil




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--destination', '-d', default='.')
    parser.add_argument('--prefix', default='chkpt')
    parser.add_argument('--groups', default=None)
    args = parser.parse_args()

    os.makedirs(args.destination, exist_ok=True)
    out_prefix = '' if args.prefix == 'chkpt' else '.diagnostics'

    for directory in args.directories:
        if os.path.isdir(directory):

            chkpts = sorted([item for item in os.listdir(directory) if item.startswith(args.prefix)])

            if chkpts:
                old_name = chkpts[-1]
                new_name = old_name.replace(args.prefix, os.path.split(os.path.normpath(directory))[-1] + out_prefix)

                src = os.path.join(directory, old_name)
                dst = os.path.join(args.destination, new_name)

                if args.groups:
                    for group in args.groups.split(','):

                        cmd = 'h5copy -i {} -o {} -s /{} -d /{}'.format(src, dst, group, group)
                        print(cmd)
                        os.system(cmd)

                else:
                    print('cp {} {}'.format(src, dst))
                    shutil.copyfile(src, dst)
