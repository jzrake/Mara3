#!/usr/bin/env python3



import os
import datetime
import getpass
import pathlib
import configparser
import argparse
import copy



readme_template = """\
User: {user}
Date: {date}

{comment}
"""



def mara_command(subprog, exe='mara', **kwargs):
    return './{} {} {}'.format(exe, subprog,
        ' '.join(['{}={}'.format(k, v) for k, v in kwargs.items()]))



def run_script(template, subprog, exe='mara', runid='test', nodes=1, hours=8, **kwargs):
    return template.format(
        nodes=nodes,
        hours=hours,
        job_name=runid,
        output=os.path.join(kwargs.get('outdir', './'), runid + '.out'),
        command=mara_command(subprog, exe=exe, **kwargs))



def load_suite(suite_file):
    suite_file_defs = dict()
    exec(open(suite_file).read(), suite_file_defs)
    return suite_file_defs['suite']



def load_machine(machine_file):
    try:
        machine_file_defs = dict()
        exec(open(machine_file).read(), machine_file_defs)
        return machine_file_defs['machine']
    except:
        print("Using the default machine parameters")
        return {
            'submit_command': 'bash',
            'submit_script': '{command}',
        }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('suite_file', help="Python file defining a 'suite' variable")
    parser.add_argument('--machine-file', default='machine.py', help="Python file defining a 'machine' variable")
    parser.add_argument('--submit', action='store_true', help="Also launch jobs")
    parser.add_argument('--quiet', '-q', action='store_true', help="Suppress printing of submit script contents to the terminal")
    args = parser.parse_args()

    suite = load_suite(args.suite_file)
    machine = load_machine(args.machine_file)

    for runid in suite['runs']:

        suite_name          = suite.get('name', pathlib.Path(args.suite_file).stem)
        suite_dir           = suite.get('root_dir', pathlib.Path(args.suite_file).parent)
        mara_opts           = copy.deepcopy(suite.get('mara_opts', dict()))
        run_dir             = os.path.join(str(suite_dir), suite_name, runid)
        submit_file         = os.path.join(run_dir, 'submit.sh')
        readme_file         = os.path.join(run_dir, 'README')
        mara_opts['outdir'] = run_dir
        mara_opts.update(suite['runs'][runid])

        submit_content = run_script(
            machine['submit_script'],
            suite['subprog'],
            exe=suite.get('exe', 'mara'),
            runid=runid,
            nodes=suite['job_params']['nodes'],
            hours=suite['job_params']['hours'],
            **mara_opts)

        readme_content = readme_template.format(
                date=datetime.datetime.now(),
                user=getpass.getuser(),
                comment=suite.get('comment', ''))

        os.makedirs(run_dir, exist_ok=True)

        with open(readme_file, 'w') as readme: readme.write(readme_content)
        with open(submit_file, 'w') as submit: submit.write(submit_content)

        if not args.quiet: print(submit_content)
        if args.submit: os.system(machine['submit_command'] + ' ' + submit_file)
