"""Execute notebook(s) in place with the sagemath kernel and report timing.
Usage: sage -python scratch/exec_nb.py <nb1> <nb2> ...
"""
import sys
import time

import nbformat
from nbclient import NotebookClient


def run_one(path):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name='sagemath-10.8',
                            allow_errors=False)
    t0 = time.time()
    try:
        client.execute()
        nbformat.write(nb, path)
        print(f'OK   {time.time()-t0:7.1f}s  {path}')
        return True
    except Exception as e:
        print(f'FAIL {time.time()-t0:7.1f}s  {path}')
        print('   ', type(e).__name__, str(e)[:600])
        return False


if __name__ == '__main__':
    ok = all(run_one(p) for p in sys.argv[1:])
    sys.exit(0 if ok else 1)
