"""Execute notebook(s) WITHOUT writing back (verify 'do they run'), emit one
compact JSON line per notebook. Usage: sage -python scratch/check_nb.py <nb...>
Env: CELL_TIMEOUT (s, default 1200)."""
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

TIMEOUT = int(os.environ.get('CELL_TIMEOUT', '1200'))


def check(path):
    rec = {'path': path, 'ok': False, 'secs': None, 'error': None,
           'err_cell': None}
    t0 = time.time()
    try:
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=TIMEOUT, kernel_name='sagemath-10.8',
                       allow_errors=False).execute()
        rec['ok'] = True
    except CellExecutionError as e:
        rec['error'] = f'{e.ename}: {str(e.evalue)[:300]}'
    except Exception as e:  # timeout, kernel death, read error, …
        rec['error'] = f'{type(e).__name__}: {str(e)[:300]}'
    rec['secs'] = round(time.time() - t0, 1)
    print('RESULT ' + json.dumps(rec), flush=True)
    return rec['ok']


if __name__ == '__main__':
    results = [check(p) for p in sys.argv[1:]]
    print(f'SUMMARY {sum(results)}/{len(results)} ok', flush=True)
