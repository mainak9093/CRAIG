"""Download + decompress the LIBSVM datasets used by the CRAIG paper into /tmp/data.

Datasets (exactly those read by util.load_dataset / logistic.py):
  - covtype.libsvm.binary.scale   (581,012 x 54)
  - ijcnn1.tr / ijcnn1.t          (49,990 / 91,701 x 22)
  - combined_scale / combined_scale.t  (78,823 / 19,705 x 100, SensIT Vehicle)

Primary source: LIBSVM site (NTU). Fallback: `libsvmdata` package mirrors.
Run:  python download_data.py
"""
import bz2
import os
import shutil
import ssl
import urllib.request

try:  # Python's bundled openssl may lack the NTU site's issuer chain; certifi has it
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
# Python 3.13 turns on VERIFY_X509_STRICT by default; the NTU cert has no Subject
# Key Identifier and fails strict verification, so relax that single flag.
SSL_CTX.verify_flags &= ~ssl.VERIFY_X509_STRICT

# All datasets live inside the repo: craig-official/data (created on first run).
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
BASE = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets'

# NOTE on ijcnn1: util.load_dataset expects 'ijcnn1.tr' to hold 49,990 rows. On the
# LIBSVM site that is the file 'ijcnn1.bz2' (original 35,000 train + 14,990 val,
# merged); LIBSVM's own 'ijcnn1.tr.bz2' has only 35,000 rows and would be silently
# zero-padded by the loader. So we download ijcnn1.bz2 -> save as ijcnn1.tr.
# NOTE on combined (SensIT Vehicle): files live under multiclass/vehicle/.
FILES = {
    'covtype.libsvm.binary.scale': f'{BASE}/binary/covtype.libsvm.binary.scale.bz2',
    'ijcnn1.tr': f'{BASE}/binary/ijcnn1.bz2',
    'ijcnn1.t': f'{BASE}/binary/ijcnn1.t.bz2',
    'combined_scale': f'{BASE}/multiclass/vehicle/combined_scale.bz2',
    'combined_scale.t': f'{BASE}/multiclass/vehicle/combined_scale.t.bz2',
}


def download(url, dst):
    print(f'downloading {url}')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=300, context=SSL_CTX) as r, open(dst, 'wb') as f:
        shutil.copyfileobj(r, f)


def fetch_via_libsvmdata(name, out):
    """Fetch a dataset with the libsvmdata package and re-serialize it to the raw
    1-based libsvm text format that util.load_dataset parses."""
    from libsvmdata import fetch_libsvm
    from sklearn.datasets import dump_svmlight_file
    lib_names = {  # our filename -> libsvmdata dataset name
        'covtype.libsvm.binary.scale': 'covtype.binary_scale',
        'ijcnn1.tr': 'ijcnn1',            # libsvmdata serves the train split as 'ijcnn1'
        'ijcnn1.t': 'ijcnn1_test',
        'combined_scale': 'sensit_scale',
        'combined_scale.t': 'sensit_scale_test',
    }
    X, y = fetch_libsvm(lib_names[name])
    dump_svmlight_file(X, y, out, zero_based=False)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in FILES.items():
        out = os.path.join(DATA_DIR, name)
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            print(f'[skip] {name} already present')
            continue
        bz = out + '.bz2'
        try:
            download(url, bz)
            print(f'decompressing {name}')
            with bz2.open(bz, 'rb') as src, open(out, 'wb') as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
            os.remove(bz)
            print(f'[ok] {name} ({os.path.getsize(out) >> 20} MB)')
        except Exception as e:
            print(f'[direct download failed] {name}: {e} -- trying libsvmdata fallback')
            try:
                fetch_via_libsvmdata(name, out)
                print(f'[ok via libsvmdata] {name}')
            except Exception as e2:
                print(f'[FAIL] {name}: {e2}')

    print('\nLine counts (expect covtype 581012, ijcnn1.tr 49990, ijcnn1.t 91701, '
          'combined_scale 78823, combined_scale.t 19705):')
    for name in FILES:
        p = os.path.join(DATA_DIR, name)
        if os.path.isfile(p):
            with open(p, 'rb') as f:
                n = sum(1 for _ in f)
            print(f'  {name}: {n}')
        else:
            print(f'  {name}: MISSING')


if __name__ == '__main__':
    main()
