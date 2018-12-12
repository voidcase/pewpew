import json
from pprint import pprint
from pathlib import Path
import re


ROOT = Path('/data/staff/common/ML-crystals/meta_sandbox')
# ROOT = Path('/data/visitors/biomax/20180479/20181119/raw')


def get_sample(x: Path):
    return str(re.search('Sample-([0-9]+-[0-9]+)', str(x)).group(1))


def group_by_sample(root: Path):
    for path in root.glob('*.meta.txt'):
        meta_reader = open(str(path), 'r')
        meta = json.load(meta_reader)
        meta_reader.close()
        meta['fileinfo']['filename']
        sample = get_sample(meta['fileinfo']['filename'])
        sample_dir = root / f'Sample-{sample}'
        if not sample_dir.is_dir():
            sample_dir.mkdir()
        path.rename(sample_dir / path.name)
        print(f'moved {path}')


if __name__ == '__main__':
    subs = json.load(open('./manual_zoom_assignment.json', 'r'))
    print(subs)

    for metapath in ROOT.rglob('*.meta.txt'):
        print(metapath)
        readfile = open(str(metapath), 'r')
        metadata = json.load(readfile)
        readfile.close()
        sample = get_sample(metadata['fileinfo']['filename'])
        if sample in subs and 'zoom1' not in metadata:
            print('setting zoom', subs[sample], f'for Sample-{sample}')
            metadata['zoom1'] = f'Zoom {subs[sample]}'
            metadata['MANUALLY_SET_ZOOM'] = True
            print(metadata['zoom1'])
            writefile = open(str(metapath), 'w')
            json.dump(metadata, writefile)
            writefile.close()
