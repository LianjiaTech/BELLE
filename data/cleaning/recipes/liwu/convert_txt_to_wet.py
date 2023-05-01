import os
import sys

import warcio
from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter
from warcio.statusandheaders import StatusAndHeaders
from io import BytesIO


OUTPUT_DIR = 'liwu/output_liwu'
FILE_CHUNK_SIZE = 500


one_file = sys.argv[1]
print(one_file)


output = None
writer = None
num_outputs = 0

gzip = True
gzip_ext = ''
if gzip:
    gzip_ext = '.gz'


def load_txt_file(path: str) -> str:
    for enc in ['utf-8', 'gb18030']:
        try:
            return open(path, encoding=enc).read()
        except UnicodeDecodeError:
            continue
    return None


num_done = 0
with open(one_file) as fp_in:
    for line_idx, line in enumerate(fp_in):
        if output is None:
            output_path = os.path.join(OUTPUT_DIR, os.path.basename(one_file) + f'.{num_outputs}.warc.wet{gzip_ext}')
            print(output_path)
            output = open(output_path, 'wb')
            writer = WARCWriter(output, gzip=gzip)
            num_outputs += 1
        
        text = load_txt_file(line.strip())
        if text is None:
            continue

        target_url = f'http://liwu.data{line.strip()}'

        buffer = BytesIO(text.encode('utf-8'))

        record = writer.create_warc_record(target_url, 'conversion',
                                           payload=buffer,
                                           warc_content_type='text/plain')

        writer.write_record(record)

        num_done += 1
        if (num_done + 1) % FILE_CHUNK_SIZE == 0:
            output.close()
            output = None
            writer = None