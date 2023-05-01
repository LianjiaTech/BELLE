import os
import sys
import json
from warcio.warcwriter import WARCWriter
from warcio.statusandheaders import StatusAndHeaders
from io import BytesIO


OUTPUT_DIR = 'oscar_2301/output_oscar'

one_file = sys.argv[1]
print(one_file)
output = None
writer = None
num_outputs = 0

gzip = True
gzip_ext = ''
if gzip:
    gzip_ext = '.gz'

with open(one_file) as fp_in:
    for line_idx, line in enumerate(fp_in):
        if output is None:
            output_path = os.path.join(OUTPUT_DIR, os.path.basename(one_file) + f'.{num_outputs}.warc.wet{gzip_ext}')
            print(output_path)
            output = open(output_path, 'wb')
            writer = WARCWriter(output, gzip=gzip)
            num_outputs += 1
        item = json.loads(line.strip())
        headers_list = item['meta']['warc_headers']
        target_url = headers_list['warc-target-uri']
        headers_list.pop('warc-identified-content-language', None)
        headers_list['WARC-Date'] = headers_list['warc-date']

        text = item['text']

        buffer = BytesIO(text.encode('utf-8'))

        http_headers = StatusAndHeaders('200 OK', headers_list, protocol='HTTP/1.0')

        record = writer.create_warc_record(target_url, 'conversion',
                                           payload=buffer, warc_content_type='text/plain')

        writer.write_record(record)

        if (line_idx + 1) % 10000 == 0:
            output.close()
            output = None
            writer = None
