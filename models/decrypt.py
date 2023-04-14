import os
import sys
import hashlib
import multiprocessing
import os


def xor_bytes(data, key):
    return bytes(a ^ b for a, b in zip(data, (key * (len(data) // len(key) + 1))[:len(data)]))

def xor_worker(task_queue, result_queue):
    while True:
        chunk_idx, data, key = task_queue.get()
        result_queue.put((chunk_idx, xor_bytes(data, key)))
        task_queue.task_done()

def write_result_chunk(fp, w_chunk_idx, pending, hasher):
    if not pending:
        return w_chunk_idx, pending
    pending.sort()
    for pending_idx, (chunk_idx, chunk) in enumerate(pending):
        if chunk_idx != w_chunk_idx:
            return w_chunk_idx, pending[pending_idx:]
        fp.write(chunk)
        hasher.update(chunk)
        w_chunk_idx += 1
    return w_chunk_idx, []

def main(input_file, key_file, output_dir):
    worker_count = 2
    print(f"Decrypting file {input_file} with {worker_count} workers")

    task_queue = multiprocessing.JoinableQueue(worker_count * 1)
    result_queue = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(target=xor_worker, args=(task_queue, result_queue))
        for _ in range(worker_count)
    ]
    for p in processes:
        p.daemon = True
        p.start()

    chunk_size = 10 * 1024 * 1024
    key_chunk_size = 10 * 1024 * 1024

    hasher = hashlib.sha256()

    # Get the checksum from the input file name
    input_file_basename = os.path.basename(input_file)
    checksum_hex = input_file_basename.split(".")[-2]

    with open(input_file, "rb") as in_file, open(key_file, "rb") as key_file:
        # Get the size of the input file
        file_size = os.path.getsize(input_file)

        # Minus the checksum size
        file_size -= hasher.digest_size

        # Read the checksum from the beginning of the input file
        expected_hash = in_file.read(hasher.digest_size)

        # Create the output file path without the checksum in the filename
        # remove .<checksum>.enc
        input_file_basename = input_file_basename[:-len(checksum_hex) - 5]
        output_file = os.path.join(output_dir, input_file_basename)

        with open(output_file, "wb") as out_file:
            r_chunk_idx = 0  # how many chunks we have read
            w_chunk_idx = 0  # how many chunks have been written
            write_pending = []  # have xor results, awaiting to be written to file

            bytes_read = 0
            while True:
                chunk = in_file.read(chunk_size)
                if not chunk:
                    break

                key_chunk = key_file.read(key_chunk_size)
                if not key_chunk:
                    key_file.seek(0)
                    key_chunk = key_file.read(key_chunk_size)
                
                task_queue.put((r_chunk_idx, chunk, key_chunk))
                # read available results
                while not result_queue.empty():
                    write_pending.append(result_queue.get())
                    
                w_chunk_idx_new, write_pending = write_result_chunk(out_file, w_chunk_idx, write_pending, hasher)

                bytes_read += (w_chunk_idx_new - w_chunk_idx) * chunk_size
                progress = bytes_read / file_size * 100
                sys.stdout.write(f"\rProgress: {progress:.2f}%")
                sys.stdout.flush()
                
                w_chunk_idx = w_chunk_idx_new
                r_chunk_idx += 1

            # wait for xor workers
            sys.stdout.write('\rWaiting for workers...')
            sys.stdout.flush()
            task_queue.join()
            while not result_queue.empty():
                write_pending.append(result_queue.get())
            sys.stdout.write('\rWriting final chunks...')
            sys.stdout.flush()
            write_result_chunk(out_file, w_chunk_idx, write_pending, hasher)

            computed_hash = hasher.digest()

            if computed_hash != expected_hash:
                print("\nError: Checksums do not match. The file may be corrupted.")
                sys.exit(1)

        print ("\nDecryption completed.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: decrypt.py input_file key_file output_dir")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
