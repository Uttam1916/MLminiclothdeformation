"""
Download additional motion sequences for TailorNet t-shirt_female.

Downloads sequences 001 through 017 for style/shape 000_023 directly from
Hugging Face using HTTP range requests on the official dataset archive,
expanding the dataset from 110 frames to ~1,782 frames (~160 MB).
"""

import os
import sys
import io
import time
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


class RemoteZipFile(io.RawIOBase):
    """File-like object that streams bytes on-demand via HTTP range requests."""
    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as resp:
            self.size = int(resp.headers['content-length'])
        self.pos = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size == -1 or self.pos + size > self.size:
            size = self.size - self.pos
        if size <= 0:
            return b''
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{self.pos + size - 1}'})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                self.pos += len(data)
                return data
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(1.0 * (attempt + 1))


def download_missing_sequences(
    output_dir: str = "data/extracted/t-shirt_female/pose/000_023",
    max_sequences: int = 18,
    max_workers: int = 4,
):
    os.makedirs(output_dir, exist_ok=True)
    archive_url = "https://huggingface.co/datasets/zycliao/TailorNet_dataset/resolve/main/t-shirt_female.zip"

    # Check which sequences are missing
    needed_files = []
    for i in range(max_sequences):
        p_name = f"poses_{i:03d}.npz"
        u_name = f"unposed_{i:03d}.npy"
        p_path = os.path.join(output_dir, p_name)
        u_path = os.path.join(output_dir, u_name)
        if not os.path.exists(p_path):
            needed_files.append((f"t-shirt_female/pose/000_023/{p_name}", p_path))
        if not os.path.exists(u_path):
            needed_files.append((f"t-shirt_female/pose/000_023/{u_name}", u_path))

    if not needed_files:
        print(f"All {max_sequences} sequences already exist in {output_dir}")
        return

    print(f"Connecting to TailorNet repository to fetch {len(needed_files)} files...")
    remote = RemoteZipFile(archive_url)
    zf = zipfile.ZipFile(remote)
    # Cache file info (offset, size) so threads can fetch independently
    file_entries = []
    for zip_path, local_path in needed_files:
        info = zf.getinfo(zip_path)
        file_entries.append((zip_path, local_path))

    print(f"Downloading {len(needed_files)} files using {max_workers} worker threads...")
    t0 = time.time()

    def fetch_file(entry):
        zip_path, local_path = entry
        # Each thread creates its own ZipFile / RemoteZipFile to avoid seek/read race conditions
        th_remote = RemoteZipFile(archive_url)
        th_zf = zipfile.ZipFile(th_remote)
        data = th_zf.read(zip_path)
        with open(local_path, "wb") as f:
            f.write(data)
        return os.path.basename(local_path), len(data)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_file, entry): entry for entry in file_entries}
        for fut in as_completed(futures):
            fn, sz = fut.result()
            completed += 1
            print(f"[{completed}/{len(file_entries)}] Downloaded {fn} ({sz/1024/1024:.2f} MB)")

    dt = time.time() - t0
    print(f"Finished downloading all sequences in {dt:.1f}s.")


if __name__ == "__main__":
    max_seq = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    download_missing_sequences(max_sequences=max_seq)
