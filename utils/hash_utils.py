from pathlib import Path
import xxhash


# Credit to https://stackoverflow.com/a/54477583
def hash_update_from_dir(directory, hash):
    assert Path(directory).is_dir()
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash.update(path.name.encode())
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash.update(chunk)
        elif path.is_dir(): # it is recursive
            hash = hash_update_from_dir(path, hash)
    return hash

def hash_dir(directory: str) -> str:
    '''
    Return a hash digest representing the directory

    Hash changes if filenames, filecontent, or number of files changes
    
    It is recursive
    '''
    hash = xxhash.xxh3_64()
    return hash_update_from_dir(directory, hash).hexdigest()
