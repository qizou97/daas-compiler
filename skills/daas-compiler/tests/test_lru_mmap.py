import mmap, os, tempfile
import pytest
from daas.dataset import LRUMmapCache

def make_temp_file(content: bytes) -> str:
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(content); f.close()
    return f.name

def test_get_returns_correct_bytes():
    path = make_temp_file(b"hello world")
    cache = LRUMmapCache(maxsize=2)
    mm = cache.get(path)
    assert bytes(mm[0:5]) == b"hello"
    os.unlink(path)

def test_eviction_closes_lru():
    paths = [make_temp_file(f"file{i}".encode()) for i in range(3)]
    cache = LRUMmapCache(maxsize=2)
    cache.get(paths[0])
    cache.get(paths[1])
    assert len(cache._cache) == 2
    cache.get(paths[2])           # should evict paths[0]
    assert len(cache._cache) == 2
    assert paths[0] not in cache._cache
    assert paths[2] in cache._cache
    for p in paths:
        try: os.unlink(p)
        except: pass

def test_lru_order_updated_on_access():
    paths = [make_temp_file(f"x{i}".encode()) for i in range(3)]
    cache = LRUMmapCache(maxsize=2)
    cache.get(paths[0])
    cache.get(paths[1])
    cache.get(paths[0])   # refresh paths[0] → paths[1] is now LRU
    cache.get(paths[2])   # should evict paths[1]
    assert paths[1] not in cache._cache
    assert paths[0] in cache._cache
    for p in paths:
        try: os.unlink(p)
        except: pass
