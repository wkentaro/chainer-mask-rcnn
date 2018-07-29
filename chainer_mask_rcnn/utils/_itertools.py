def batch(iterable, n=1):
    size = len(iterable)
    for ndx in range(0, size, n):
        yield iterable[ndx:min(ndx + n, size)]
