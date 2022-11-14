def get_verbose_print(verbose: bool) -> callable:
    if verbose:

        def verbose_print(*args, **kwargs):
            print(*args, **kwargs)

    else:

        def verbose_print(*args, **kwargs):
            pass

    return verbose_print
