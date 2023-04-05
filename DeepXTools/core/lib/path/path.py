import os
import platform
from datetime import datetime
from os import scandir
from pathlib import Path
from typing import List


def creation_date(path_to_file) -> datetime:
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        t = os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            t = stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            t = stat.st_mtime

    return datetime.fromtimestamp(t)


def parents_remained(path : Path, count : int) -> Path:
    """leave only 'count' of last parents in Path"""
    if count == 0:
        return Path(path.name)

    parents = [parent.name for parent in reversed(path.parents) if len(parent.name) != 0]
    if len(path.drive) != 0:
        parents = [path.drive] + parents

    parents = parents[-count:]
    return Path( os.sep.join(parents)  ) / path.name

# def parents_as_string(path : Path, parents_count=0,  ) -> str:
#     parents = [parent.name for parent in reversed(path.parents) if len(parent.name) != 0]

#     if parents_count != 0:
#         parents = parents[-parents_count:]

#     return os.pathsep.join(parents)



def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)
        else:
            yield entry


def get_files_paths(dir_path, extensions=None, subdirs=False) -> List[Path]:
    """
    returns array of Path() of files
    ```
        extensions      ['.jpg', ...]
    ```

    raise on error
    """
    dir_path = Path(dir_path)

    result = []
    if dir_path.exists():

        if subdirs:
            gen = scantree(str(dir_path))
        else:
            gen = scandir(str(dir_path))

        for x in gen:
            p = Path(x.path)
            if p.is_file() and (extensions is None or p.suffix.lower() in extensions):
                result.append(p)

    return sorted(result)

def get_dir_paths(dir_path, subdirs=False) -> List[Path]:
    """
    returns array of Path() of directories
    """
    dir_path = Path (dir_path)

    result = []
    if dir_path.exists():

        if subdirs:
            gen = scantree(str(dir_path))
        else:
            gen = scandir(str(dir_path))

        for x in list(gen):
            if x.is_dir():
                result.append( Path(x.path) )

    return sorted(result)

def relpath(path : Path, cwd : Path = None) -> Path:
    """if possible makes Path relative to `cwd` (or `os.getcwd()`)"""
    try:
        path = Path(os.path.relpath(path, cwd))
        #path = path.relative_to( cwd if cwd is not None else Path(os.getcwd()) )
    except: ...
    
    return path

def abspath(path : Path, cwd : Path = None) -> Path:
    """resolve absolute path according `cwd` (or `os.getcwd()`)"""
    if not path.is_absolute():
        path = Path( os.path.abspath( (cwd if cwd is not None else Path(os.getcwd())) / path ) )
        #path = (cwd if cwd is not None else Path(os.getcwd())) / path
    return path

# def get_image_unique_filestem_paths(dir_path, verbose_print_func=None):
#     result = get_image_paths(dir_path)
#     result_dup = set()

#     for f in result[:]:
#         f_stem = Path(f).stem
#         if f_stem in result_dup:
#             result.remove(f)
#             if verbose_print_func is not None:
#                 verbose_print_func ("Duplicate filenames are not allowed, skipping: %s" % Path(f).name )
#             continue
#         result_dup.add(f_stem)

#     return sorted(result)

# def get_paths(dir_path):
#     dir_path = Path (dir_path)

#     if dir_path.exists():
#         return [ Path(x) for x in sorted([ x.path for x in list(scandir(str(dir_path))) ]) ]
#     else:
#         return []

# def get_file_paths(dir_path):
#     dir_path = Path (dir_path)

#     if dir_path.exists():
#         return [ Path(x) for x in sorted([ x.path for x in list(scandir(str(dir_path))) if x.is_file() ]) ]
#     else:
#         return []

# def get_all_dir_names (dir_path):
#     dir_path = Path (dir_path)

#     if dir_path.exists():
#         return sorted([ x.name for x in list(scandir(str(dir_path))) if x.is_dir() ])
#     else:
#         return []

# def get_all_dir_names_startswith (dir_path, startswith):
#     dir_path = Path (dir_path)
#     startswith = startswith.lower()

#     result = []
#     if dir_path.exists():
#         for x in list(scandir(str(dir_path))):
#             if x.name.lower().startswith(startswith):
#                 result.append ( x.name[len(startswith):] )
#     return sorted(result)

# def get_first_file_by_stem (dir_path, stem, exts=None):
#     dir_path = Path (dir_path)
#     stem = stem.lower()

#     if dir_path.exists():
#         for x in sorted(list(scandir(str(dir_path))), key=lambda x: x.name):
#             if not x.is_file():
#                 continue
#             xp = Path(x.path)
#             if xp.stem.lower() == stem and (exts is None or xp.suffix.lower() in exts):
#                 return xp

#     return None

# def move_all_files (src_dir_path, dst_dir_path):
#     paths = get_file_paths(src_dir_path)
#     for p in paths:
#         p = Path(p)
#         p.rename ( Path(dst_dir_path) / p.name )

# def delete_all_files (dir_path):
#     paths = get_file_paths(dir_path)
#     for p in paths:
#         p = Path(p)
#         p.unlink()
