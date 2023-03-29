import argparse
import os
import shutil
import ssl
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import List


class WindowsFolderBuilder:
    """
    Builds stand-alone portable all-in-one python folder for Windows with the project from scratch.
    """

    # Constants
    URL_PIP     = r'https://bootstrap.pypa.io/get-pip.py'

    URL_VSCODE  = r'https://code.visualstudio.com/sha/download?build=stable&os=win32-x64-archive'
    #URL_FFMPEG  = r'https://github.com/GyanD/codexffmpeg/releases/download/4.4/ffmpeg-4.4-full_build.zip'
    #URL_7ZIP    = r'https://github.com/iperov/DeepFaceLive/releases/download/7za/7za.zip'
    URL_MSVC    = r'https://github.com/iperov/DeepXTools/releases/download/msvc/msvc.zip'

    URLS_PYTHON = {'3.10.9' : r'https://www.python.org/ftp/python/3.10.9/python-3.10.9-embed-amd64.zip'}

    DIRNAME_INTERNAL = '_internal'
    DIRNAME_INTERNAL_CUDA = 'CUDA'
    DIRNAME_INTERNAL_PYTHON = 'python'
    DIRNAME_INTERNAL_FFMPEG = 'ffmpeg'
    DIRNAME_LOCALENV = '_z'
    DIRNAME_TEMP = 't'
    DIRNAME_USERPROFILE = 'u'
    DIRNAME_APPDATA = 'AppData'
    DIRNAME_LOCAL = 'Local'
    DIRNAME_ROAMING = 'Roaming'
    DIRNAME_DESKTOP = 'Desktop'
    DIRNAME_INTERNAL_VSCODE = 'VSCode'

    def __init__(self,  release_path : Path,
                        cache_path : Path,
                        python_ver : str):

        if release_path.exists():
            for _ in range(3):
                print(f'WARNING !!! {release_path} will be removed !')

            input('Press enter to continue.')
            input('Are you sure? Press enter to continue.')

            shutil.rmtree(release_path)
        while release_path.exists():
            time.sleep(0.1)
        release_path.mkdir(parents=True)

        self._release_path = release_path
        self._python_ver = python_ver
        self._cache_path = cache_path
        self._download_cache_path = cache_path / '_dl_cache'
        self._pip_cache_path = cache_path / '_pip_cache'

        self._validate_env()
        self._install_internal()
        self._install_python()

    def copyfiletree(self, src, dst):
        shutil.copytree(src, dst)

    def copyfile(self, src, dst):
        shutil.copyfile(src, dst)

    def download_file(self, url, savepath : Path, progress_bar=True, use_cached=True):
        """
        Download the file or use cached and save to savepath
        """
        urlpath = Path(url)

        if progress_bar:
            print(f'Downloading {url}')

        f = None
        while True:
            try:

                url_request = urllib.request.urlopen(url, context=ssl._create_unverified_context())
                url_size = int( url_request.getheader('content-length') )

                if use_cached:
                    cached_filepath = self._download_cache_path / urlpath.name
                    if cached_filepath.exists():
                        if url_size == cached_filepath.stat().st_size:
                            print(f'Using cached {cached_filepath}')
                            break
                        else:
                            print('Cached file size mismatch. Downloading from url.')
                else:
                    cached_filepath = savepath

                cached_filepath.parent.mkdir(parents=True, exist_ok=True)

                file_size_dl = 0
                f = open(cached_filepath, 'wb')
                while True:
                    buffer = url_request.read(8192)
                    if not buffer:
                        break

                    f.write(buffer)

                    file_size_dl += len(buffer)

                    if progress_bar:
                        print(f'Downloading {file_size_dl} / {url_size}', end='\r')

            except:
                print(f'Unable to download {url}')
                raise
            break

        if f is not None:
            f.close()

        if use_cached:
            shutil.copy2(cached_filepath, savepath)

    def rmdir(self, path):
        os.system('del /F /S /Q "{}" > nul'.format(str(path)))
        os.system('rmdir /S /Q "{}"'.format(str(path)))


    def rmdir_in_all_subdirs(self, path, subdirname):
        for root, dirs, files in os.walk( str(path), topdown=False):
            if subdirname in dirs:
                self.rmdir( Path(root) / subdirname )

    def get_cuda_bin_path(self) -> Path: return self._cuda_bin_path
    def get_internal_path(self) -> Path: return self._internal_path
    def get_python_site_packages_path(self) -> Path: return self._python_site_packages_path
    def get_release_path(self) -> Path: return self._release_path

    def _validate_env(self):
        env = os.environ.copy()

        self._internal_path = self._release_path / self.DIRNAME_INTERNAL
        self._internal_path.mkdir(exist_ok=True, parents=True)

        self._local_env_path = self._internal_path / self.DIRNAME_LOCALENV
        self._local_env_path.mkdir(exist_ok=True, parents=True)

        self._temp_path = self._local_env_path / self.DIRNAME_TEMP
        self._temp_path.mkdir(exist_ok=True, parents=True)

        self._userprofile_path = self._local_env_path / self.DIRNAME_USERPROFILE
        self._userprofile_path.mkdir(exist_ok=True, parents=True)

        self._desktop_path = self._userprofile_path / self.DIRNAME_DESKTOP
        self._desktop_path.mkdir(exist_ok=True, parents=True)

        self._localappdata_path = self._userprofile_path / self.DIRNAME_APPDATA / self.DIRNAME_LOCAL
        self._localappdata_path.mkdir(exist_ok=True, parents=True)

        self._appdata_path = self._userprofile_path / self.DIRNAME_APPDATA / self.DIRNAME_ROAMING
        self._appdata_path.mkdir(exist_ok=True, parents=True)

        self._python_path = self._internal_path / self.DIRNAME_INTERNAL_PYTHON
        self._python_path.mkdir(exist_ok=True, parents=True)

        self._python_site_packages_path = self._python_path / 'Lib' / 'site-packages'
        self._python_site_packages_path.mkdir(exist_ok=True, parents=True)

        self._cuda_path = self._internal_path / self.DIRNAME_INTERNAL_CUDA
        self._cuda_path.mkdir(exist_ok=True, parents=True)

        self._cuda_bin_path = self._cuda_path / 'bin'
        self._cuda_bin_path.mkdir(exist_ok=True, parents=True)

        self._vscode_path = self._internal_path / self.DIRNAME_INTERNAL_VSCODE
        self._ffmpeg_path = self._internal_path / self.DIRNAME_INTERNAL_FFMPEG

        self._7zip_path = self._temp_path / '7zip'


        env['INTERNAL']     = str(self._internal_path)
        env['LOCALENV']     = str(self._local_env_path)
        env['TMP']          = \
        env['TEMP']         = str(self._temp_path)
        env['HOME']         = \
        env['HOMEPATH']     = \
        env['USERPROFILE']  = str(self._userprofile_path)
        env['DESKTOP']      = str(self._desktop_path)
        env['LOCALAPPDATA'] = str(self._localappdata_path)
        env['APPDATA']      = str(self._appdata_path)
        env['PYTHONHOME']   = ''
        env['PYTHONPATH']   = ''
        env['PYTHON_PATH']  = str(self._python_path)
        env['PYTHONEXECUTABLE']  = \
        env['PYTHON_EXECUTABLE'] = \
        env['PYTHON_BIN_PATH']   = str(self._python_path / 'python.exe')
        env['PYTHONWEXECUTABLE'] = \
        env['PYTHON_WEXECUTABLE'] = str(self._python_path / 'pythonw.exe')
        env['PYTHON_LIB_PATH']    = str(self._python_path / 'Lib' / 'site-packages')
        env['CUDA_PATH']    = str(self._cuda_path)
        env['PATH']   = f"{str(self._cuda_path)};{str(self._python_path)};{str(self._python_path / 'Scripts')};{env['PATH']}"

        if self._pip_cache_path is not None:
            env['PIP_CACHE_DIR'] = str(self._pip_cache_path)

        self.env = env

    def _install_internal(self):

        (self._internal_path / 'setenv.bat').write_text(
fr"""@echo off
SET INTERNAL=%~dp0
SET INTERNAL=%INTERNAL:~0,-1%
SET LOCALENV=%INTERNAL%\{self.DIRNAME_LOCALENV}
SET TMP=%LOCALENV%\{self.DIRNAME_TEMP}
SET TEMP=%TMP%
SET HOME=%LOCALENV%\{self.DIRNAME_USERPROFILE}
SET HOMEPATH=%HOME%
SET USERPROFILE=%HOME%
SET DESKTOP=%HOME%\{self.DIRNAME_DESKTOP}
SET LOCALAPPDATA=%USERPROFILE%\{self.DIRNAME_APPDATA}\{self.DIRNAME_LOCAL}
SET APPDATA=%USERPROFILE%\{self.DIRNAME_APPDATA}\{self.DIRNAME_ROAMING}

SET PYTHONHOME=
SET PYTHONPATH=
SET PYTHON_PATH=%INTERNAL%\python
SET PYTHONEXECUTABLE=%PYTHON_PATH%\python.exe
SET PYTHON_EXECUTABLE=%PYTHONEXECUTABLE%
SET PYTHONWEXECUTABLE=%PYTHON_PATH%\pythonw.exe
SET PYTHONW_EXECUTABLE=%PYTHONWEXECUTABLE%
SET PYTHON_BIN_PATH=%PYTHONEXECUTABLE%
SET PYTHON_LIB_PATH=%PYTHON_PATH%\Lib\site-packages
SET CUDA_PATH=%INTERNAL%\CUDA
SET CUDA_BIN_PATH=%CUDA_PATH%\bin
SET QT_QPA_PLATFORM_PLUGIN_PATH=%PYTHON_LIB_PATH%\PyQT6\Qt6\Plugins\platforms

SET PATH=%INTERNAL%\ffmpeg;%PYTHON_PATH%;%CUDA_BIN_PATH%;%PYTHON_PATH%\Scripts;%PATH%
""")
        self.clearenv_bat_path = self._internal_path / 'clearenv.bat'
        self.clearenv_bat_path.write_text(
fr"""@echo off
cd /D %~dp0
call setenv.bat
rmdir %LOCALENV% /s /q 2>nul
mkdir %LOCALENV%
mkdir %TEMP%
mkdir %USERPROFILE%
mkdir %DESKTOP%
mkdir %LOCALAPPDATA%
mkdir %APPDATA%
""")
        (self._internal_path / 'python_console.bat').write_text(
fr"""
@echo off
cd /D %~dp0
call setenv.bat
cd python
cmd
""")

    def _install_python(self):
        python_url = self.URLS_PYTHON.get(self._python_ver, None)
        if python_url is None:
            raise Exception(f'No python URL defined for {self._python_ver}')

        print (f"Installing python {self._python_ver} to {self._python_path}\n")

        python_dl_path = self._python_path / f'python-{self._python_ver}.zip'

        if not python_dl_path.exists():
            self.download_file(python_url, python_dl_path)

        with zipfile.ZipFile(python_dl_path, 'r') as zip_ref:
            zip_ref.extractall(self._python_path)

        python_dl_path.unlink()

        # Remove _pth file
        for pth_file in self._python_path.glob("*._pth"):
            pth_file.unlink()

        print('Installing MS VC dlls.')

        self.download_and_unzip(self.URL_MSVC, self._python_path)

        print ("Installing pip.\n")

        python_pip_path = self._python_path / 'get-pip.py'

        self.download_file(self.URL_PIP, python_pip_path)

        subprocess.Popen(args='python.exe get-pip.py', cwd=str(self._python_path), shell=True, env=self.env).wait()
        python_pip_path.unlink()

    def _get_7zip_bin_path(self):
        if not self._7zip_path.exists():
            self.download_and_unzip(self.URL_7ZIP, self._7zip_path)
        return self._7zip_path / '7za.exe'

    def cleanup(self):
        print ('Cleanup.\n')
        subprocess.Popen(args=str(self.clearenv_bat_path), shell=True).wait()
        self.rmdir_in_all_subdirs (self._release_path, '__pycache__')

    def pack_sfx_release(self, archive_name):
        archiver_path = self._get_7zip_bin_path()
        archive_path = self._release_path.parent / (archive_name+'.exe')

        subprocess.Popen(args='"%s" a -t7z -sfx7z.sfx -m0=LZMA2 -mx9 -mtm=off -mmt=8 "%s" "%s"' % ( \
                                str(archiver_path),
                                str(archive_path),
                                str(self._release_path)  ),
                            shell=True).wait()

    def download_and_unzip(self, url, unzip_dirpath, only_files_list : List =None):
        """
        Download and unzip entire content to unzip_dirpath

         only_files_list(None)  if specified
                                only first match of these files
                                will be extracted to unzip_dirpath without folder structure
        """
        unzip_dirpath.mkdir(parents=True, exist_ok=True)

        tmp_zippath = unzip_dirpath / '__dl.zip'

        self.download_file(url, tmp_zippath)

        with zipfile.ZipFile(tmp_zippath, 'r') as zip_ref:
            for entry in zip_ref.filelist:

                if only_files_list is not None:
                    if not entry.is_dir():
                        entry_filepath = Path( entry.filename )
                        if entry_filepath.name in only_files_list:
                            only_files_list.remove(entry_filepath.name)
                            (unzip_dirpath / entry_filepath.name).write_bytes ( zip_ref.read(entry) )
                else:
                    entry_outpath = unzip_dirpath / Path(entry.filename)

                    if entry.is_dir():
                        entry_outpath.mkdir(parents=True, exist_ok=True)
                    else:
                        entry_outpath.write_bytes ( zip_ref.read(entry) )

        tmp_zippath.unlink()

    def install_pip_package(self, pkg_name):
        subprocess.Popen(args=f'python.exe -m pip install {pkg_name}', cwd=str(self._python_path), shell=True, env=self.env).wait()

    def run_python(self, argsline, cwd=None):
        if cwd is None:
            cwd = self._python_path
        subprocess.Popen(args=f'python.exe {argsline}', cwd=str(cwd), shell=True, env=self.env).wait()

    def install_ffmpeg_binaries(self):
        print('Installing ffmpeg binaries.')
        self._ffmpeg_path.mkdir(exist_ok=True, parents=True)
        self.download_and_unzip(self.URL_FFMPEG, self._ffmpeg_path, only_files_list=['ffmpeg.exe', 'ffprobe.exe'] )

    def install_vscode(self, folders : List[str] = None):
        """
        Installs vscode
        """
        print('Installing VSCode.\n')

        self._vscode_path.mkdir(exist_ok=True, parents=True)
        vscode_zip_path = self._vscode_path / 'VSCode.zip'
        self.download_file(self.URL_VSCODE, vscode_zip_path, use_cached=False)
        with zipfile.ZipFile(vscode_zip_path, 'r') as zip_ref:
            zip_ref.extractall(self._vscode_path)
        vscode_zip_path.unlink()

        # Create bat
        (self._internal_path  / 'vscode.bat').write_text(
fr"""@echo off
cd /D %~dp0
call setenv.bat
start "" /D "%~dp0" "%INTERNAL%\{self.DIRNAME_INTERNAL_VSCODE}\Code.exe" --disable-workspace-trust "project.code-workspace"
""")

        # Enable portable mode in VSCode
        (self._vscode_path / 'data').mkdir(exist_ok=True)

        # Create vscode project
        if folders is None:
            folders = ['.']

        s_folders = ',\n'.join( f'{{ "path" : "{f}" }}' for f in folders )


        (self._internal_path / 'project.code-workspace').write_text (
fr'''{{
	"folders": [{s_folders}
    ],

	"settings": {{
        "breadcrumbs.enabled": false,
        "debug.showBreakpointsInOverviewRuler": true,
        "diffEditor.ignoreTrimWhitespace": true,
        "extensions.ignoreRecommendations": true,
        "editor.renderWhitespace": "none",
        "editor.fastScrollSensitivity": 10,
		"editor.folding": false,
        "editor.minimap.enabled": false,
        "editor.mouseWheelScrollSensitivity": 2,
		"editor.mouseWheelScrollSensitivity": 3,
		"editor.glyphMargin": false,
        "editor.quickSuggestions": {{"other": false,"comments": false,"strings": false}},
			"editor.trimAutoWhitespace": false,
			"python.linting.pylintArgs": ["--disable=import-error"],
            "python.linting.enabled": false,
            "editor.lightbulb.enabled": false,
            "python.languageServer": "Pylance"
        "window.menuBarVisibility": "default",
        "window.zoomLevel": 0,
        "python.defaultInterpreterPath": "${{env:PYTHON_EXECUTABLE}}",
        "python.linting.enabled": false,
        "python.linting.pylintEnabled": false,
        "python.linting.pylamaEnabled": false,
        "python.linting.pydocstyleEnabled": false,
        "telemetry.enableTelemetry": false,
        "workbench.colorTheme": "Visual Studio Light",
        "workbench.activityBar.visible": true,
		"workbench.editor.tabCloseButton": "off",
		"workbench.editor.tabSizing": "shrink",
		"workbench.editor.highlightModifiedTabs": true,
        "workbench.enableExperiments": false,
        "workbench.sideBar.location": "right",
		"files.exclude": {{
			"**/__pycache__": true,
			"**/.github": true,
			"**/.vscode": true,
			"**/*.dat": true,
			"**/*.h5": true,
            "**/*.npy": true
		}},
	}}
}}
''')
        subprocess.Popen(args=f'bin\code.cmd --disable-workspace-trust --install-extension ms-python.python', cwd=self._vscode_path, shell=True, env=self.env).wait()
        subprocess.Popen(args=f'bin\code.cmd --disable-workspace-trust --install-extension ms-python.vscode-pylance', cwd=self._vscode_path, shell=True, env=self.env).wait()
        subprocess.Popen(args=f'bin\code.cmd --disable-workspace-trust --install-extension searking.preview-vscode', cwd=self._vscode_path, shell=True, env=self.env).wait()

    def create_run_python_script(self, script_name : str, internal_relative_path : str, args_str : str):

        (self._release_path / script_name).write_text(
fr"""@echo off
cd /D %~dp0
call {self.DIRNAME_INTERNAL}\setenv.bat
"%PYTHONEXECUTABLE%" {self.DIRNAME_INTERNAL}\{internal_relative_path} {args_str}
""")

    def create_internal_run_python_script(self, script_name : str, internal_relative_path : str, args_str : str):

        (self._internal_path / script_name).write_text(
fr"""@echo off
cd /D %~dp0
call setenv.bat
"%PYTHONEXECUTABLE%" {internal_relative_path} {args_str}
""")

def install_deepxtools(release_dir, cache_dir, python_ver='3.10.9', backend='cuda'):
    builder = WindowsFolderBuilder(release_path=Path(release_dir),
                                   cache_path=Path(cache_dir),
                                   python_ver=python_ver)

    # PIP INSTALLATIONS
    builder.install_pip_package('numpy==1.23.5')
    builder.install_pip_package('numba==0.56.4')
    builder.install_pip_package('PySide6==6.4.1')
    builder.install_pip_package('opencv-python==4.7.0.68')
    builder.install_pip_package('opencv-contrib-python==4.7.0.68')

    if backend == 'cuda':
        builder.install_pip_package('torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html')

        print('Moving CUDA dlls from Torch to shared directory')
        cuda_bin_path = builder.get_cuda_bin_path()
        torch_lib_path = builder.get_python_site_packages_path() / 'torch' / 'lib'

        for cu_file in torch_lib_path.glob("**/cu*64*.dll"):
            target = cuda_bin_path / cu_file.name
            print (f'Moving {target}')
            shutil.move (str(cu_file), str(target) )

        for file in torch_lib_path.glob("**/nvrtc*.dll"):
            target = cuda_bin_path / file.name
            print (f'Moving {target}')
            shutil.move (str(file), str(target) )

        for file in torch_lib_path.glob("**/zlibwapi.dll"):
            target = cuda_bin_path / file.name
            print (f'Copying {target}')
            shutil.copy (str(file), str(target) )


    repo_path = builder.get_internal_path() / 'repo'

    print('Copying DeepXTools repository.')
    builder.copyfiletree(Path(__file__).parent.parent.parent, repo_path)
    builder.rmdir_in_all_subdirs(repo_path, '.git')

    print('Creating files.')

    release_path = builder.get_release_path()

    data_dirpath = release_path / 'data'
    data_dirpath.mkdir(parents=True, exist_ok=True)

    saves_dirpath = release_path / 'saves'
    saves_dirpath.mkdir(parents=True, exist_ok=True)

    builder.create_run_python_script('MaskEditor.bat', 'repo\\DeepXTools\\main.py', 'run MaskEditor --ui-data-dir "%~dp0_internal"')
    builder.create_run_python_script('DeepRoto.bat', 'repo\\DeepXTools\\main.py', 'run DeepRoto  --ui-data-dir "%~dp0_internal"')

    builder.install_vscode(folders=['repo/DeepXTools','repo'])

    builder.cleanup()


class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--release-dir', action=fixPathAction, default=None)
    p.add_argument('--cache-dir', action=fixPathAction, default=None)
    p.add_argument('--backend', choices=['cuda', 'directml'], default='cuda')

    args = p.parse_args()

    install_deepxtools(release_dir=args.release_dir,
                       cache_dir=args.cache_dir,
                       backend=args.backend)