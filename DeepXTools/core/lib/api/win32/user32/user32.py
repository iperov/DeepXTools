from ..wintypes import dll_import, HWND, INT

SW_HIDE = 0
SW_SHOW = 5

@dll_import('user32')
def ShowWindow( wnd : HWND, nCmdShow : INT ) -> None: ...
