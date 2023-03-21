from .. import qt
from ._helpers import q_init
from .QWidget import QWidget
from ._constants import Align, Align_to_AlignmentFlag

class QTextBrowser(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_text_browser', qt.QTextBrowser, **kwargs), **kwargs)
        
        q_text_browser = self.get_q_text_browser()
        
    
    def get_q_text_browser(self) -> qt.QTextBrowser: return self.get_q_widget()
    
    def set_align(self, align : Align):
        self.get_q_text_browser().setAlignment(Align_to_AlignmentFlag[align])
        return self
    
    def set_html(self, html : str):
        self.get_q_text_browser().setHtml(html)
        return self
    
    def set_open_external_links(self, b : bool):
        self.get_q_text_browser().setOpenExternalLinks(b)
        return self