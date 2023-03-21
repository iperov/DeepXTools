from .. import mx, qt
from ._constants import Align
from .QBox import QHBox, QVBox
from .QFontDB import Font
from .QFrame import QVFrame
from .QIonIconDB import IonIcon, QIonIconDB
from .QLabel import QLabel
from .QPushButton import QPushButton
from .QTextEdit import QTextEdit


class QMsgNotifyMxTextEmitter(QHBox):
    def __init__(self, text_emitter : mx.ITextEmitter_r):
        super().__init__()
        self._text_emitter = text_emitter
        self._title = None

        te = self._te = QTextEdit().set_font(Font.FixedWidth).set_read_only(True)

        self._title_label = QLabel()

        (self.hide().set_spacing(1)
            .add(QVBox()
                    .add(QVFrame().add(self._title_label, align=Align.CenterH))
                    .add(te)
                    )
            .add(QPushButton().h_compact()
                    .set_icon(QIonIconDB.instance().icon(IonIcon.checkmark_done, qt.QColor(100,200,0)))
                    .inline(lambda btn: btn.mx_clicked.listen(lambda: (te.clear(), self.hide())))))

        text_emitter.listen(self._on_text).dispose_with(self)

    def set_title(self, title : str|None):
        self._title_label.set_text(title).set_visible(title is not None)
        return self

    def _on_text(self, text : str):
        te = self._te.get_q_text_edit()

        cursor = te.textCursor()
        cursor.clearSelection()

        cursor.movePosition(qt.QTextCursor.MoveOperation.End)

        if cursor.position() != 0:
            cursor.insertText('\n')

        cursor.insertText(text)

        te.verticalScrollBar().setValue(te.verticalScrollBar().maximum())

        self.show()