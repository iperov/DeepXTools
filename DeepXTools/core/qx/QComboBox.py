from .. import lx, mx, qt
from ._helpers import q_init
from .QApplication import QApplication
from .QEvent import QEvent1
from .QFuncWrap import QFuncWrap
from .QWidget import QWidget


class QComboBox(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_combobox', qt.QComboBox, **kwargs), **kwargs)

        q_combobox = self.get_q_combobox()

        self.__show_popup_wrap = QFuncWrap(q_combobox, 'showPopup', lambda *args, **kwargs: self.show_popup()).dispose_with(self)

        self._mx_current_index = mx.GetSetProperty[int|None](self.get_current_index, self.set_current_index, QEvent1[int](q_combobox.currentIndexChanged).dispose_with(self)).dispose_with(self)

        q_combobox.setSizeAdjustPolicy(qt.QComboBox.SizeAdjustPolicy.AdjustToContents)

    @property
    def mx_current_index(self) -> mx.IProperty[int|None]: return self._mx_current_index

    def get_q_combobox(self) -> qt.QComboBox: return self.get_q_widget()

    def get_placeholder_text(self) -> str: return self.get_q_combobox().placeholderText()

    def get_current_index(self) -> int|None:
        idx = self.get_q_combobox().currentIndex()
        return None if idx == -1 else idx

    def set_current_index(self, idx : int|None):
        self.get_q_combobox().setCurrentIndex(-1 if idx is None else idx)
        return self

    def set_placeholder_text(self, text : str|None):
        self.get_q_combobox().setPlaceholderText(text)
        return self

    def clear(self):
        self.get_q_combobox().clear()
        return self

    def add_item(self, text : str):
        self.get_q_combobox().addItem(text)
        idx = self.get_q_combobox().count()-1

        attr_name = f'_QComboBox_item_{idx}_text_disp'
        if (disp := getattr(self, attr_name, None)) is not None:
            disp.dispose()

        setattr(self, attr_name,
                QApplication.instance().mx_language.reflect(lambda lang: self.get_q_combobox().setItemText(idx, lx.L(text, lang))).dispose_with(self)
                )


        return self

    def show_popup(self):
        self.__show_popup_wrap.get_super()()
        return self
