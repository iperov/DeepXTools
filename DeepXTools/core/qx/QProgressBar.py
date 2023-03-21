from .. import mx, qt
from ._constants import Align, Align_to_AlignmentFlag, Orientation
from ._helpers import q_init
from .QEvent import QEvent1
from .QWidget import QWidget


class QProgressBar(QWidget):

    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_progress_bar', qt.QProgressBar, **kwargs), **kwargs)

        q_progress_bar = self.get_q_progress_bar()

        self._mx_value = mx.GetSetProperty[int](self.get_value, self.set_value, QEvent1[int](q_progress_bar.valueChanged).dispose_with(self) ).dispose_with(self)

        self.set_orientation(Orientation.Horizontal)


    @property
    def mx_value(self) -> mx.IProperty[int]: return self._mx_value

    def get_q_progress_bar(self) -> qt.QProgressBar: return self.get_q_widget()

    def get_value(self) -> int: return self.get_q_progress_bar().value()

    def set_value(self, value : int):
        self.get_q_progress_bar().setValue(value)
        return self

    def set_alignment(self, align : Align):
        self.get_q_progress_bar().setAlignment(Align_to_AlignmentFlag[align])
        return self

    def set_orientation(self, orientation : Orientation):
        self.get_q_progress_bar().setOrientation(orientation)
        if orientation == Orientation.Horizontal:
            self.v_compact()
            self.h_normal()
        else:
            self.h_compact()
            self.v_normal()

        return self

    def set_minimum(self, min : int):
        self.get_q_progress_bar().setMinimum(min)
        return self

    def set_maximum(self, max : int):
        self.get_q_progress_bar().setMaximum(max)
        return self

    def set_format(self, format : str):
        """
        string used to generate the current text

        %p - is replaced by the percentage completed. %v - is replaced by the current value. %m - is replaced by the total number of steps.

        The default value is "%p%".
        """
        self.get_q_progress_bar().setFormat(format)
        return self

    def set_text_visible(self, visible : bool):
        self.get_q_progress_bar().setTextVisible(visible)
        return self



