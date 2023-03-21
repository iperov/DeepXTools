from .. import mx, qt
from ._constants import Orientation
from ._helpers import q_init
from .QEvent import QEvent0, QEvent1
from .QWidget import QWidget


class QAbstractSlider(QWidget):
    def __init__(self, **kwargs):
        super().__init__(q_widget=q_init('q_abstract_slider', _QAbstractSliderImpl, qt.QAbstractSlider, **kwargs), **kwargs)

        q_abstract_slider = self.get_q_widget()
        self._mx_slider_moved = QEvent1[int](q_abstract_slider.sliderMoved).dispose_with(self)
        self._mx_slider_pressed = QEvent0(q_abstract_slider.sliderPressed).dispose_with(self)
        self._mx_slider_released = QEvent0(q_abstract_slider.sliderReleased).dispose_with(self)

        self._mx_value = mx.GetSetProperty[int](self.get_value, self.set_value, QEvent1[int](q_abstract_slider.valueChanged).dispose_with(self) ).dispose_with(self)

        if isinstance(q_abstract_slider, _QAbstractSliderImpl):
            ...

        self.set_orientation(Orientation.Horizontal)
        self.set_minimum(0)
        self.set_maximum(10)
        self.set_value(0)

    @property
    def mx_slider_moved(self) -> mx.IEvent1_r[int]: return self._mx_slider_moved
    @property
    def mx_slider_pressed(self) -> mx.IEvent0_r: return self._mx_slider_pressed
    @property
    def mx_slider_released(self) -> mx.IEvent0_r: return self._mx_slider_released
    @property
    def mx_value(self) -> mx.IProperty[int]: return self._mx_value

    def get_q_abstract_slider(self) -> qt.QAbstractSlider: return self.get_q_widget()

    def get_value(self) -> int: return self.get_q_abstract_slider().value()

    def set_orientation(self, orientation : Orientation):
        self.get_q_abstract_slider().setOrientation(orientation)
        return self

    def set_minimum(self, minimum : int):
        self.get_q_abstract_slider().setMinimum(minimum)
        return self

    def set_maximum(self, maximum : int):
        self.get_q_abstract_slider().setMaximum(maximum)
        return self

    def set_value(self, value : int):
        self.get_q_abstract_slider().setValue(value)
        return self


class _QAbstractSliderImpl(qt.QAbstractSlider):
    ...