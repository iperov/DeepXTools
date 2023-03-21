from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from ... import mx, qt
from .IonIcon import IonIcon

T = TypeVar('T')

class QIonIconDB(mx.Disposable):

    @staticmethod
    def instance() -> QIonIconDB:
        if QIonIconDB._instance is None:
            raise Exception('No QIonIconDB instance.')
        return QIonIconDB._instance

    def __init__(self):
        super().__init__()
        if QIonIconDB._instance is not None:
            raise Exception('QIonIconDB instance already exists.')
        QIonIconDB._instance = self
        self._cached = {}

    def __dispose__(self):
        QIonIconDB._instance = None
        self._cached = None
        super().__dispose__()

    def pixmap(self, icon : IonIcon, color : qt.QColor) -> qt.QPixmap:
        return self._get(icon, color, qt.QPixmap)

    def image(self, icon : IonIcon, color : qt.QColor) -> qt.QImage:
        return self._get(icon, color, qt.QImage)

    def icon(self, icon : IonIcon, color : qt.QColor) -> qt.QIcon:
        return self._get(icon, color, qt.QIcon)

    def _get(self, icon : IonIcon, color : qt.QColor, out_cls):
        key = (icon, color.getRgb(), out_cls)

        result = self._cached.get(key, None)

        if result is None:
            if issubclass(out_cls, qt.QPixmap):

                result = self._cached[key] = qt.QPixmap_colorized(qt.QPixmap(str(Path(__file__).parent / 'assets' / (icon.name+'.png'))), color)

            elif issubclass(out_cls, qt.QImage):
                result = self._cached[key] = self._get(icon, color, qt.QPixmap).toImage()
            elif issubclass(out_cls, qt.QIcon):
                result = self._cached[key] = qt.QIcon(self._get(icon, color, qt.QPixmap))
            else:
                raise ValueError('Unknown type out_cls')

        return result

    _instance : QIonIconDB = None


# class XIonIconImage(Image):
#     def __init__(self, ionicon_db : QIonIconDB, icon : IonIcon, color : ):
#         super().__init__()
#         self._ionicon_db = ionicon_db
#         self._icon = icon
#         self._color = EColor_to_QColor[color]

#     def _get_q_pixmap(self) -> QPixmap:
#         return self._ionicon_db._get( self._icon.name, self._color, None, QPixmap )

#     def _get_q_icon(self) -> QIcon:
#         return self._ionicon_db._get( self._icon.name, self._color, None, QIcon )

#     def _get_q_image(self) -> QImage:
#         return self._ionicon_db._get( self._icon.name, self._color, None, QImage )

