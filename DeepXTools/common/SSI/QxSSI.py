from core import qt, qx

from .MxSSI import MxSSI


class QxSSI:

    class Image(qx.QGrid):
        def __init__(self, ssi_image : MxSSI.Image):
            super().__init__()
            self.set_row_stretch(0,1,1,0)

            if (image := ssi_image.image) is not None:
                self.add( qx.QPixmapWidget().set_pixmap(qt.QPixmap_from_np(ssi_image.image.HWC())), 0, 0 )

            if ssi_image.caption is not None:
                caption = ssi_image.caption
                if image is not None:
                    caption = f'{caption}\n({image.shape[1]}x{image.shape[0]})'

                self.add( qx.QLabel().set_font(qx.Font.FixedWidth).set_align(qx.Align.CenterH).set_text(caption), 1, 0, align=qx.Align.CenterH)

    class Grid(qx.QGrid):
        def __init__(self, ssi_grid : MxSSI.Grid):
            super().__init__()

            for (row,col), item in ssi_grid.items.items():
                if isinstance(item, MxSSI.Image):
                    item_widget = QxSSI.Image(item)
                else:
                    raise NotImplementedError()

                self.add(item_widget, row, col)

    class Sheet(qx.QVBox):
        def __init__(self, ssi_sheet : MxSSI.Sheet):
            super().__init__()

            tab_widget = qx.QTabWidget().set_tab_position(qx.QTabWidget.TabPosition.South)

            for name, section in ssi_sheet.sections.items():
                if isinstance(section, MxSSI.Grid):
                    section_widget = QxSSI.Grid(section)
                else:
                    raise NotImplementedError()

                tab_widget.add_tab(lambda tab: tab.set_title(name).add(section_widget))

            self.add(tab_widget)

