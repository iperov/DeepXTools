from ..qt import QColor


class StyleColor:
    Base : QColor = ...
    AlternateBase : QColor = ...
    NoRole : QColor = ...

    Shadow : QColor = ...
    Dark : QColor = ...
    Mid : QColor = ...
    Midlight : QColor = ...
    Light : QColor = ...

    Window : QColor = ...
    Button : QColor = ...

    Text : QColor = ...
    TextDisabled : QColor = ...
    BrightText : QColor = ...
    ButtonText : QColor = ...
    ButtonTextDisabled : QColor = ...
    PlaceholderText : QColor = ...
    WindowText : QColor = ...
    WindowTextDisabled : QColor = ...

    Highlight : QColor = ...
    HighlightedText : QColor = ...

    Link : QColor = ...
    LinkVisited : QColor = ...

    ToolTipBase : QColor = ...
    ToolTipText : QColor = ...
