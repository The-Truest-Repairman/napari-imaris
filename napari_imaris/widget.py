from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel

def resolution_widget(viewer, layer, reader):
    """Dock widget placeholder for resolution info."""
    w = QWidget()
    layout = QVBoxLayout()
    label = QLabel("Zoom-based resolution switching enabled")
    layout.addWidget(label)
    w.setLayout(layout)
    return w
