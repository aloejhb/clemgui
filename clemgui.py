import os
import tifffile
import napari
import numpy as np
import pandas as pd

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QShortcut
)
from qtpy.QtGui import QKeySequence
from qtpy.QtCore import QTimer

from functools import partial

from manage_landmarks import LandmarkManager, LandmarkTableWidget
from compute_affine3d_from_landmarks import transform_points, invert_affine_matrix


def launch_clemgui(
    em_stack_file,
    lm_stack_file,
    landmark_file,
    em_affine_file=None,
    lm_scale=(100, 4, 4)
):
    # ----------------- Load Data -----------------
    em_stack = tifffile.imread(em_stack_file)
    lm_stack = tifffile.imread(lm_stack_file)

    if em_affine_file:
        raw_landmark_df = pd.read_csv(landmark_file)
        affine3d_mat = np.loadtxt(em_affine_file)
        transformed_points = transform_points(
            raw_landmark_df[['z_em_downsampled', 'y_em_downsampled', 'x_em_downsampled']].values,
            affine3d_mat
        )

        landmark_df = raw_landmark_df[['landmark_name']].copy()
        landmark_df['x_EM'] = transformed_points[:, 2]
        landmark_df['y_EM'] = transformed_points[:, 1]
        landmark_df['z_EM'] = transformed_points[:, 0]
        landmark_df['x_LM'] = raw_landmark_df['x_lm'] * lm_scale[2]
        landmark_df['y_LM'] = raw_landmark_df['y_lm'] * lm_scale[1]
        landmark_df['z_LM'] = raw_landmark_df['z_lm'] * lm_scale[0]
        landmark_df['tag'] = ''
    else:
        landmark_df = pd.read_csv(landmark_file)

    # ----------------- Setup Napari Viewer -----------------
    viewer = napari.Viewer()
    em_layer = viewer.add_image(em_stack, name='EM Stack')
    lm_layer = viewer.add_image(lm_stack, name='LM Stack', scale=lm_scale)

    em_points_layer = viewer.add_points(name='EM Landmarks', ndim=3, size=20, face_color='red', out_of_slice_display=True)
    lm_points_layer = viewer.add_points(name='LM Landmarks', ndim=3, size=20, face_color='blue', out_of_slice_display=True)

    # ----------------- Landmark Management -----------------
    landmark_manager = LandmarkManager()
    table_widget = LandmarkTableWidget(landmark_manager)
    viewer.window.add_dock_widget(table_widget, area='right')

    def load_landmarks_from_dataframe(df):
        landmark_manager.load_from_dataframe(df)
        table_widget.update_table()
        em_points_layer.data = landmark_manager.get_em_points_zyx()
        lm_points_layer.data = landmark_manager.get_lm_points_zyx()
        em_points_layer.refresh()
        lm_points_layer.refresh()

    load_landmarks_from_dataframe(landmark_df)

    # ----------------- GUI Controls -----------------
    class ControlWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Controls")
            self.layout = QVBoxLayout(self)
            self.save_button = QPushButton("Save Landmarks")
            self.save_button.clicked.connect(lambda: landmark_manager.df.to_csv(landmark_file, index=False))
            self.layout.addWidget(self.save_button)

    control_widget = ControlWidget()
    viewer.window.add_dock_widget(control_widget, area='right')

    # ----------------- Viewer Status State -----------------
    class ViewerStatus:
        def __init__(self):
            self.replace_landmark_idx = None
            self.adding_pair = False
            self.initial_lm_count = 0
            self.initial_em_count = 0

    status = ViewerStatus()

    # ----------------- Utility Functions -----------------
    def tag_selected_landmark(tag):
        rows = table_widget.table.selectionModel().selectedRows()
        if rows:
            idx = rows[0].row()
            landmark_manager.set_tag(idx, tag)
            table_widget.update_table()
            table_widget.table.selectRow(idx)

    def select_next_landmark():
        rows = table_widget.table.selectionModel().selectedRows()
        row_count = table_widget.table.rowCount()
        if row_count == 0:
            return
        current_index = rows[0].row() if rows else -1
        table_widget.table.selectRow((current_index + 1) % row_count)

    def select_previous_landmark():
        rows = table_widget.table.selectionModel().selectedRows()
        row_count = table_widget.table.rowCount()
        if row_count == 0:
            return
        current_index = rows[0].row() if rows else 0
        table_widget.table.selectRow((current_index - 1) % row_count)

    # ----------------- Key Bindings -----------------
    @viewer.bind_key('c')  # confirmed
    def _tag_c(viewer): tag_selected_landmark('confirmed')

    @viewer.bind_key('x')  # questioned
    def _tag_x(viewer): tag_selected_landmark('questioned')

    @viewer.bind_key('d')  # delete
    def _tag_d(viewer): tag_selected_landmark('delete')

    @viewer.bind_key('Down')
    def _next(viewer): select_next_landmark()

    @viewer.bind_key('Up')
    def _prev(viewer): select_previous_landmark()

    @viewer.bind_key('w')
    def jump_to_selected_lm(viewer):
        viewer.layers['LM Stack'].visible = True
        viewer.layers['EM Stack'].visible = False
        if lm_points_layer.selected_data:
            idx = next(iter(lm_points_layer.selected_data))
            point = lm_points_layer.data[idx]
            viewer.camera.center = point
            viewer.dims.set_point(0, point[0])
            viewer.layers.selection.active = lm_points_layer

    @viewer.bind_key('e')
    def jump_to_selected_em(viewer):
        viewer.layers['EM Stack'].visible = True
        viewer.layers['LM Stack'].visible = False
        if not em_points_layer.selected_data:
            if lm_points_layer.selected_data:
                idx = next(iter(lm_points_layer.selected_data))
                em_points_layer.selected_data = {idx}
        if em_points_layer.selected_data:
            idx = next(iter(em_points_layer.selected_data))
            point = em_points_layer.data[idx]
            viewer.camera.center = point
            viewer.dims.set_point(0, point[0])
            viewer.layers.selection.active = em_points_layer

    @viewer.bind_key('t')
    def _toggle_lm(viewer): lm_points_layer.visible = not lm_points_layer.visible

    @viewer.bind_key('y')
    def _toggle_em(viewer): em_points_layer.visible = not em_points_layer.visible

    # ----------------- Sync Callbacks -----------------
    def on_lm_selection_pair_em(event):
        selected = lm_points_layer.selected_data
        em_points_layer.selected_data = {next(iter(selected))} if len(selected) == 1 else set()

    def on_lm_selection_update_table(event):
        selected = lm_points_layer.selected_data
        table_widget.table.itemSelectionChanged.disconnect(on_table_selection)
        if len(selected) == 1:
            idx = next(iter(selected))
            table_widget.table.selectRow(idx)
        else:
            table_widget.table.clearSelection()
        table_widget.table.itemSelectionChanged.connect(on_table_selection)

    def on_table_selection():
        rows = table_widget.table.selectionModel().selectedRows()
        lm_points_layer.selected_data.events.items_changed.disconnect(on_lm_selection_update_table)
        if rows:
            lm_points_layer.selected_data = {rows[0].row()}
        else:
            lm_points_layer.selected_data.clear()
        lm_points_layer.selected_data.events.items_changed.connect(on_lm_selection_update_table)

    lm_points_layer.selected_data.events.items_changed.connect(on_lm_selection_pair_em)
    lm_points_layer.selected_data.events.items_changed.connect(on_lm_selection_update_table)
    table_widget.table.itemSelectionChanged.connect(on_table_selection)

    # ----------------- Pair Addition -----------------
    def on_lm_added(event):
        if status.adding_pair:
            viewer.layers.selection.active = em_points_layer
            em_points_layer.mode = 'add'
            em_points_layer.events.data.connect(on_em_added)

    def on_em_added(event):
        if not status.adding_pair:
            return

        def finalize_em_point():
            if len(em_points_layer.data) <= status.initial_em_count:
                return
            new_em = em_points_layer.data[-1]
            new_lm = lm_points_layer.data[-1]
            landmark_manager.add_pair(new_em, new_lm)
            table_widget.update_table()
            status.adding_pair = False
            em_points_layer.mode = 'pan_zoom'
            lm_points_layer.mode = 'pan_zoom'
            reconnect_selection()
            em_points_layer.events.data.disconnect(on_em_added)

        QTimer.singleShot(0, finalize_em_point)

    def reconnect_selection():
        lm_points_layer.selected_data.events.items_changed.connect(on_lm_selection_pair_em)
        lm_points_layer.selected_data.events.items_changed.connect(on_lm_selection_update_table)

    @viewer.bind_key('Space')
    def _start_add_pair(viewer):
        status.adding_pair = True
        status.initial_em_count = len(em_points_layer.data)
        status.initial_lm_count = len(lm_points_layer.data)
        lm_points_layer.mode = 'add'
        lm_points_layer.events.data.connect(on_lm_added)
        lm_points_layer.selected_data.events.items_changed.disconnect(on_lm_selection_pair_em)
        lm_points_layer.selected_data.events.items_changed.disconnect(on_lm_selection_update_table)

    @viewer.bind_key('Escape')
    def _cancel_add(viewer):
        if status.adding_pair:
            em_points_layer.data = em_points_layer.data[:status.initial_em_count]
            lm_points_layer.data = lm_points_layer.data[:status.initial_lm_count]
            em_points_layer.mode = 'pan_zoom'
            lm_points_layer.mode = 'pan_zoom'
            status.adding_pair = False
            reconnect_selection()

    napari.run()
