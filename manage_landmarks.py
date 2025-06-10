import re
import pandas as pd
from qtpy.QtWidgets import (
    QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView
)

class LandmarkManager:
    def __init__(self):
        # Define the DataFrame structure explicitly
        self.columns = [
            "landmark_name", "tag",
            "x_EM", "y_EM", "z_EM",
            "x_LM", "y_LM", "z_LM"
        ]
        # Initialize an empty DataFrame
        self.df = pd.DataFrame(columns=self.columns)
        self.max_id = self._get_max_id()


    def _get_max_id(self):
        if self.df.empty:
            return 0
        landmark_names = self.df["landmark_name"].dropna().tolist()
        numbers = [int(s.split('-')[1]) for s in landmark_names]
        max_id = max(numbers)
        return max_id


    def get_new_landmark_name(self):
        landmark_name = "landmark-{}".format(self.max_id + 1)
        self.max_id += 1
        return landmark_name


    def add_pair(self, em_coords, lm_coords, tag=""):
        landmark_name = self.get_new_landmark_name()
        new_row = {
            "landmark_name": landmark_name,
            "tag": tag,
            "x_EM": em_coords[2],
            "y_EM": em_coords[1],
            "z_EM": em_coords[0],
            "x_LM": lm_coords[2],
            "y_LM": lm_coords[1],
            "z_LM": lm_coords[0],
        }
        self.df.loc[len(self.df)] = new_row

    def clear_pairs(self):
        self.df = self.df.iloc[0:0]


    def update_landmark_coords(self, index, lm_coords=None, em_coords=None):
        if em_coords is not None:
            self.df.loc[index, ["z_EM", "y_EM", "x_EM"]] = em_coords
        if lm_coords is not None:
            self.df.loc[index, ["z_LM", "y_LM", "x_LM"]] = lm_coords


    def set_tag(self, index, tag):
        self.df.at[index, "tag"] = tag


    def get_landmark_name(self, index):
        return self.df.at[index, "landmark_name"]


    def set_landmark_name(self, index, name):
        self.df.at[index, "landmark_name"] = name


    def to_dataframe(self):
        return self.df.copy()


    def load_from_dataframe(self, df):
        self.df = df[self.columns].copy()
        self.max_id = self._get_max_id()
    

    def get_em_points_zyx(self):
        return self.df[['z_EM', 'y_EM', 'x_EM']].dropna().to_numpy(dtype=float)

    def get_lm_points_zyx(self):
        return self.df[['z_LM', 'y_LM', 'x_LM']].dropna().to_numpy(dtype=float)
    
    def set_tag(self, index, tag):
        self.df.at[index, "tag"] = tag



# --- Create a GUI widget with a table to display landmark pairs ---
class LandmarkTableWidget(QWidget):
    def __init__(self, landmark_manager: LandmarkManager):
        super().__init__()
        self.landmark_manager = landmark_manager

        self.setWindowTitle("Landmark Pairs")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.landmark_manager.columns))
        self.table.setHorizontalHeaderLabels(self.landmark_manager.columns)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        self.table.setEditTriggers(QTableWidget.NoEditTriggers)


        self.layout.addWidget(self.table)
        self.update_table()

    def update_table(self):
        df = self.landmark_manager.df
        self.table.setRowCount(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(self.landmark_manager.columns):
                value = row[col] if pd.notna(row[col]) else ""
                item = QTableWidgetItem(str(value))
                self.table.setItem(i, j, item)