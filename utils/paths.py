import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent

class Paths:
    def __init__(self):
        self.base = BASE_DIR

    @staticmethod
    def base_dir():
        return BASE_DIR

    @staticmethod
    def data_dir():
        return BASE_DIR / "data"
    
    @staticmethod
    def data_raw():
        return Paths.data_dir() / "raw"

    @staticmethod
    def data_raw_arc1():
        return Paths.data_raw() / "1. BatteryAgingARC-FY08Q4"

    @staticmethod
    def data_processed():
        return Paths.data_dir() / "processed"

    @staticmethod
    def experiments_dir():
        return BASE_DIR / "pinn_experiment_results"