class Snapshot_control:
    _instance = None

    @staticmethod
    def get_instance():
        if Snapshot_control._instance is None:
            Snapshot_control._instance = Snapshot_control()
        return Snapshot_control._instance

    def __init__(self):
        if Snapshot_control._instance is not None:
            raise Exception("Esta clase es un singleton y ya fue instanciada.")
        self.num_snapshots = 0
        # Añade más parámetros según sea necesario
