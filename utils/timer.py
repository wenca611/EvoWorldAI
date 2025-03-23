import time


class Timer:
    """

    """
    def __init__(self, function_name=None):
        """

        :param function_name:
        """
        self.function_name = str(function_name)

    def __enter__(self):
        """

        :return:
        """
        """Metoda pro začátek měření času."""
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        """Metoda pro konec měření času a výpis výsledků."""
        end_time = time.time()
        exec_time = end_time - self.start_time
        name = self.function_name if self.function_name else "Blok kódu"
        print(f"{name} trval(a) {exec_time:.4f} sekund.")