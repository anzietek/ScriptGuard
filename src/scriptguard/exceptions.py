class DataIngestionError(Exception):
    pass


class TokenizationError(Exception):
    pass


class ModelTrainingError(Exception):
    pass


class InferenceError(Exception):
    pass


class ModelRegistrationError(Exception):
    pass
