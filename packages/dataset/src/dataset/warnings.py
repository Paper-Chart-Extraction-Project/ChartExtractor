"""A module containing generic warnings for dataset issues."""


class DatasetWarning(UserWarning):
    """A generic warning regarding dataset issues."""


class EmptyDatasetWarning(DatasetWarning):
    """A warning issued when a dataset is totally empty."""


class EmptyEvaluationSetWarning(DatasetWarning):
    """A warning issued when a dataset's optional evaluation set is empty."""


class EmptySplitWarning(DatasetWarning):
    """A warning issued when a dataset split is empty."""
