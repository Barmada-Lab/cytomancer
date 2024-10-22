import enum
from functools import total_ordering


@total_ordering
class NaturalOrderStrEnum(str, enum.Enum):
    """
    Provides a total ordering for enum members by caching insertion order
    """

    def __init__(self, *args):
        try:
            super().__init__(*args)
        except TypeError:
            # there are no other parents
            pass

        self._order = len(self.__class__.__members__) + 1

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self._order > other._order
        return NotImplemented

    def __lt__(self, other):
        return not self.__gt__(other)

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self._order == other._order
        return NotImplemented

    def __hash__(self):
        return id(self)
