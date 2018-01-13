"""
A base for all the models defined in our networks to inhert from. It will include all the methods common accross all models.
"""
class Params:
    """
    Empty class to Hold all specified paramters for any Model at runtime
    """
    pass


class BaseModel:
    """
    Base class to be inherited from any model
    """

    def __init__(self, flags):
        """
        Just Creating a params to be used in any MODEL
        """
        self.params = Params()
        self.flags = flags

    def init_input(self):
        raise NotImplementedError("Init input is not Implemented in the model")

    def init_network(self):
        raise NotImplementedError("Init network is not implemented in the model")

    def init_output(self):
        raise NotImplementedError("Init network is not implemented in the model")