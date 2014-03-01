class Serializable(object):

    @classmethod
    def load_model(cls, raw):
        model = cls()
        model.load(raw)
        return model

    @staticmethod
    def dump_model(model):
        return model.dump()

    @classmethod
    def load_group(cls, raw):
        group = cls.Group()
        group.load(raw)
        return group

    @staticmethod
    def dump_group(group):
        return group.dump()
