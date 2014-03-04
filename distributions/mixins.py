class ComponentModel(object):

    def group_create(self, values=[]):
        group = self.Group()
        self.group_init(group)
        for value in values:
            self.group_add_value(group, value)
        return group


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
