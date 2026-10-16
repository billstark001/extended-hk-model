"""Study colors, labels and exact parameter selection; generic style lives in common.plotting."""

COLORS = ("#555555", "#86b5d8", "#166da8", "#eba981", "#bd4e19")
LABELS = ("Random", "Opinion1", "Opinion4", "Structure1", "Structure4")
RULES = ("hk", "deffuant")


def matches(row, **values):
    return all(row[k] == v for k, v in values.items())
