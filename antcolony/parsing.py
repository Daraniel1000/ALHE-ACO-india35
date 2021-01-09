import networkx as nx


def until(string, sep):
    """Return string from beginning to first occurrence of sep (exclusive)"""
    return string.partition(sep)[0]


def runtil(string, sep):
    """Return string from beginning to last occurrence of sep (exclusive)"""
    return string.rpartition(sep)[0]


def after(string, sep):
    """Return string from first occurrence of sep (exclusive) to end"""
    return string.partition(sep)[2]


def rafter(string, sep):
    """Return string from last occurrence of sep (exclusive) to end"""
    return string.rpartition(sep)[2]


def between(string, sep1, sep2):
    """Return string between first occurrence of sep1 (exclusive) and first occurrence of sep2 (exclusive) after that"""
    return until(after(string, sep1), sep2)


def rbetween(string, sep1, sep2):
    """Return string between last occurrence of sep1 (exclusive) and first occurrence of sep2 (exclusive) after that"""
    return until(rafter(string, sep1), sep2)


def rrbetween(string, sep1, sep2):
    """Return string between last occurrence of sep1 (exclusive) and last occurrence of sep2 (exclusive) after that"""
    return runtil(rafter(string, sep1), sep2)


def parse_sndlib_weighted(content, weight_label="weight"):
    """Return NetworkX weighted graph parsed from string with graph definition"""

    def get_section(content, title):
        return between(after(content, title), "(\n", "\n)")

    def parse_node(line):
        return until(line, "(").strip()

    def parse_edge(line):
        node_a, node_b = tuple(between(line, "(", ")").strip().split())
        weight = float(rbetween(line, "(", ")").strip().split()[1])
        return node_a, node_b, weight

    nodes = [parse_node(line) for line in get_section(content, "NODES").splitlines()]
    edges = [parse_edge(line) for line in get_section(content, "LINKS").splitlines()]

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_weighted_edges_from(edges, weight=weight_label)
    return g
